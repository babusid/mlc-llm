/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/medusa_new_request_prefill.cc
 */

#include "../sampler/sampler.h"
#include "batch_prefill_base.h"

namespace mlc {
namespace llm {
namespace serve {

class MedusaNewRequestPrefillActionObj : public BatchPrefillBaseActionObj {
 public:
  explicit MedusaNewRequestPrefillActionObj(Array<Model> models, LogitProcessor logit_processor,
                                            Sampler sampler,
                                            std::vector<ModelWorkspace> model_workspaces,
                                            DraftTokenWorkspaceManager draft_token_workspace_manager,
                                            EngineConfig engine_config,
                                            std::vector<tvm::ffi::json::Object> model_configs,
                                            Optional<EventTraceRecorder> trace_recorder)
      : BatchPrefillBaseActionObj(std::move(models), std::move(engine_config),
                                  std::move(model_configs), std::move(trace_recorder)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        model_workspaces_(std::move(model_workspaces)),
        draft_token_workspace_manager_(std::move(draft_token_workspace_manager)) {}

  Array<Request> Step(EngineState estate) final {
    std::vector<PrefillInput> prefill_inputs;
    {
      NVTXScopedRange nvtx_scope("NewRequestPrefill getting requests");
      prefill_inputs = GetRequestStateEntriesToPrefill(estate);
      if (prefill_inputs.empty()) {
        return {};
      }
    }

    int num_rsentries = prefill_inputs.size();
    {
      NVTXScopedRange nvtx_scope("NewRequestPrefill matching prefix");
      for (int i = 0; i < num_rsentries; ++i) {
        MatchPrefixCache(estate, &prefill_inputs[i]);
      }
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    Array<String> request_ids;
    std::vector<RequestState> rstates_of_entries;
    std::vector<RequestStateStatus> status_before_prefill;
    UpdateRequestToAlive(prefill_inputs, estate, &request_ids, &rstates_of_entries,
                         &status_before_prefill);

    std::vector<int> prefill_lengths;
    prefill_lengths.resize(/*size=*/num_rsentries, /*value=*/-1);
    ObjectRef hidden_states_for_sample{nullptr};
    Tensor logits_for_sample{nullptr};
    std::unordered_map<int, std::unordered_set<int>> fork_rsentry_child_map;

    for (int model_id = 0; model_id < static_cast<int>(models_.size()); ++model_id) {
      std::vector<int64_t> request_internal_ids;
      request_internal_ids.reserve(num_rsentries);

      Array<Tensor> multi_step_logits{nullptr};
      if (model_id > 0) {
        multi_step_logits = models_[model_id]->GetMultiStepLogits(hidden_states_for_sample);
      } else {
        ObjectRef embeddings = model_workspaces_[model_id].embeddings;
        int cum_prefill_length = 0;
        bool single_input =
            num_rsentries == 1 && prefill_inputs[0].rsentry->mstates[model_id]->inputs.size() == 1;
        for (int i = 0; i < num_rsentries; ++i) {
          const RequestStateEntry& rsentry = prefill_inputs[i].rsentry;
          RequestModelState mstate = rsentry->mstates[model_id];
          TVM_FFI_ICHECK(mstate->draft_output_tokens.empty());
          TVM_FFI_ICHECK(mstate->draft_token_slots.empty());
          if (status_before_prefill[i] == RequestStateStatus::kPending) {
            if (!estate->prefix_cache->HasSequence(mstate->internal_id)) {
              if (rsentry->parent_idx == -1) {
                models_[model_id]->AddNewSequence(mstate->internal_id);
              } else {
                models_[model_id]->ForkSequence(rstates_of_entries[i]
                                                    ->entries[rsentry->parent_idx]
                                                    ->mstates[model_id]
                                                    ->internal_id,
                                                mstate->internal_id);
              }
            }
            if (rsentry->child_indices.empty()) {
              models_[model_id]->EnableSlidingWindowForSeq(mstate->internal_id);
            }
          }
          request_internal_ids.push_back(mstate->internal_id);

          auto [input_data, input_length] =
              ChunkPrefillInputData(mstate, prefill_inputs[i].max_prefill_length);
          if (prefill_lengths[i] == -1) {
            prefill_lengths[i] = input_length;
          } else {
            TVM_FFI_ICHECK_EQ(prefill_lengths[i], input_length);
          }
          mstate->num_prefilled_tokens += input_length;

          RECORD_EVENT(trace_recorder_, prefill_inputs[i].rsentry->request->id, "start embedding");
          for (int j = 0; j < static_cast<int>(input_data.size()); ++j) {
            mstate->prefilled_inputs.push_back(input_data[j]);
            embeddings = input_data[j]->GetEmbedding(
                models_[model_id], /*dst=*/!single_input ? &model_workspaces_[model_id].embeddings
                                                          : nullptr,
                /*offset=*/cum_prefill_length);
            cum_prefill_length += input_data[j]->GetLength();
          }
          RECORD_EVENT(trace_recorder_, rsentry->request->id, "finish embedding");
        }

        RECORD_EVENT(trace_recorder_, request_ids, "start prefill");
        ObjectRef hidden_states = models_[model_id]->BatchPrefillToLastHidden(
            embeddings, request_internal_ids, prefill_lengths);
        RECORD_EVENT(trace_recorder_, request_ids, "finish prefill");

        estate->prefix_cache->CommitSequenceExtention();

        int sample_model_id = !models_[model_id]->CanGetLogits() ? 0 : model_id;
        std::vector<int> logit_positions;
        logit_positions.reserve(prefill_lengths.size());
        int total_len = 0;
        for (int i = 0; i < prefill_lengths.size(); ++i) {
          total_len += prefill_lengths[i];
          logit_positions.push_back(total_len - 1);
        }
        hidden_states_for_sample = models_[sample_model_id]->GatherHiddenStates(
            hidden_states, logit_positions, &model_workspaces_[model_id].hidden_states);
        logits_for_sample = models_[sample_model_id]->GetLogits(hidden_states_for_sample);
      }

      Array<String> child_request_ids;
      std::vector<int> child_sample_indices;
      std::vector<RequestStateEntry> rsentries_for_sample;
      std::vector<RandomGenerator*> rngs;
      std::vector<bool> rsentry_activated;
      Array<GenerationConfig> child_generation_cfg;
      child_sample_indices.reserve(num_rsentries);
      child_generation_cfg.reserve(num_rsentries);
      child_request_ids.reserve(num_rsentries);
      rsentries_for_sample.reserve(num_rsentries);
      rngs.reserve(num_rsentries);
      rsentry_activated.reserve(num_rsentries);

      for (int i = 0; i < num_rsentries; ++i) {
        const RequestStateEntry& rsentry = prefill_inputs[i].rsentry;
        if (!rsentry->mstates[0]->inputs.empty()) {
          continue;
        }

        int remaining_num_child_to_activate = prefill_inputs[i].num_child_to_activate;
        for (int child_idx : rsentry->child_indices) {
          if ((rstates_of_entries[i]->entries[child_idx]->status == RequestStateStatus::kPending &&
                   rstates_of_entries[i]
                       ->entries[child_idx]
                       ->mstates[0]
                       ->committed_tokens.empty() ||
               fork_rsentry_child_map[i].count(child_idx))) {
            fork_rsentry_child_map[i].insert(child_idx);
            child_sample_indices.push_back(i);
            rsentries_for_sample.push_back(rstates_of_entries[i]->entries[child_idx]);
            child_request_ids.push_back(rsentry->request->id);
            child_generation_cfg.push_back(rsentry->request->generation_cfg);
            rngs.push_back(&rstates_of_entries[i]->entries[child_idx]->rng);

            if (remaining_num_child_to_activate == 0) {
              rsentry_activated.push_back(false);
              continue;
            }
            rsentry_activated.push_back(true);
            --remaining_num_child_to_activate;
            if (model_id == 0) {
              TVM_FFI_ICHECK(rstates_of_entries[i]->entries[child_idx]->status ==
                             RequestStateStatus::kPending);
              rstates_of_entries[i]->entries[child_idx]->status = RequestStateStatus::kAlive;
            }
            int64_t child_internal_id =
                rstates_of_entries[i]->entries[child_idx]->mstates[model_id]->internal_id;
            models_[model_id]->ForkSequence(rsentry->mstates[model_id]->internal_id,
                                            child_internal_id);
            if (rstates_of_entries[i]->entries[child_idx]->child_indices.empty()) {
              models_[model_id]->EnableSlidingWindowForSeq(child_internal_id);
            }
          }
        }
        if (rsentry->child_indices.empty()) {
          child_sample_indices.push_back(i);
          rsentries_for_sample.push_back(rsentry);
          child_request_ids.push_back(rsentry->request->id);
          child_generation_cfg.push_back(rsentry->request->generation_cfg);
          rngs.push_back(&rsentry->rng);
          rsentry_activated.push_back(true);
        }
      }

      Array<GenerationConfig> generation_cfg;
      Array<RequestModelState> mstates_for_logitproc;
      std::vector<int> sample_indices(num_rsentries);
      generation_cfg.reserve(num_rsentries);
      mstates_for_logitproc.reserve(num_rsentries);
      std::iota(sample_indices.begin(), sample_indices.end(), 0);
      for (int i = 0; i < num_rsentries; ++i) {
        generation_cfg.push_back(prefill_inputs[i].rsentry->request->generation_cfg);
        mstates_for_logitproc.push_back(prefill_inputs[i].rsentry->mstates[model_id]);
      }

      if (model_id == 0) {
        TVM_FFI_ICHECK(logits_for_sample.defined());
        const auto& [renormalized_probs, sample_results] = ApplyLogitProcessorAndSample(
            logit_processor_, sampler_, logits_for_sample, generation_cfg, request_ids,
            mstates_for_logitproc, rngs, sample_indices, child_generation_cfg, child_request_ids,
            child_sample_indices);
        UpdateRequestStateEntriesWithSampleResults(rsentries_for_sample, rsentry_activated,
                                                   sample_results);
      } else {
        TVM_FFI_ICHECK_NE(estate->spec_draft_length, 0);
        for (int draft_id = 0; draft_id < estate->spec_draft_length; ++draft_id) {
          const auto& [renormalized_probs, sample_results] = ApplyLogitProcessorAndSample(
              logit_processor_, sampler_, multi_step_logits[draft_id], generation_cfg, request_ids,
              mstates_for_logitproc, rngs, sample_indices, child_generation_cfg, child_request_ids,
              child_sample_indices);
          UpdateRequestStatesWithDraftProposals(
              rsentries_for_sample, sample_results, model_id, renormalized_probs,
              /*hidden_states=*/ObjectRef{nullptr}, estate, child_sample_indices);
        }
      }
    }

    auto tend = std::chrono::high_resolution_clock::now();
    estate->metrics.engine_prefill_time_sum += static_cast<double>((tend - tstart).count()) / 1e9;

    std::vector<Request> processed_requests =
        RemoveProcessedRequests(prefill_inputs, estate, rstates_of_entries);
    estate->running_rsentries_changed = true;
    return processed_requests;
  }

  void UpdateRequestStatesWithDraftProposals(
      const std::vector<RequestStateEntry>& rsentries_for_sample,
      const std::vector<SampleResult>& sample_results, int model_id,
      const Tensor& renormalized_probs, const ObjectRef& hidden_states_for_sample,
      EngineState estate, const std::vector<int>& sample_indices) {
    (void)hidden_states_for_sample;
    (void)estate;
    std::vector<int> reuse_count(renormalized_probs->shape[0], 0);
    for (int i = 0; i < static_cast<int>(sample_indices.size()); ++i) {
      reuse_count[sample_indices[i]]++;
    }
    draft_token_workspace_manager_->AllocSlots(renormalized_probs->shape[0], reuse_count,
                                               &draft_token_slots_);

    models_[0]->ScatterDraftProbs(renormalized_probs, draft_token_slots_,
                                  &model_workspaces_[0].draft_probs_storage);
    for (int i = 0; i < static_cast<int>(rsentries_for_sample.size()); ++i) {
      int parent_idx =
          rsentries_for_sample[i]->mstates[model_id]->draft_output_tokens.empty()
              ? -1
              : rsentries_for_sample[i]->mstates[model_id]->draft_output_tokens.size() - 1;
      rsentries_for_sample[i]->mstates[model_id]->AddDraftToken(
          sample_results[i], draft_token_slots_[sample_indices[i]], parent_idx);
    }
  }

 private:
  LogitProcessor logit_processor_;
  Sampler sampler_;
  std::vector<ModelWorkspace> model_workspaces_;
  DraftTokenWorkspaceManager draft_token_workspace_manager_;
  std::vector<int> draft_token_slots_;

  int MatchPrefixCache(EngineState estate, PrefillInput* input) final {
    RequestStateEntry rsentry = input->rsentry;
    if (estate->prefix_cache->Mode() == PrefixCacheMode::kDisable) {
      return 0;
    }
    if (rsentry->parent_idx == -1 && rsentry->status == RequestStateStatus::kPending &&
        !estate->prefix_cache->HasSequence(rsentry->mstates[0]->internal_id)) {
      std::vector<int32_t> tokens = GetConcatPrefillInputData(rsentry->mstates[0]);
      if (tokens.empty()) {
        return 0;
      }
      PrefixCacheMatchedResult result = estate->prefix_cache->InsertSequence(
          rsentry->mstates[0]->internal_id, tokens, models_[0]->GetSlidingWindowSize(),
          models_[0]->GetAttentionSinkSize());
      if (result.prefilled_offset == 0) {
        TVM_FFI_ICHECK_EQ(result.forked_seq_id, -1);
        TVM_FFI_ICHECK_EQ(result.reused_seq_id, -1);
        TVM_FFI_ICHECK_EQ(result.reused_seq_pop_last_tokens, 0);
        for (int i = 0; i < models_.size(); ++i) {
          models_[i]->AddNewSequence(rsentry->mstates[0]->internal_id);
          if (rsentry->child_indices.empty()) {
            models_[i]->EnableSlidingWindowForSeq(rsentry->mstates[0]->internal_id);
          }
        }
      } else {
        if (result.forked_seq_id != -1) {
          TVM_FFI_ICHECK_EQ(result.reused_seq_id, -1);
          TVM_FFI_ICHECK_EQ(result.reused_seq_pop_last_tokens, 0);
          estate->prefix_cache->RollBackSequence(rsentry->mstates[0]->internal_id, 1);
          for (int i = 0; i < models_.size(); ++i) {
            models_[i]->ForkSequence(result.forked_seq_id, rsentry->mstates[0]->internal_id,
                                     result.prefilled_offset - 1);
            if (rsentry->child_indices.empty()) {
              models_[i]->EnableSlidingWindowForSeq(rsentry->mstates[0]->internal_id);
            }
          }
        } else {
          TVM_FFI_ICHECK_EQ(result.forked_seq_id, -1);
          estate->id_manager.RecycleId(rsentry->mstates[0]->internal_id);
          for (int i = 0; i < rsentry->mstates.size(); ++i) {
            rsentry->mstates[i]->internal_id = result.reused_seq_id;
          }
          estate->prefix_cache->RollBackSequence(rsentry->mstates[0]->internal_id, 1);
          for (int i = 0; i < models_.size(); ++i) {
            models_[i]->PopNFromKVCache(rsentry->mstates[0]->internal_id,
                                        result.reused_seq_pop_last_tokens + 1);
          }
          result.prefilled_offset -= 1;
        }
      }
      if (result.prefilled_offset > 0) {
        for (int i = 0; i < rsentry->mstates.size(); ++i) {
          PopPrefillInputData(rsentry->mstates[i], result.prefilled_offset);
        }
      }
      input->max_prefill_length =
          std::min(input->max_prefill_length, rsentry->mstates[0]->GetInputLength());
      return result.prefilled_offset - 1;
    }
    return 0;
  }
};

EngineAction EngineAction::MedusaNewRequestPrefill(
    Array<Model> models, LogitProcessor logit_processor, Sampler sampler,
    std::vector<ModelWorkspace> model_workspaces,
    DraftTokenWorkspaceManager draft_token_workspace_manager, EngineConfig engine_config,
    std::vector<tvm::ffi::json::Object> model_configs,
    Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(tvm::ffi::make_object<MedusaNewRequestPrefillActionObj>(
      std::move(models), std::move(logit_processor), std::move(sampler),
      std::move(model_workspaces), std::move(draft_token_workspace_manager),
      std::move(engine_config), std::move(model_configs), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
