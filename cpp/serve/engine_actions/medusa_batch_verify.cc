/*!
 *  Copyright (c) 2023-2025 by Contributors
 * \file serve/engine_actions/medusa_batch_verify.cc
 */

#include <cmath>
#include <exception>
#include <numeric>

#include "../../support/random.h"
#include "../config.h"
#include "../model.h"
#include "../sampler/sampler.h"
#include "action.h"
#include "action_commons.h"

namespace mlc {
namespace llm {
namespace serve {

class MedusaBatchVerifyActionObj : public EngineActionObj {
 public:
  explicit MedusaBatchVerifyActionObj(Array<Model> models, LogitProcessor logit_processor,
                                      Sampler sampler, std::vector<ModelWorkspace> model_workspaces,
                                      DraftTokenWorkspaceManager draft_token_workspace_manager,
                                      EngineConfig engine_config,
                                      Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        model_workspaces_(std::move(model_workspaces)),
        draft_token_workspace_manager_(std::move(draft_token_workspace_manager)),
        engine_config_(std::move(engine_config)),
        trace_recorder_(std::move(trace_recorder)),
        rng_(RandomGenerator::GetInstance()) {}

  Array<Request> Step(EngineState estate) final {
    if (models_.size() != 2 || estate->running_queue.empty()) {
      return {};
    }

    const auto& [rsentries, draft_lengths, total_draft_length] = GetDraftsToVerify(estate);
    TVM_FFI_ICHECK_EQ(rsentries.size(), draft_lengths.size());
    if (rsentries.empty()) {
      return {};
    }

    auto tstart = std::chrono::high_resolution_clock::now();
    int num_rsentries = rsentries.size();
    Array<String> request_ids =
        rsentries.Map([](const RequestStateEntry& rstate) { return rstate->request->id; });

    std::vector<int64_t> request_internal_ids;
    std::vector<int32_t> all_tokens_to_verify;
    Array<RequestModelState> verify_request_mstates;
    Array<RequestModelState> draft_request_mstates;
    Array<GenerationConfig> generation_cfg;
    std::vector<RandomGenerator*> rngs;
    std::vector<std::vector<SampleResult>> draft_output_tokens;
    std::vector<std::vector<int>> draft_token_indices;
    std::vector<int64_t> token_tree_parent_ptr;
    request_internal_ids.reserve(num_rsentries);
    all_tokens_to_verify.reserve(total_draft_length);
    token_tree_parent_ptr.reserve(total_draft_length);
    verify_request_mstates.reserve(num_rsentries);
    draft_request_mstates.reserve(num_rsentries);
    rngs.reserve(num_rsentries);
    generation_cfg.reserve(num_rsentries);
    draft_output_tokens.reserve(num_rsentries);
    draft_token_indices.reserve(num_rsentries);
    draft_token_slots_.clear();

    for (int i = 0; i < num_rsentries; ++i) {
      RequestModelState verify_mstate = rsentries[i]->mstates[verify_model_id_];
      RequestModelState draft_mstate = rsentries[i]->mstates[draft_model_id_];
      request_internal_ids.push_back(verify_mstate->internal_id);
      TVM_FFI_ICHECK(!draft_lengths.empty());
      TVM_FFI_ICHECK_EQ(draft_lengths[i], draft_mstate->draft_output_tokens.size());
      TVM_FFI_ICHECK_EQ(draft_lengths[i], draft_mstate->draft_token_slots.size());
      all_tokens_to_verify.push_back(draft_mstate->committed_tokens.back().GetTokenId());
      draft_token_slots_.push_back(0);
      token_tree_parent_ptr.push_back(-1);

      for (int j = 0; j < static_cast<int>(draft_mstate->draft_output_tokens.size()); ++j) {
        all_tokens_to_verify.push_back(draft_mstate->draft_output_tokens[j].GetTokenId());
        draft_token_slots_.push_back(draft_mstate->draft_token_slots[j]);
        token_tree_parent_ptr.push_back(draft_mstate->draft_token_parent_idx[j] + 1);
      }
      std::vector<int> cur_draft_token_indices(draft_mstate->draft_output_tokens.size() + 1);
      std::iota(cur_draft_token_indices.begin(), cur_draft_token_indices.end(), -1);
      draft_token_indices.emplace_back(std::move(cur_draft_token_indices));
      verify_request_mstates.push_back(verify_mstate);
      draft_request_mstates.push_back(draft_mstate);
      generation_cfg.push_back(rsentries[i]->request->generation_cfg);
      rngs.push_back(&rsentries[i]->rng);
      draft_output_tokens.push_back(draft_mstate->draft_output_tokens);
    }

    Tensor draft_probs_on_device = models_[draft_model_id_]->GatherDraftProbs(
        model_workspaces_[verify_model_id_].draft_probs_storage, draft_token_slots_,
        &model_workspaces_[verify_model_id_].draft_probs);

    std::vector<int> cum_verify_lengths = {0};
    cum_verify_lengths.reserve(num_rsentries + 1);
    std::vector<int> verify_lengths;
    for (int i = 0; i < num_rsentries; ++i) {
      verify_lengths.push_back(draft_lengths[i] + 1);
      cum_verify_lengths.push_back(cum_verify_lengths.back() + verify_lengths.back());
    }

    RECORD_EVENT(trace_recorder_, request_ids, "start verify embedding");
    ObjectRef embeddings = models_[verify_model_id_]->TokenEmbed(
        {Shape{all_tokens_to_verify.begin(), all_tokens_to_verify.end()}});
    RECORD_EVENT(trace_recorder_, request_ids, "finish verify embedding");

    RECORD_EVENT(trace_recorder_, request_ids, "start verify");
    ObjectRef hidden_states = models_[verify_model_id_]->BatchVerifyToLastHidden(
        embeddings, request_internal_ids, verify_lengths, token_tree_parent_ptr);
    Tensor logits = models_[verify_model_id_]->GetLogits(hidden_states);
    RECORD_EVENT(trace_recorder_, request_ids, "finish verify");
    TVM_FFI_ICHECK_EQ(logits->ndim, 2);
    TVM_FFI_ICHECK_EQ(logits->shape[0], cum_verify_lengths.back());

    logit_processor_->InplaceUpdateLogits(logits, generation_cfg, verify_request_mstates,
                                          request_ids, &cum_verify_lengths, &draft_request_mstates,
                                          &draft_token_indices);

    Tensor probs_on_device = logit_processor_->ComputeProbsFromLogits(
        logits, generation_cfg, request_ids, &cum_verify_lengths);

    estate->prefix_cache->CommitSequenceExtention();

    std::vector<int> sample_indices(num_rsentries);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    Tensor renormalized_probs = sampler_->BatchRenormalizeProbsByTopP(
        probs_on_device, sample_indices, request_ids, generation_cfg);
    auto [sample_results_arr, _] = sampler_->BatchVerifyDraftTokensWithProbAfterTopP(
        renormalized_probs, request_ids, cum_verify_lengths, generation_cfg, rngs,
        draft_output_tokens, token_tree_parent_ptr, draft_probs_on_device);
    TVM_FFI_ICHECK_EQ(sample_results_arr.size(), num_rsentries);

    std::vector<int64_t> verify_model_seq_internal_ids;
    std::vector<int64_t> accepted_token_tree_leaf_nodes;
    verify_model_seq_internal_ids.reserve(num_rsentries);
    accepted_token_tree_leaf_nodes.reserve(num_rsentries);
    std::vector<int> last_accepted_hidden_positions;
    last_accepted_hidden_positions.reserve(num_rsentries);

    for (int i = 0; i < num_rsentries; ++i) {
      const std::vector<SampleResult>& sample_results = sample_results_arr[i];
      int accept_length = sample_results.size();
      TVM_FFI_ICHECK_GE(accept_length, 1);
      for (SampleResult sample_result : sample_results) {
        rsentries[i]->mstates[verify_model_id_]->CommitToken(sample_result);
        rsentries[i]->mstates[draft_model_id_]->CommitToken(sample_result);
      }
      rsentries[i]->rstate->metrics.completion_tokens += accept_length;
      rsentries[i]->rstate->metrics.decode_tokens += accept_length;
      estate->metrics.spec_decode.Update(cum_verify_lengths[i + 1] - cum_verify_lengths[i],
                                         accept_length);
      int rollback_length =
          std::max(cum_verify_lengths[i + 1] - cum_verify_lengths[i] - accept_length, 0);

      verify_model_seq_internal_ids.push_back(rsentries[i]->mstates[verify_model_id_]->internal_id);
      accepted_token_tree_leaf_nodes.push_back(accept_length - 1);
      if (rollback_length > 0) {
        models_[draft_model_id_]->PopNFromKVCache(
            rsentries[i]->mstates[draft_model_id_]->internal_id, rollback_length - 1);
      }
      rsentries[i]->mstates[draft_model_id_]->RemoveAllDraftTokens(&draft_token_slots_);
      draft_token_workspace_manager_->FreeSlots(draft_token_slots_);
      last_accepted_hidden_positions.push_back(cum_verify_lengths[i] + accept_length - 1);
    }

    models_[verify_model_id_]->CommitAcceptedTokenTreeNodesToKVCache(
        verify_model_seq_internal_ids, accepted_token_tree_leaf_nodes);

    {
      hidden_states = models_[0]->GatherHiddenStates(hidden_states, last_accepted_hidden_positions,
                                                     &model_workspaces_[0].hidden_states);

      std::vector<int> input_tokens;
      Array<RequestModelState> mstates;
      input_tokens.reserve(num_rsentries);
      mstates.reserve(num_rsentries);
      for (const RequestStateEntry& rsentry : rsentries) {
        mstates.push_back(rsentry->mstates[draft_model_id_]);
      }
      for (int i = 0; i < num_rsentries; ++i) {
        TVM_FFI_ICHECK(!mstates[i]->committed_tokens.empty());
        input_tokens.push_back(mstates[i]->committed_tokens.back().GetTokenId());
      }

      TVM_FFI_ICHECK_NE(estate->spec_draft_length, 0);
      Array<Tensor> multi_step_logits = models_[draft_model_id_]->GetMultiStepLogits(hidden_states);
      std::vector<int> sample_indices(num_rsentries);
      std::iota(sample_indices.begin(), sample_indices.end(), 0);

      for (int draft_id = 0; draft_id < estate->spec_draft_length; ++draft_id) {
        const auto& [renormalized_probs, sample_results] = ApplyLogitProcessorAndSample(
            logit_processor_, sampler_, multi_step_logits[draft_id], generation_cfg, request_ids,
            mstates, rngs, sample_indices, generation_cfg, request_ids, sample_indices);
        UpdateRequestStatesWithDraftProposals(mstates, sample_results, draft_model_id_,
                                              renormalized_probs, hidden_states, estate);
      }
    }

    for (const RequestStateEntry& rsentry : rsentries) {
      rsentry->mstates[verify_model_id_]->num_tokens_for_next_decode = 0;
      rsentry->mstates[draft_model_id_]->num_tokens_for_next_decode = 0;
    }
    auto tend = std::chrono::high_resolution_clock::now();
    double elapsed_time = static_cast<double>((tend - tstart).count()) / 1e9;
    estate->metrics.engine_decode_time_sum += elapsed_time;
    estate->metrics.UpdateVerifyTimeByBatchSize(cum_verify_lengths.back(), elapsed_time);

    return estate->running_queue;
  }

 private:
  struct DraftRequestStateEntries {
    Array<RequestStateEntry> draft_rsentries;
    std::vector<int> draft_lengths;
    int total_draft_length;
  };

  DraftRequestStateEntries GetDraftsToVerify(EngineState estate) {
    std::vector<int> draft_lengths;
    int total_draft_length = 0;
    int total_required_pages = 0;

    std::vector<RequestStateEntry> running_rsentries = estate->GetRunningRequestStateEntries();
    std::vector<int> num_page_requirement;
    num_page_requirement.reserve(running_rsentries.size());
    for (const RequestStateEntry& rsentry : running_rsentries) {
      int draft_length = rsentry->mstates[draft_model_id_]->draft_output_tokens.size();
      int num_require_pages = (draft_length + engine_config_->kv_cache_page_size - 1) /
                              engine_config_->kv_cache_page_size;
      draft_lengths.push_back(draft_length);
      num_page_requirement.push_back(num_require_pages);
      total_draft_length += draft_length;
      total_required_pages += num_require_pages;
    }
    while (!CanVerify(total_required_pages)) {
      if (estate->prefix_cache->TryFreeMemory()) continue;
      RequestStateEntry preempted = PreemptLastRunningRequestStateEntry(
          estate, models_, draft_token_workspace_manager_, trace_recorder_);
      if (preempted.same_as(running_rsentries.back())) {
        total_draft_length -= draft_lengths.back();
        total_required_pages -= num_page_requirement.back();
        draft_lengths.pop_back();
        num_page_requirement.pop_back();
        running_rsentries.pop_back();
      }
    }

    return {running_rsentries, draft_lengths, total_draft_length};
  }

  bool CanVerify(int num_required_pages) {
    int num_available_pages = models_[0]->GetNumAvailablePages();
    return num_required_pages <= num_available_pages;
  }

  void UpdateRequestStatesWithDraftProposals(const Array<RequestModelState>& mstates,
                                             const std::vector<SampleResult>& sample_results,
                                             int model_id, const Tensor& renormalized_probs,
                                             const ObjectRef& hidden_states_for_sample,
                                             EngineState estate) {
    (void)hidden_states_for_sample;
    (void)estate;
    (void)model_id;
    draft_token_workspace_manager_->AllocSlots(mstates.size(), &draft_token_slots_);
    models_[0]->ScatterDraftProbs(renormalized_probs, draft_token_slots_,
                                  &model_workspaces_[0].draft_probs_storage);
    for (int i = 0; i < static_cast<int>(mstates.size()); ++i) {
      int64_t parent_idx = static_cast<int64_t>(mstates[i]->draft_output_tokens.size()) - 1;
      mstates[i]->AddDraftToken(sample_results[i], draft_token_slots_[i], parent_idx);
    }
  }

  Array<Model> models_;
  LogitProcessor logit_processor_;
  Sampler sampler_;
  std::vector<ModelWorkspace> model_workspaces_;
  DraftTokenWorkspaceManager draft_token_workspace_manager_;
  EngineConfig engine_config_;
  Optional<EventTraceRecorder> trace_recorder_;
  RandomGenerator& rng_;
  const int verify_model_id_ = 0;
  const int draft_model_id_ = 1;
  const float eps_ = 1e-5;
  std::vector<int> draft_token_slots_;
};

EngineAction EngineAction::MedusaBatchVerify(
    Array<Model> models, LogitProcessor logit_processor, Sampler sampler,
    std::vector<ModelWorkspace> model_workspaces,
    DraftTokenWorkspaceManager draft_token_workspace_manager, EngineConfig engine_config,
    Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(tvm::ffi::make_object<MedusaBatchVerifyActionObj>(
      std::move(models), std::move(logit_processor), std::move(sampler),
      std::move(model_workspaces), std::move(draft_token_workspace_manager),
      std::move(engine_config), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
