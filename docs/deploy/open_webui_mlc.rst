Open WebUI + MLC-LLM (Docker) Backend Blueprint
===============================================

This guide outlines a practical way to run `Open WebUI <https://github.com/open-webui/open-webui>`_
on top of MLC-LLM for an OpenAI-compatible chat experience.

Why this is feasible
--------------------

MLC-LLM already exposes OpenAI-style endpoints in its serving stack, including:

- ``/v1/chat/completions``
- ``/v1/completions``
- ``/v1/models``
- ``/v1/embeddings``

That means Open WebUI can talk to MLC-LLM the same way it talks to OpenAI-compatible
providers, with only endpoint and key configuration.

Complexity estimate
-------------------

For a chat-only MVP:

- **Backend integration:** low-to-medium effort (roughly 1-3 engineering days).
- **Production hardening:** medium effort (1-2 weeks depending on GPU matrix,
  model caching strategy, and CI for images).
- **Open WebUI UX polish for model onboarding:** medium effort (few days), if you
  add custom workflow for model downloads from the MLC Hugging Face registry.

Reference architecture
----------------------

Use two containers in a single ``docker-compose.yml``:

1. ``mlc-server`` (MLC-LLM OpenAI-compatible REST server)
2. ``open-webui`` (frontend + session management)

Open WebUI points at the MLC endpoint (for example ``http://mlc-server:8000/v1``)
using an API key shared via environment variable.

Minimal compose example
-----------------------

.. code-block:: yaml

   services:
     mlc-server:
       image: ghcr.io/mlc-ai/mlc-llm:nightly-cu121
       container_name: mlc-server
       command: >
         python -m mlc_llm serve /models/Qwen2.5-7B-Instruct-MLC
         --mode server
         --device cuda
         --host 0.0.0.0
         --port 8000
         --api-key ${MLC_API_KEY}
       environment:
         - MLC_API_KEY=${MLC_API_KEY}
       volumes:
         - ./models:/models
       ports:
         - "8000:8000"
       deploy:
         resources:
           reservations:
             devices:
               - capabilities: [gpu]

     open-webui:
       image: ghcr.io/open-webui/open-webui:main
       container_name: open-webui
       depends_on:
         - mlc-server
       environment:
         - OPENAI_API_BASE_URL=http://mlc-server:8000/v1
         - OPENAI_API_KEY=${MLC_API_KEY}
       ports:
         - "3000:8080"

Notes:

- Use a model path that exists in your mounted ``./models`` directory.
- For local testing, this setup can be enough without changing Open WebUI code.

Automating model download from the MLC registry
------------------------------------------------

To make this end-user friendly, add one thin layer around model acquisition:

- Add a startup script (or sidecar) that:

  1. Reads ``MODEL_ID`` from env.
  2. Pulls matching artifact(s) from ``https://huggingface.co/mlc-ai/models``.
  3. Materializes them into the mounted ``/models`` volume.
  4. Boots ``mlc_llm serve`` against that resolved local path.

A pragmatic first version is a shell/python entrypoint that resolves aliases
(eg. ``qwen2.5-7b``) to full model repo/path and caches to persistent volume.

Suggested rollout
-----------------

1. **Phase 1 (chat MVP)**

   - Compose-based deployment with static model path.
   - Open WebUI manually configured to MLC endpoint.

2. **Phase 2 (usability)**

   - Add model bootstrap automation for Hugging Face registry.
   - Offer one-command startup with model selection env vars.

3. **Phase 3 (polish)**

   - Add health checks and startup ordering.
   - Publish CPU/CUDA/Metal image variants and compatibility matrix.
   - Add metrics/logging defaults for debugging user deployments.

MLC-LLM capabilities relevant to this setup
-------------------------------------------

MLC-LLM's serve CLI already includes options needed for container deployments,
such as binding host/port and API key auth.

You can also evolve to multiple served models (``--additional-models``) once
chat MVP is stable.
