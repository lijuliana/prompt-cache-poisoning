# system-prompt-poisoning

Does the engine step that first computes a shared, cached system prompt leave a mark that every other request reusing the cache can read?

Setup: `code/setup.sh` (vLLM 0.33.0, Qwen3-8B).
