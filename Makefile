# MasterFabric LLM Serving Bench. Run `make help` for the list of targets.
SHELL := /bin/bash
PY    ?= .venv/bin/python

.PHONY: help venv venv-mlx models models-extra bench concurrency prefill kvcache promptcache perplexity mlx mlx-concurrency mlx-single ctx spec spec-mismatch mlx-ppl charts clean-servers

help: ## Show targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

venv: ## Create .venv with uv and install the pinned client and chart dependencies
	uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt

venv-mlx: ## Add MLX (Apple Silicon only) to .venv
	uv pip install -p .venv -r requirements-mlx.txt

models: ## Download the GGUF files to ~/.cache/mf-models
	bench/get_models.sh

models-extra: ## Download the 7B, 0.5B draft and MLX models (about 12 GB)
	bench/get_models_extra.sh

bench: ## Run every experiment (about 25 minutes on an M4 Pro)
	bench/run_matrix.sh all

concurrency prefill kvcache promptcache perplexity: ## Run one experiment
	bench/run_matrix.sh $@

mlx: ## MLX vs llama.cpp, long-context curve and speculative decoding (about 20 minutes)
	bench/run_mlx.sh all

mlx-concurrency mlx-single ctx spec: ## Run one MLX-related experiment
	bench/run_mlx.sh $(patsubst mlx-%,%,$@)

mlx-ppl: ## Perplexity of the MLX 4-bit and 8-bit weights
	$(PY) bench/mlx_perplexity.py

spec-mismatch: ## Why draft output differs from the target alone (7B and 1.5B, about 10 minutes)
	cd bench && ../$(PY) spec_mismatch.py --target 7b --out ../results/air/spec_mismatch_7b_air.json
	cd bench && ../$(PY) spec_mismatch.py --target 1.5b --out ../results/air/spec_mismatch_1.5b_air.json

charts: ## Render charts, the setup diagram and the hero from results/*.json
	cd bench && ../$(PY) make_charts.py && ../$(PY) make_diagrams.py && ../$(PY) make_hero.py

clean-servers: ## Stop any server this repo started
	-[ -f results/raw/metal.pid ] && kill $$(cat results/raw/metal.pid) 2>/dev/null; rm -f results/raw/metal.pid
	-docker rm -f mf-bench-cpu 2>/dev/null
	-pkill -f 'mlx_lm.server .*--port 58082' 2>/dev/null; pkill -f 'llama-server .*--port 5808[34]' 2>/dev/null; true
