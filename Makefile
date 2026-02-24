SHELL := /bin/sh

CONFIG ?= config/runtime.json
SERIES ?= KXBTC15M
STRATEGY ?= pipeline_data/strategies/rnn_edge_latest/strategy.json
PYTHON ?= ./venv/bin/python
TRAIN_ARGS ?=
EVAL_ARGS ?=
SYNC_ARGS ?=
EVAL_AFTER_TRAIN ?= 1
POST_TRAIN_EVAL_ARGS ?= --mode fixed

.PHONY: help check-python download labels sync-labels train smoke-train evaluate live-dry live

help:
	@echo "Targets:"
	@echo "  make download   CONFIG=$(CONFIG) SERIES=$(SERIES)"
	@echo "  make labels     CONFIG=$(CONFIG)"
	@echo "  make sync-labels CONFIG=$(CONFIG) SYNC_ARGS='...'"
	@echo "  make train      CONFIG=$(CONFIG) STRATEGY=$(STRATEGY) EVAL_AFTER_TRAIN=$(EVAL_AFTER_TRAIN)"
	@echo "  make smoke-train CONFIG=$(CONFIG)"
	@echo "  make evaluate   CONFIG=$(CONFIG) STRATEGY=$(STRATEGY) EVAL_ARGS='...'"
	@echo "  make live-dry   CONFIG=$(CONFIG) STRATEGY=$(STRATEGY) SERIES=$(SERIES)"
	@echo "  make live       CONFIG=$(CONFIG) STRATEGY=$(STRATEGY) SERIES=$(SERIES)"

check-python:
	@test -x "$(PYTHON)" || { echo "Missing Python at $(PYTHON). Set PYTHON=python3 or create ./venv."; exit 1; }

download: check-python
	$(PYTHON) scripts/download_highres.py \
		--config "$(CONFIG)" \
		--series "$(SERIES)"

labels: check-python
	$(PYTHON) scripts/build_labels.py \
		--config "$(CONFIG)"

sync-labels: check-python
	$(PYTHON) scripts/sync_labels_from_api.py \
		--config "$(CONFIG)" \
		$(SYNC_ARGS)

train: check-python
	$(PYTHON) scripts/train_edge_rnn.py --config "$(CONFIG)" $(TRAIN_ARGS)
ifneq ($(EVAL_AFTER_TRAIN),0)
	$(PYTHON) scripts/evaluate_strategy.py \
		--config "$(CONFIG)" \
		--strategy "$(STRATEGY)" \
		$(POST_TRAIN_EVAL_ARGS)
endif

evaluate: check-python
	$(PYTHON) scripts/evaluate_strategy.py \
		--config "$(CONFIG)" \
		--strategy "$(STRATEGY)" \
		$(EVAL_ARGS)

smoke-train: check-python
	@SNAP=$$(find pipeline_data/highres -maxdepth 1 -type f -name '*.snap_*ms.jsonl*' | head -n 1); \
	if [ -z "$$SNAP" ]; then \
		echo "No snapshot files found under pipeline_data/highres"; \
		exit 1; \
	fi; \
	if ! TICKER=$$($(PYTHON) scripts/infer_snapshot_ticker.py "$$SNAP"); then \
		echo "Could not infer market_ticker from $$SNAP"; \
		exit 1; \
	fi; \
	LABELS=$$(mktemp /tmp/kalshi_smoke_labels.XXXXXX.json); \
	printf '{"%s":"yes"}\n' "$$TICKER" > "$$LABELS"; \
	echo "Smoke train on $$SNAP (ticker=$$TICKER)"; \
	STATUS=0; \
	$(PYTHON) scripts/train_edge_rnn.py \
		--config "$(CONFIG)" \
		--snapshots "$$SNAP" \
		--labels "$$LABELS" \
		--outdir /tmp/kalshi_smoke_strategy \
		--epochs 1 \
		--batch-size 4 \
		--seq-len 64 \
		--min-records 10 \
		--device cpu || STATUS=$$?; \
	rm -f "$$LABELS"; \
	exit $$STATUS

live-dry: check-python
	$(PYTHON) scripts/run_live.py \
		--config "$(CONFIG)" \
		--strategy "$(STRATEGY)" \
		--series "$(SERIES)"

live: check-python
	$(PYTHON) scripts/run_live.py \
		--config "$(CONFIG)" \
		--strategy "$(STRATEGY)" \
		--series "$(SERIES)" \
		--live
