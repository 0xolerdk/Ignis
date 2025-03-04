# ==== Python & Environment Setup ====
VENV ?= .venv
SYSTEM_PYTHON ?= python3.11
PYTHON ?= $(VENV)/bin/python
PIP ?= $(PYTHON) -m pip

FRONTEND_DIR := frontend
BACKEND_DIR := backend
BACKEND_PORT ?= 8000
FRONTEND_PORT ?= 5173
DATASET_ARGS ?= --fixtures --patch-size=64 --stride=32 --limit=3
NOWCAST_DATA_DIR ?= data/nowcast
NOWCAST_OUTPUT ?= runs/nowcast

.PHONY: setup activate lint test dataset train-seg train-nowcast serve frontend compose-up compose-down clean

# Create virtual environment if it doesn't exist
$(VENV)/bin/python:
	$(SYSTEM_PYTHON) -m venv $(VENV)

# ==== Setup Project ====
setup: $(VENV)/bin/python
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	cd $(FRONTEND_DIR) && npm install

# ==== Activate Virtual Environment ====
activate:
	@echo "Run this to activate your venv:"
	@echo "source $(VENV)/bin/activate"

# ==== Lint ====
lint:
	$(PYTHON) -m ruff check $(BACKEND_DIR)
	$(PYTHON) -m black --check $(BACKEND_DIR)
	cd $(FRONTEND_DIR) && npm run lint

# ==== Tests ====
test:
	$(PYTHON) -m pytest $(BACKEND_DIR)/tests --cov=$(BACKEND_DIR)

# ==== Dataset Build ====
dataset:
	$(PYTHON) $(BACKEND_DIR)/scripts/build_dataset.py $(DATASET_ARGS)

# ==== Training ====
train-seg:
	$(PYTHON) $(BACKEND_DIR)/scripts/train_segmentation.py

train-nowcast:
 	$(PYTHON) $(BACKEND_DIR)/scripts/train_nowcast.py --data-dir $(NOWCAST_DATA_DIR) --output-dir $(NOWCAST_OUTPUT)

# ==== Run Backend ====
serve:
	$(PYTHON) -m uvicorn backend.app.main:app --host 0.0.0.0 --port $(BACKEND_PORT) --reload

# ==== Run Frontend ====
frontend:
	cd $(FRONTEND_DIR) && npm run dev -- --host --port $(FRONTEND_PORT)

# ==== Docker Compose ====
compose-up:
	docker compose up --build

compose-down:
	docker compose down -v

# ==== Cleanup ====
clean:
	rm -rf $(VENV)
