.PHONY: setup train stream stream-drift run-serving test lint feast-dry-run docker-up docker-down git-sync

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

train:
	$(PYTHON) -m src.training.train
	$(PYTHON) -m src.training.export_onnx

feast-dry-run:
	cd feature_repo && (feast apply --dry-run || feast plan)

stream:
	$(PYTHON) -m src.producer.transaction_generator --count 500 --http-url http://localhost:8000/v1/predict

stream-drift:
	$(PYTHON) -m src.producer.transaction_generator --count 500 --inject-drift --http-url http://localhost:8000/v1/predict

run-serving:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

test:
	ruff check src feature_repo tests
	pytest -v

lint:
	ruff check src feature_repo tests
	ruff format --check src feature_repo tests

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

git-sync:
	bash scripts/init_github.sh
