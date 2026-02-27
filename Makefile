.PHONY: install test lint clean run serve docker docker-compose train demo

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ --fix
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m src.cli serve

serve:
	python -m src.cli serve

train:
	python -m src.cli train data/sample/train.csv

docker:
	docker build -t $(shell basename $(CURDIR)) .

docker-compose:
	docker compose up --build

demo:
	python scripts/demo.py
