.PHONY: install dev test unit integration lint assets assets-video benchmark build clean

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[dev]"

unit:
	pytest -m "not integration"

integration:
	pytest -m integration

test:
	pytest

lint:
	ruff check .

assets:
	python generate_assets.py

assets-video:
	python generate_assets.py --include-video

benchmark:
	python -m benchmarks.run_benchmarks --input assets/original.png --key docs-only-demo-key-2026 --message DSP-WATERMARK-2026

build:
	python -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
