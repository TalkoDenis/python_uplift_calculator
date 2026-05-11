.PHONY: test lint format check

test:
	uv run pytest -v --cov=src

lint:
	uv run ruff check .

format:
	uv run ruff format .

check: format lint test
