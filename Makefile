.PHONY: black style validate test install serve

install:
	uv pip install -e .

black:
	uv run black src --line-length 120
	uv run black scripts --line-length 120

validate:
	uv run black src --line-length 120
	uv run black scripts --line-length 120
	uv run flake8 src --count --statistics
	uv run flake8 scripts --count --statistics
