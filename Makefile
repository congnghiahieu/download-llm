PYTHON ?= uv run python
SRC_IMAGE_NAME ?= agent-mgmt-src
SRC_IMAGE_REPOSITORY ?= hieucien/$(SRC_IMAGE_NAME)
SRC_DOCKERFILE ?= docker/Dockerfile.src
SRC_IMAGE ?=

vars:
	@echo "PYTHON=$(PYTHON)"
	@echo "SRC_IMAGE_REPOSITORY=$(SRC_IMAGE_REPOSITORY)"
	@echo "SRC_DOCKERFILE=$(SRC_DOCKERFILE)"
	@echo "SRC_IMAGE=$(SRC_IMAGE)"

push-src:
	@$(PYTHON) scripts/src_image.py --project-name $(SRC_IMAGE_NAME) push --image-repository $(SRC_IMAGE_REPOSITORY) --dockerfile $(SRC_DOCKERFILE)

pull-src:
	@test -n "$(SRC_IMAGE)" || (echo "SRC_IMAGE is required, for example: make pull-src SRC_IMAGE=hieucien/$(SRC_IMAGE_NAME):2026-06-03_17-22-20" && exit 1)
	@$(PYTHON) scripts/src_image.py --project-name $(SRC_IMAGE_NAME) pull --image $(SRC_IMAGE)

sync:
	@uv sync
	@uv run pyrefly init

lint:
	@uv run ruff check --fix

lint-unsafe:
	@uv run ruff check --fix --unsafe-fixes

format: lint
	@uv run ruff format

type-check:
	@uv run pyrefly check --output pyrefly-type-check-errors.json --output-format json --summarize-errors
