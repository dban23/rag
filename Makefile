.PHONY: help setup pull-llm index up down logs cleanup

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Build images and start the stack (do this first)
	docker compose up -d --build

pull-llm: ## Pull the models into the ollama volume (one-time ~2.2GB)
	docker compose exec ollama ollama pull llama3.2:3b
	docker compose exec ollama ollama pull nomic-embed-text

index: ## (Re)build the vector index from data/ into the app_index volume
	docker compose run --rm app python src/indexing.py

up: ## Start the stack
	docker compose up -d

down: ## Stop the stack (keeps volumes and models)
	docker compose down

logs: ## Follow app logs
	docker compose logs -f app

cleanup: ## Wipe containers AND volumes (models, index, uploads) - DANGER
	docker compose down -v
