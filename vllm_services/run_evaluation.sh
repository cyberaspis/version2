#!/bin/bash
# Wrapper to run classifier evaluation inside the dev container

# Ensure the container is running and rebuilt
echo "Restarting classifier dev container to pick up updates (including results volume)..."
docker compose --profile dev up -d classifier-dev

echo "Running evaluation on full dataset..."
docker exec -e OUTPUT_FILE="/app/vllm_services/results/evaluation_results.json" classifier_dev python3 -m vllm_services.merge_and_evaluate_classifier
echo "Results saved to: ./results/evaluation_results.json"
