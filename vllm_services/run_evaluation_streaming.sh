#!/bin/bash
# Wrapper to run streaming classifier evaluation (TTD) inside the dev container

# Ensure the container is running and rebuilt
echo "Restarting classifier dev container to pick up updates..."
docker compose --profile dev up -d classifier-dev

echo "Running streaming evaluation on full dataset..."
docker exec -e OUTPUT_FILE="/app/vllm_services/results/streaming_evaluation_results.json" classifier_dev python3 -m vllm_services.merge_and_evaluate_streaming_classifier
echo "Detailed results saved to: ./results/streaming_evaluation_results.json"
