#!/bin/bash
# switch_env.sh: Utility to switch between vLLM worktrees sequentially.
# Usage: ./switch_env.sh [dev1|dev2|main] [prod|dev|staging]

TARGET_DIR=$1
PROFILE=${2:-dev}
BASE_DIR=${VLLM_BASE_DIR:-$(dirname "$(realpath "$0")")/..}

if [ -z "$TARGET_DIR" ]; then
    echo "Usage: ./switch_env.sh [dev1|dev2|main] [profile]"
    exit 1
fi

# Resolve directory
if [ "$TARGET_DIR" == "main" ]; then
    DIR="$BASE_DIR/vllm_services"
else
    DIR="$BASE_DIR/vllm_services_$TARGET_DIR"
fi

if [ ! -d "$DIR" ]; then
    echo "Error: Directory $DIR does not exist."
    exit 1
fi

echo "Switching to environment in $DIR (Profile: $PROFILE)..."

# Since we use 'name: vllm_services' in docker-compose.yml,
# running 'down' in ANY worktree will stop ALL vllm_services containers.
echo "Stopping any currently running instances of vllm_services..."
docker compose -p vllm_services down

cd "$DIR"
echo "Starting $PROFILE in $DIR..."
docker compose --profile "$PROFILE" up -d

echo "Done. Environments switched."
