#!/bin/bash
# Query the orchestrator from the command line.
#
# Usage:
#   ./run_orchestrator_query.sh "how many people are in the scene?"
#   ./run_orchestrator_query.sh "find the red cup"
#   ./run_orchestrator_query.sh "read the text on the cube"
#   ./run_orchestrator_query.sh "select the cube with a 2"
#
# The query is sent to /orchestrator/query and the response
# is printed from /orchestrator/response.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

source "$DIR/.venv/bin/activate"
source /opt/ros/humble/setup.bash
source "$DIR/install/setup.bash" 2>/dev/null || true

if [ $# -eq 0 ]; then
  echo "Usage: $0 \"your text query\""
  echo ""
  echo "Examples:"
  echo "  $0 \"how many people are in the scene?\""
  echo "  $0 \"find the red cup\""
  echo "  $0 \"read the text on the cube\""
  echo "  $0 \"select the cube with a 2\""
  echo "  $0 \"plant closest to the person\""
  exit 1
fi

python3 "$DIR/run_orchestrator_query.py" "$@"
