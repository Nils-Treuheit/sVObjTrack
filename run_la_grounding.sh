#!/bin/bash
# Query objects in the scene via LocateAnything visual grounding.
#
# Usage:
#   ./run_la_grounding.sh "human plant chair"   # find multiple objects
#   ./run_la_grounding.sh "plant"                # find a single object
#   ./run_la_grounding.sh "person wearing a hat" # relational query (may work)
#
# The query is sent to /la/grounding_query and the LA node responds
# on /la/grounding_text with the model's answer and /la/debug_image
# with the annotated frame.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

source "$DIR/.venv/bin/activate"
source /opt/ros/humble/setup.bash

if [ $# -eq 0 ]; then
  echo "Usage: $0 \"objects to find\""
  echo ""
  echo "Examples:"
  echo "  $0 \"human plant chair\""
  echo "  $0 \"plant\""
  echo "  $0 \"person wearing a red hat\""
  echo "  $0 \"car\""
  exit 1
fi

QUERY="$*"
echo "[run_la_grounding] Query: $QUERY"

# Detect if it's a simple list of objects (space-separated names) or a phrase
# If 2+ words and no stopwords/prepositions, treat as list of separate objects
GROUNDING_QUERY="$QUERY"

# Echo the grounding_text topic — use full_length to avoid truncation
stdbuf -oL ros2 topic echo /la/grounding_text --full-length &
ECHO_PID=$!
sleep 0.5

ros2 topic pub -1 /la/grounding_query std_msgs/msg/String "data: '$GROUNDING_QUERY'"

sleep 3
kill $ECHO_PID 2>/dev/null || true
