#!/bin/bash
# Π.5 -- one-command Azriel server start.
# Stops Ollama (frees GPU), starts FastAPI on 0.0.0.0:8080, logs to
# ~/.azriel/logs/server-<timestamp>.log, prints the URL and tails logs.
#
# Run on a development machine:
# bash ~/azriel-arch/scripts/serve.sh
#
# To background: prefix with `nohup` or run inside tmux. Stop with Ctrl-C.
set -u
PATH=/opt/homebrew/bin:/usr/local/bin:$PATH

cd "$HOME/azriel-arch" || { echo "azriel-arch not found"; exit 1; }

# Free the GPU for MLX.
brew services stop ollama 2>/dev/null

LOG_DIR="$HOME/.azriel/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/server-$(date +%Y%m%d-%H%M%S).log"
ln -sfn "$LOG" "$LOG_DIR/server-current.log"

HOST="${AZRIEL_HOST:-0.0.0.0}"
PORT="${AZRIEL_PORT:-8080}"

echo "azriel server -> http://$HOST:$PORT/ (log: $LOG)"
echo "ctrl-c to stop"
echo

AZRIEL_HOST="$HOST" AZRIEL_PORT="$PORT" \
  PYTHONPATH=. \
  "$HOME/.azriel/.venv/bin/python" -m azriel.server 2>&1 | tee "$LOG"
