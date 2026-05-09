#!/usr/bin/env bash
# launchd agent installer.
#
# The plists in this directory ship with __USER_HOME__ and
# __BASIC_AUTH_PASS__ placeholders. This script substitutes them with
# real values, copies into ~/Library/LaunchAgents/, and bootstraps
# each agent. Re-runnable; existing agents are booted out before
# re-bootstrapping.
#
# Usage:
# AZRIEL_BASIC_AUTH_USER=Azriel AZRIEL_BASIC_AUTH_PASS=changeme \
# scripts/install_launchagents.sh
#
# Or interactive (will prompt for the password):
# scripts/install_launchagents.sh
#
# Required: macOS, $HOME pointing at the user's home dir, and the
# repo cloned to $HOME/azriel-arch (mirror) with a venv at
# $HOME/.azriel/.venv. Edit the source plists if your layout
# differs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/scripts"
DST_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$DST_DIR"

# Ask for the Basic Auth password if not in env. Username defaults to
# "Azriel" -- override via AZRIEL_BASIC_AUTH_USER if you want
# something else.
USER_NAME="${AZRIEL_BASIC_AUTH_USER:-Azriel}"
if [ -z "${AZRIEL_BASIC_AUTH_PASS:-}" ]; then
    read -r -s -p "Basic Auth password (will not echo): " AZRIEL_BASIC_AUTH_PASS
    echo
fi
if [ -z "$AZRIEL_BASIC_AUTH_PASS" ]; then
    echo "Refusing to install with empty password. Set AZRIEL_BASIC_AUTH_PASS." >&2
    exit 1
fi

UID_NUM="$(id -u)"
HOME_ESC="${HOME//\//\\/}"
PASS_ESC="${AZRIEL_BASIC_AUTH_PASS//\//\\/}"

for plist in com.azriel.server.plist com.azriel.autoresearch.plist com.azriel.drift.plist; do
    src="$SRC_DIR/$plist"
    dst="$DST_DIR/$plist"
    if [ ! -f "$src" ]; then
        echo "skip: $src not found" >&2
        continue
    fi
    # Substitute placeholders and write to ~/Library/LaunchAgents/.
    sed \
        -e "s/__USER_HOME__/$HOME_ESC/g" \
        -e "s/__BASIC_AUTH_PASS__/$PASS_ESC/g" \
        "$src" > "$dst"
    # Also bake the auth user into the server plist so the live server
    # has it; the source plist doesn't carry the user/pass keys (only
    # the drift plist does, baked from ι.22). Add them to server here:
    if [ "$plist" = "com.azriel.server.plist" ]; then
        /usr/libexec/PlistBuddy -c \
            "Add :EnvironmentVariables:AZRIEL_BASIC_AUTH_USER string $USER_NAME" \
            "$dst" 2>/dev/null || \
        /usr/libexec/PlistBuddy -c \
            "Set :EnvironmentVariables:AZRIEL_BASIC_AUTH_USER $USER_NAME" \
            "$dst"
        /usr/libexec/PlistBuddy -c \
            "Add :EnvironmentVariables:AZRIEL_BASIC_AUTH_PASS string $AZRIEL_BASIC_AUTH_PASS" \
            "$dst" 2>/dev/null || \
        /usr/libexec/PlistBuddy -c \
            "Set :EnvironmentVariables:AZRIEL_BASIC_AUTH_PASS $AZRIEL_BASIC_AUTH_PASS" \
            "$dst"
    fi
    # Reload the agent.
    launchctl bootout "gui/$UID_NUM/${plist%.plist}" 2>/dev/null || true
    launchctl bootstrap "gui/$UID_NUM" "$dst"
    echo "installed: $plist"
done

echo
echo "Verify: launchctl list | grep com.azriel"
echo " curl -fsS -u \"$USER_NAME:<password>\" http://127.0.0.1:8080/health"
