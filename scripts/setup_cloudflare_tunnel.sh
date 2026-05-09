#!/bin/bash
# Set up a Cloudflare Tunnel mapping your-host.example -> a development machine localhost:8080.
#
# This script is GUIDED -- it pauses for the one-time browser auth step,
# then drives the rest. Run from the development machine:
# bash ~/azriel-arch/scripts/setup_cloudflare_tunnel.sh
#
# Prerequisites:
# - cloudflared installed (brew install cloudflared)
# - your-domain.example on Cloudflare DNS (verified)
# - Azriel server running on localhost:8080 (start with serve.sh)
#
# This is private -- before going public, add a Cloudflare Access policy
# in the dashboard restricting the hostname to your email allowlist.
set -u
PATH=/opt/homebrew/bin:/usr/local/bin:$PATH

TUNNEL_NAME="azriel"
HOSTNAME="your-host.example"
LOCAL_URL="http://localhost:8080"
CONFIG_DIR="$HOME/.cloudflared"

if ! command -v cloudflared >/dev/null; then
  echo "ERROR: cloudflared not found. Install: brew install cloudflared"
  exit 1
fi

# Step 1: cert.pem (login). Required once per Cloudflare account.
if [ ! -f "$CONFIG_DIR/cert.pem" ]; then
  echo
  echo "==> [1/5] cloudflared login"
  echo " A browser will open. Pick the 'your-domain.example' zone from the list and authorize."
  echo " Press Enter when ready..."
  read -r
  cloudflared tunnel login || { echo "login failed"; exit 1; }
fi

# Step 2: create tunnel (idempotent -- skip if already exists by name).
TUNNEL_UUID=$(cloudflared tunnel list 2>/dev/null | awk -v n="$TUNNEL_NAME" '$2==n {print $1}')
if [ -z "$TUNNEL_UUID" ]; then
  echo
  echo "==> [2/5] creating tunnel '$TUNNEL_NAME'"
  cloudflared tunnel create "$TUNNEL_NAME"
  TUNNEL_UUID=$(cloudflared tunnel list 2>/dev/null | awk -v n="$TUNNEL_NAME" '$2==n {print $1}')
else
  echo "==> [2/5] tunnel '$TUNNEL_NAME' already exists ($TUNNEL_UUID)"
fi

if [ -z "$TUNNEL_UUID" ]; then
  echo "ERROR: failed to create or find tunnel"
  exit 1
fi

# Step 3: write config.
echo
echo "==> [3/5] writing $CONFIG_DIR/config.yml"
cat > "$CONFIG_DIR/config.yml" <<EOF
tunnel: $TUNNEL_UUID
credentials-file: $CONFIG_DIR/$TUNNEL_UUID.json

ingress:
  - hostname: $HOSTNAME
    service: $LOCAL_URL
    originRequest:
      httpHostHeader: $HOSTNAME
      noTLSVerify: true
  - service: http_status:404
EOF
echo " wrote config.yml"

# Step 4: route DNS (creates CNAME on Cloudflare side).
echo
echo "==> [4/5] routing DNS $HOSTNAME -> tunnel"
cloudflared tunnel route dns "$TUNNEL_NAME" "$HOSTNAME" 2>&1 | tail -5

# Step 5: run the tunnel.
echo
echo "==> [5/5] starting tunnel"
echo " Verify after start: curl -I https://$HOSTNAME/health"
echo " For persistent run, install as a service:"
echo " sudo cloudflared --config $CONFIG_DIR/config.yml service install"
echo
echo "Starting in foreground now (Ctrl-C to stop)..."
echo
cloudflared tunnel --config "$CONFIG_DIR/config.yml" run "$TUNNEL_NAME"
