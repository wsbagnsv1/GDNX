#!/bin/bash
# Wait for the GLM endpoint to come back after the quant swap, discover the real
# model id from /v1/models, patch models.json + the loop, verify, then launch.
# Safe: only launches if a live pi test call succeeds; otherwise logs and stops.
set -u
cd /home/dev/gdn3_two_timescale_release
LOG=research/resume.log
MODELS_JSON=/home/dev/.pi/agent/models.json
NODE=/home/dev/.local/share/pi-node/node-v22.23.1-linux-x64/bin/node
CLI=/home/dev/.local/share/pi-node/current/lib/node_modules/@earendil-works/pi-coding-agent/dist/cli.js
BASE=$(python3 -c "import json;print(json.load(open('$MODELS_JSON'))['providers']['glm']['baseUrl'])")
KEY=$(python3 -c "import json;print(json.load(open('$MODELS_JSON'))['providers']['glm']['apiKey'])")
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "auto-resume watching $BASE (polling every 30s, up to ~2 h while it loads)"
ID=""
for i in $(seq 1 240); do
  code=$(curl -s -m 10 -o /tmp/glm_models.json -w "%{http_code}" "$BASE/models" -H "Authorization: Bearer $KEY" 2>/dev/null)
  if [ "$code" = "200" ]; then
    ID=$(python3 -c "import json;d=json.load(open('/tmp/glm_models.json'))['data'];\
ids=[m['id'] for m in d];\
g=[x for x in ids if 'glm' in x.lower()];\
print((g or ids)[0])" 2>/dev/null)
    say "endpoint UP. /v1/models reports id: $ID"
    break
  fi
  sleep 30
done
[ -z "$ID" ] && { say "endpoint never came back — NOT launching. Rerun this script or launch manually when ready."; exit 1; }

# patch models.json glm id + loop --model to the server-reported id
python3 - "$MODELS_JSON" "$ID" <<'PY'
import json,sys
p,i=sys.argv[1],sys.argv[2]
d=json.load(open(p)); d['providers']['glm']['models'][0]['id']=i
json.dump(d,open(p,'w'),indent=2)
PY
sed -i -E "s#(--model \")[^\"]*(\")#\1$ID\2#" research/run_auto_research.sh
say "patched models.json + run_auto_research.sh to id: $ID"

# verify with a live pi call
OUT=$(timeout 150 "$NODE" "$CLI" -p --no-session --approve --provider glm --model "$ID" --tools read \
      "Reply with exactly the token RESUME_OK and nothing else." 2>&1 | tail -3)
if echo "$OUT" | grep -q "RESUME_OK"; then
  say "pi test OK — launching auto-research loop."
  rm -f research/STOP
  nohup bash research/run_auto_research.sh > research/loop.out 2>&1 &
  say "🚀 loop launched, pid $!  (model=$ID, proxy on cuda:1)"
else
  say "pi test FAILED (model up but pi couldn't use it). NOT launching. Last output:"; echo "$OUT" | tee -a "$LOG"
  exit 1
fi
