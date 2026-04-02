#!/bin/bash
# =============================================================================
# neo4j_init.sh — Neo4j setup + IFC graph load for each WSL session
#
# /tmp is cleared on every WSL restart, so Neo4j must be reinstalled each time.
# This script handles the full chain:
#   1. Install Neo4j Community 5.26.0 (if missing)
#   2. Start Neo4j and wait for bolt port readiness
#   3. Load IFC data into Neo4j (skip if graph already populated)
#   4. Add ADJACENT_TO edges + CONTINUOUS properties (topology enrichment)
#   5. Print graph verification stats
#
# Usage:
#   ./script/neo4j_init.sh              # Smart init — skips steps already done
#   ./script/neo4j_init.sh --reload     # Force re-export IFC data (clears existing graph)
#   ./script/neo4j_init.sh --start-only # Only start Neo4j, skip all data loading
#   ./script/neo4j_init.sh --status     # Print current graph state and exit
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config ────────────────────────────────────────────────────────────────────
NEO4J_DIR="/tmp/neo4j-community-5.26.0"
NEO4J_URL="https://dist.neo4j.org/neo4j-community-5.26.0-unix.tar.gz"
NEO4J_TARBALL="/tmp/neo4j-community-5.26.0-unix.tar.gz"
NEO4J_CONF="${NEO4J_DIR}/conf/neo4j.conf"
NEO4J_PASSWORD="password"
HTTP_PORT=7474
BOLT_PORT=7687
BOLT_WAIT_SECS=30

# Multi-model IFC paths: AP (primary), DXA, BH
IFC_PATH_AP="$(cd "$SCRIPT_DIR" && realpath ../data/ifc/AdvancedProject/IFC/AdvancedProject.ifc)"
IFC_PATH_DXA="$(cd "$SCRIPT_DIR" && realpath ../../data_curation/ifc_models/Duplex_A_20110505.ifc)"
IFC_PATH_BH="$(cd "$SCRIPT_DIR" && realpath ../data/ifc/BasicHouse.ifc)"

INDEX_PATH_AP="$(cd "$SCRIPT_DIR" && realpath ../../data_curation/references/element_index.jsonl)"
INDEX_PATH_DXA="$(cd "$SCRIPT_DIR" && realpath ../../data_curation/datasets/synth_v0.4_dxa/element_index.jsonl)"
INDEX_PATH_BH="$(cd "$SCRIPT_DIR" && realpath ../../data_curation/datasets/synth_v0.4_bh/element_index.jsonl)"

# Legacy single-model aliases (backwards compat)
IFC_PATH="$IFC_PATH_AP"
INDEX_PATH="$INDEX_PATH_AP"

# ── Parse args ────────────────────────────────────────────────────────────────
RELOAD=false
START_ONLY=false
STATUS_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --reload)     RELOAD=true ;;
        --start-only) START_ONLY=true ;;
        --status)     STATUS_ONLY=true ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg  (use --help for usage)"
            exit 1
            ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

info()    { echo "[neo4j_init] $*"; }
ok()      { echo "[neo4j_init] ✅  $*"; }
warn()    { echo "[neo4j_init] ⚠️   $*"; }
fail()    { echo "[neo4j_init] ❌  $*" >&2; exit 1; }

ensure_neo4j_network_config() {
    local conf="$NEO4J_CONF"
    local tmp

    [ -f "$conf" ] || fail "Neo4j config not found: ${conf}"

    info "Normalizing Neo4j network config (idempotent)"

    tmp=$(mktemp)
    awk '
        BEGIN { skip=0 }
        /^# BEGIN MSCD MANAGED NETWORK CONFIG$/ { skip=1; next }
        /^# END MSCD MANAGED NETWORK CONFIG$/   { skip=0; next }
        skip == 0 { print }
    ' "$conf" > "$tmp"
    mv "$tmp" "$conf"

    tmp=$(mktemp)
    awk '
        /^(server\.default_listen_address|server\.default_advertised_address|server\.http\.enabled|server\.http\.listen_address|server\.bolt\.enabled|server\.bolt\.listen_address|server\.https\.enabled)=/ { next }
        { print }
    ' "$conf" > "$tmp"

    cat >> "$tmp" <<EOF

# BEGIN MSCD MANAGED NETWORK CONFIG
# Normalized by mscd_demo/script/neo4j_init.sh for repeatable local startup
# and WSL-friendly Browser access from Windows.
server.default_listen_address=0.0.0.0
server.default_advertised_address=localhost
server.http.enabled=true
server.http.listen_address=:${HTTP_PORT}
server.bolt.enabled=true
server.bolt.listen_address=:${BOLT_PORT}
server.https.enabled=false
# END MSCD MANAGED NETWORK CONFIG
EOF

    mv "$tmp" "$conf"
    ok "Neo4j network config normalized"
}

validate_neo4j_config() {
    info "Validating Neo4j configuration..."
    if ! "$NEO4J_DIR/bin/neo4j-admin" server validate-config >/dev/null 2>&1; then
        "$NEO4J_DIR/bin/neo4j-admin" server validate-config || true
        fail "Configuration validation failed. Check: ${NEO4J_CONF}"
    fi
    ok "Neo4j configuration valid"
}

# Run a Python snippet in the mscd_demo conda env
py() { conda run -n mscd_demo python -c "$1" 2>/dev/null; }

# Query Neo4j and return a single integer count
neo4j_count() {
    local cypher="$1"
    py "
from py2neo import Graph
try:
    g = Graph('bolt://localhost:${BOLT_PORT}', auth=('neo4j','${NEO4J_PASSWORD}'))
    print(g.run(\"\"\"${cypher}\"\"\").data()[0]['c'])
except Exception:
    print(-1)
" 2>/dev/null || echo -1
}

bolt_ready() {
    # Returns 0 (true) if bolt port is accepting connections
    python3 -c "
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(('localhost', ${BOLT_PORT}))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

# ── Status-only mode ──────────────────────────────────────────────────────────

if [ "$STATUS_ONLY" = true ]; then
    echo "=== Neo4j Graph Status ==="
    if ! bolt_ready; then
        warn "Neo4j is NOT running (bolt:${BOLT_PORT} unreachable)"
        exit 0
    fi
    ok "Neo4j is running"
    nodes=$(neo4j_count "MATCH (n:IFCElement) RETURN count(n) AS c")
    fills=$(neo4j_count "MATCH ()-[:FILLS]->() RETURN count(*) AS c")
    adj=$(neo4j_count "MATCH ()-[:ADJACENT_TO]->() RETURN count(*) AS c")
    conn=$(neo4j_count "MATCH ()-[:CONNECTS_TO]->() RETURN count(*) AS c")
    cont=$(neo4j_count "MATCH (n:IFCElement) WHERE n.is_continuous=true RETURN count(n) AS c")
    echo "  IFCElement nodes : ${nodes}"
    echo "  FILLS edges      : ${fills}"
    echo "  ADJACENT_TO edges: ${adj}  (bidirectional)"
    echo "  CONNECTS_TO edges: ${conn}  (bidirectional)"
    echo "  is_continuous    : ${cont}"
    exit 0
fi

# ── Step 1: Install Neo4j if missing ─────────────────────────────────────────

echo "============================================="
echo "  Neo4j Init — MSCD Demo"
echo "============================================="
echo ""

if [ ! -d "$NEO4J_DIR" ]; then
    warn "Neo4j not found at ${NEO4J_DIR} (expected after WSL restart)"
    info "Downloading Neo4j Community 5.26.0 (~100MB)..."
    if [ ! -f "$NEO4J_TARBALL" ]; then
        curl -L --progress-bar "$NEO4J_URL" -o "$NEO4J_TARBALL" \
            || fail "Download failed. Check your internet connection."
    fi
    info "Extracting..."
    tar -xzf "$NEO4J_TARBALL" -C /tmp/
    ok "Extracted to ${NEO4J_DIR}"

    info "Setting initial password..."
    "$NEO4J_DIR/bin/neo4j-admin" dbms set-initial-password "$NEO4J_PASSWORD" \
        2>/dev/null || true   # harmless if already set
    ok "Password set to '${NEO4J_PASSWORD}'"
else
    ok "Neo4j found at ${NEO4J_DIR}"
fi

echo ""
ensure_neo4j_network_config
validate_neo4j_config

# ── Step 2: Start Neo4j ───────────────────────────────────────────────────────

if bolt_ready; then
    ok "Neo4j already running on bolt:${BOLT_PORT}"
else
    info "Starting Neo4j..."
    "$NEO4J_DIR/bin/neo4j" start

    info "Waiting for bolt:${BOLT_PORT} (up to ${BOLT_WAIT_SECS}s)..."
    elapsed=0
    until bolt_ready; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "$elapsed" -ge "$BOLT_WAIT_SECS" ]; then
            fail "Timed out waiting for Neo4j. Check: ${NEO4J_DIR}/logs/neo4j.log"
        fi
        printf "."
    done
    echo ""
    ok "Neo4j is ready (bolt:${BOLT_PORT})"
fi

[ "$START_ONLY" = true ] && { echo ""; ok "Done (--start-only)."; exit 0; }

# ── Step 3: Load IFC data (skip if already populated) ─────────────────────────

echo ""
info "Checking graph state..."
node_count=$(neo4j_count "MATCH (n:IFCElement) RETURN count(n) AS c")
info "Current IFCElement nodes: ${node_count}"

if [ "$RELOAD" = true ] || [ "$node_count" -lt 100 ]; then
    if [ "$RELOAD" = true ]; then
        info "-- reload requested: re-exporting IFC data --"
    else
        info "Graph appears empty — loading IFC data..."
    fi

    # ── Load AP (primary, clear existing) ──────────────────────────────
    if [ ! -f "$IFC_PATH_AP" ]; then
        fail "AP IFC file not found: ${IFC_PATH_AP}"
    fi
    info "Exporting AP (AdvancedProject) → Neo4j (clear existing)..."
    conda run -n mscd_demo python "${SCRIPT_DIR}/../src/ifc_export_cli.py" \
        --ifc "${IFC_PATH_AP}" \
        --uri "bolt://localhost:${BOLT_PORT}" \
        --password "${NEO4J_PASSWORD}" \
        || fail "AP IFC export failed"
    ok "AP export complete"

    # ── Load DXA (additive, no clear) ──────────────────────────────────
    if [ -f "$IFC_PATH_DXA" ]; then
        info "Exporting DXA (Duplex_A) → Neo4j (additive)..."
        conda run -n mscd_demo python "${SCRIPT_DIR}/../src/ifc_export_cli.py" \
            --ifc "${IFC_PATH_DXA}" \
            --uri "bolt://localhost:${BOLT_PORT}" \
            --password "${NEO4J_PASSWORD}" \
            --no-clear \
            || warn "DXA IFC export failed — continuing"
        ok "DXA export complete"
    else
        warn "DXA IFC file not found: ${IFC_PATH_DXA} — skipping"
    fi

    # ── Load BH (additive, no clear) ──────────────────────────────────
    if [ -f "$IFC_PATH_BH" ]; then
        info "Exporting BH (BasicHouse) → Neo4j (additive)..."
        conda run -n mscd_demo python "${SCRIPT_DIR}/../src/ifc_export_cli.py" \
            --ifc "${IFC_PATH_BH}" \
            --uri "bolt://localhost:${BOLT_PORT}" \
            --password "${NEO4J_PASSWORD}" \
            --no-clear \
            || warn "BH IFC export failed — continuing"
        ok "BH export complete"
    else
        warn "BH IFC file not found: ${IFC_PATH_BH} — skipping"
    fi
else
    ok "Graph already populated (${node_count} nodes) — skipping IFC export"
fi

# ── Step 4: Topology enrichment (ADJACENT_TO + CONTINUOUS) ────────────────────

echo ""
adj_count=$(neo4j_count "MATCH ()-[:ADJACENT_TO]->() RETURN count(*) AS c")
info "Current ADJACENT_TO edges: ${adj_count}"

if [ "$RELOAD" = true ] || [ "$adj_count" -lt 100 ]; then
    # Run topology enrichment for each model that has an element index
    for model_label in AP DXA BH; do
        local_index=""
        local_ifc=""
        case "$model_label" in
            AP)  local_index="$INDEX_PATH_AP"; local_ifc="$IFC_PATH_AP" ;;
            DXA) local_index="$INDEX_PATH_DXA"; local_ifc="$IFC_PATH_DXA" ;;
            BH)  local_index="$INDEX_PATH_BH"; local_ifc="$IFC_PATH_BH" ;;
        esac

        if [ ! -f "$local_index" ]; then
            warn "$model_label index not found: $local_index — skipping topology"
            continue
        fi
        if [ ! -f "$local_ifc" ]; then
            warn "$model_label IFC not found: $local_ifc — skipping topology"
            continue
        fi

        info "Running topology enrichment for $model_label..."
        conda run -n mscd_demo python add_topology_edges.py \
            --index "$local_index" \
            --threshold 1500.0 \
            --ifc "$local_ifc" \
            --uri "bolt://localhost:${BOLT_PORT}" \
            --password "$NEO4J_PASSWORD" \
            || warn "$model_label topology enrichment failed — continuing"
        ok "$model_label topology enrichment complete"
    done
else
    ok "Topology already present (${adj_count} ADJACENT_TO edges) — skipping"
fi

# ── Step 5: Verification ──────────────────────────────────────────────────────

echo ""
echo "============================================="
echo "  Graph State (verification)"
echo "============================================="
nodes=$(neo4j_count "MATCH (n:IFCElement) RETURN count(n) AS c")
fills=$(neo4j_count "MATCH ()-[:FILLS]->() RETURN count(*) AS c")
adj=$(neo4j_count "MATCH ()-[:ADJACENT_TO]->() RETURN count(*) AS c")
conn=$(neo4j_count "MATCH ()-[:CONNECTS_TO]->() RETURN count(*) AS c")
cont=$(neo4j_count "MATCH (n:IFCElement) WHERE n.is_continuous=true RETURN count(n) AS c")

echo "  IFCElement nodes : ${nodes}   (expect ~1454: AP=1233 + DXA=168 + BH=53)"
echo "  FILLS edges      : ${fills}   (expect ~420+: AP=389 + DXA + BH)"
echo "  ADJACENT_TO edges: ${adj}   (expect ~500+, bidirectional)"
echo "  CONNECTS_TO edges: ${conn}   (expect ~1400+, bidirectional)"
echo "  is_continuous    : ${cont}   (expect ~150+)"
echo ""

# Warn if numbers are far off
if [ "$nodes" -lt 1400 ]; then
    warn "Node count (${nodes}) is lower than expected (~1454 for 3 models) — consider ./script/neo4j_init.sh --reload"
fi
if [ "$fills" -eq 0 ]; then
    warn "No FILLS edges found — IFC export may have failed"
fi

ok "Neo4j ready. Config: bolt://localhost:${BOLT_PORT}  auth: neo4j / ${NEO4J_PASSWORD}"
echo "Browser: http://localhost:${HTTP_PORT}/browser/  (or use your WSL IP from Windows)"
echo ""
echo "To run tests:"
echo "  conda run -n mscd_demo python test/test_priority0_retrieval.py"
echo "  conda run -n mscd_demo python test/test_e2e_pipeline_trace.py"
echo "  conda run -n mscd_demo python eval/h2_eval.py"
