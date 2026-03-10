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
NEO4J_PASSWORD="password"
BOLT_PORT=7687
BOLT_WAIT_SECS=30

IFC_PATH="$(cd "$SCRIPT_DIR" && realpath ../../data_curation/ifc_models/AdvancedProject.ifc)"
INDEX_PATH="$(cd "$SCRIPT_DIR" && realpath ../../data_curation/references/element_index.jsonl)"

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
    cont=$(neo4j_count "MATCH (n:IFCElement) WHERE n.is_continuous=true RETURN count(n) AS c")
    echo "  IFCElement nodes : ${nodes}"
    echo "  FILLS edges      : ${fills}"
    echo "  ADJACENT_TO edges: ${adj}  (bidirectional)"
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

    if [ ! -f "$IFC_PATH" ]; then
        fail "IFC file not found: ${IFC_PATH}"
    fi

    info "Running IFC → Neo4j export (may take 30-60s)..."
    conda run -n mscd_demo python "${SCRIPT_DIR}/../src/ifc_export_cli.py" \
        --ifc "${IFC_PATH}" \
        --uri "bolt://localhost:${BOLT_PORT}" \
        --password "${NEO4J_PASSWORD}" \
        || fail "IFC export failed"

    ok "IFC export complete"
else
    ok "Graph already populated (${node_count} nodes) — skipping IFC export"
fi

# ── Step 4: Topology enrichment (ADJACENT_TO + CONTINUOUS) ────────────────────

echo ""
adj_count=$(neo4j_count "MATCH ()-[:ADJACENT_TO]->() RETURN count(*) AS c")
info "Current ADJACENT_TO edges: ${adj_count}"

if [ "$RELOAD" = true ] || [ "$adj_count" -lt 100 ]; then
    if [ ! -f "$INDEX_PATH" ]; then
        warn "element_index.jsonl not found: ${INDEX_PATH}"
        warn "Skipping topology enrichment. Run data_curation/scripts/synth/1_build_index.py first."
    else
        info "Running topology enrichment (ADJACENT_TO + CONTINUOUS)..."
        conda run -n mscd_demo python ../legacy/script/add_topology_edges.py \
            --index "$INDEX_PATH" \
            --threshold 1500.0 \
            --uri "bolt://localhost:${BOLT_PORT}" \
            --password "$NEO4J_PASSWORD"
        ok "Topology enrichment complete"
    fi
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
cont=$(neo4j_count "MATCH (n:IFCElement) WHERE n.is_continuous=true RETURN count(n) AS c")

echo "  IFCElement nodes : ${nodes}   (expect ~1257)"
echo "  FILLS edges      : ${fills}   (expect 389)"
echo "  ADJACENT_TO edges: ${adj}   (expect 466, bidirectional)"
echo "  is_continuous    : ${cont}   (expect 150)"
echo ""

# Warn if numbers are far off
if [ "$nodes" -lt 1000 ]; then
    warn "Node count (${nodes}) is lower than expected — consider ./script/neo4j_init.sh --reload"
fi
if [ "$fills" -eq 0 ]; then
    warn "No FILLS edges found — IFC export may have failed"
fi

ok "Neo4j ready. Config: bolt://localhost:${BOLT_PORT}  auth: neo4j / ${NEO4J_PASSWORD}"
echo ""
echo "To run tests:"
echo "  conda run -n mscd_demo python test/test_priority0_retrieval.py"
echo "  conda run -n mscd_demo python test/test_e2e_pipeline_trace.py"
echo "  conda run -n mscd_demo python eval/h2_eval.py"
