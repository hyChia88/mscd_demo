23/2/2026:
- Fix 3D Viewer issue:
    - Reduce latecy from ??s to ~0ms (resource cache)/~1s (browser HTTP cache)
neo4j init:
```
NEO4J_HOME=/tmp/neo4j-community-5.26.0
# Set initial password non-interactively
$NEO4J_HOME/bin/neo4j-admin dbms set-initial-password password 2>&1
# Start
$NEO4J_HOME/bin/neo4j start 2>&1
sleep 8
# Verify bolt port
ss -tlnp | grep 7687 || echo "Port not up yet"
```

#TODO:
- Explain query later: make sure no silent fallback and explain what is the template use for.