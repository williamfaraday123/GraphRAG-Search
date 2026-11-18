#!/bin/bash
# Loads the wiki snapshot's Neo4j dump into the neo4j_data volume, before Neo4j first starts.
# Runs as a one-shot container on every `docker compose up`; does nothing after the first
# load, and never touches a volume that already holds a database (an existing install keeps
# its own data, including uploads).
set -euo pipefail

SNAPSHOT=/wiki-snapshot
DONE=/data/.wiki-snapshot
LOADING=/data/.wiki-snapshot-loading
ID=$(sed -n 's/.*"snapshot_id": *"\([^"]*\)".*/\1/p' "$SNAPSHOT/manifest.json")

if [ -f "$DONE" ]; then
  echo "Wiki graph already loaded (snapshot $(cat "$DONE"))."
  exit 0
fi
if [ -d /data/databases/neo4j ] && [ ! -f "$LOADING" ]; then
  echo "Neo4j already has a database; leaving it as it is. The wiki snapshot is only loaded into a new install."
  exit 0
fi

echo "Loading wiki snapshot $ID into Neo4j..."
touch "$LOADING"   # if the load is interrupted, the next start redoes it
neo4j-admin database load neo4j --from-path="$SNAPSHOT" --overwrite-destination=true
chown -R neo4j:neo4j /data
echo "$ID" > "$DONE"
rm -f "$LOADING"
echo "Wiki graph loaded."
