#!/bin/sh
# Fix volume ownership, then drop privileges.
#
# Fly volumes (and some other hosts) mount /data root-owned and empty; a
# plain `USER appuser` image can then never write the tracker DB or drafts.
# Under docker run / compose the chown is a harmless no-op. setpriv ships
# with util-linux in python:3.12-slim.
set -e
mkdir -p /data/output
# Seed the sample inventory (public, baked into the image) on first boot so
# hosts without a pre-loaded volume — e.g. ECS Fargate — start functional.
# Never overwrites: a real inventory on the volume always wins.
[ -f /data/inventory.yaml ] || cp "/app/Stage 1/data/experience_inventory.yaml" /data/inventory.yaml
chown -R appuser:appuser /data
export HOME=/home/appuser
exec setpriv --reuid=appuser --regid=appuser --init-groups "$@"
