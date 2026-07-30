#!/bin/sh
# Fix volume ownership, then drop privileges.
#
# Fly volumes (and some other hosts) mount /data root-owned and empty; a
# plain `USER appuser` image can then never write the tracker DB or drafts.
# Under docker run / compose the chown is a harmless no-op. setpriv ships
# with util-linux in python:3.12-slim.
set -e
mkdir -p /data/output
chown -R appuser:appuser /data
export HOME=/home/appuser
exec setpriv --reuid=appuser --regid=appuser --init-groups "$@"
