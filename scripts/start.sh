#!/usr/bin/env sh
set -eu

: "${DOUBAO_WORK_API_KEY:=sk-local-change-me}"
: "${DOUBAO_WORK_PORT:=18789}"
: "${DOUBAO_WORK_ADMIN_PASSWORD:=admin}"
export DOUBAO_WORK_API_KEY DOUBAO_WORK_PORT DOUBAO_WORK_ADMIN_PASSWORD

exec python3 -m doubao_work2api
