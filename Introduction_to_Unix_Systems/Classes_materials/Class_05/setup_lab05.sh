#!/usr/bin/env bash
set -eu

if (( $# > 1 )); then
    printf 'Usage: bash %s [new-workspace]\n' "${0##*/}" >&2
    exit 2
fi

workspace=${1:-unix-lab05}
if [[ -e "$workspace" || -L "$workspace" ]]; then
    printf 'Path already exists: %s\n' "$workspace" >&2
    printf 'Pass a new directory name as the first argument.\n' >&2
    exit 1
fi

mkdir -- "$workspace"
mkdir -- "$workspace/data"

cat > "$workspace/data/simulation.log" <<'LOG'
08:00 run01 saved OK
08:03 run02 exceeded time limit WARN
08:04 run02 timed out FAIL
08:07 run03 retried after a FAIL message OK
08:09 run04 low disk space warn
08:12 run04 write failed FAIL
08:15 run05 saved OK
LOG

cat > "$workspace/data/runs.txt" <<'RUNS'
run01;12;OK
run02;29;FAIL
run03;20;OK
run04;36;FAIL
run05;24;OK
RUNS

printf 'Created %s/data/simulation.log and %s/data/runs.txt\n' "$workspace" "$workspace"
