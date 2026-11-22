#!/usr/bin/env bash
set -euo pipefail
cp /solution/triage.py /app/triage.py
python /app/triage.py --indir /app/data --outdir /app/out
