#!/usr/bin/env bash
set -euo pipefail
cp /solution/inhibitor_fit.py /app/inhibitor_fit.py
python /app/inhibitor_fit.py --indir /app/data --outdir /app/out
