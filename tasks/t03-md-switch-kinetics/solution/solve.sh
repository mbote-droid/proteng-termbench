#!/usr/bin/env bash
set -euo pipefail
cp /solution/switch_analysis.py /app/switch_analysis.py
python /app/switch_analysis.py --indir /app/data --outdir /app/out
