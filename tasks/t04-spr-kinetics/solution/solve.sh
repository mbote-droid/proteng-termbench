#!/usr/bin/env bash
set -euo pipefail
cp /solution/spr_fit.py /app/spr_fit.py
python /app/spr_fit.py --indir /app/data --outdir /app/out
