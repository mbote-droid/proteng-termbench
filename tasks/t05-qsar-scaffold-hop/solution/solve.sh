#!/usr/bin/env bash
set -euo pipefail
cp /solution/qsar_model.py /app/qsar_model.py
python /app/qsar_model.py --indir /app/data --outdir /app/out
