#!/usr/bin/env bash
set -euo pipefail
cp /solution/design_variant.py /app/design_variant.py
python /app/design_variant.py --indir /app/data --outdir /app/out
