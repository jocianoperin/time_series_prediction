#!/usr/bin/env python3
"""Detecta códigos de barras que têm CSV bruto em Market 1 mas não têm pasta de previsões
XGBoost correspondente em data/predictions/Market 1/XGBoost.

Executa no root do projeto: python scripts/check_missing_preds.py
"""
from pathlib import Path
import re
import sys

RAW_DIR = Path("data/raw/Market 1")
PRED_DIR = Path("data/predictions/Market 1/XGBoost")

if not RAW_DIR.exists():
    sys.exit(f"Pasta não encontrada: {RAW_DIR}")
if not PRED_DIR.exists():
    sys.exit(f"Pasta não encontrada: {PRED_DIR}")

raw_barcodes = set()
for csv_path in RAW_DIR.glob("produto_*.csv"):
    m = re.match(r"produto_(.+)\.csv", csv_path.name)
    if m:
        raw_barcodes.add(m.group(1))

pred_barcodes = {p.name for p in PRED_DIR.iterdir() if p.is_dir()}

missing = sorted(raw_barcodes - pred_barcodes)

if missing:
    print("Barcodes sem pasta de previsão XGBoost:")
    for bc in missing:
        print(bc)
else:
    print("Nenhum barcode faltando – todas as previsões XGBoost presentes.")
