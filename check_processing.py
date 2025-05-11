#!/usr/bin/env python3
import os
import re

# Diretórios base
processed_dir = 'data/processed'
pred_base      = 'data/predictions/XGBoost'
plot_base      = 'data/plots/XGBoost'

# Meses esperados (01 a 12)
expected_months = [f'{i:02d}' for i in range(1, 13)]

# Expressões regulares
prod_pattern  = re.compile(r'^produto_(\d+)\.csv$')
pred_pattern  = re.compile(r'XGBoost_daily_\d+_(\d{4})_(\d{2})\.csv')
plot_pattern  = re.compile(r'XGBoost_\d+_(\d{4})_(\d{2})\.png')

# Listas de resultado
complete   = []
incomplete = []

for fname in os.listdir(processed_dir):
    m = prod_pattern.match(fname)
    if not m:
        continue
    barcode = m.group(1)

    # Verifica predictions
    pred_dir = os.path.join(pred_base, barcode)
    if os.path.isdir(pred_dir):
        pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.csv')]
        months_pred = {pred_pattern.match(f).group(2)
                       for f in pred_files
                       if pred_pattern.match(f)}
    else:
        months_pred = set()

    # Verifica plots
    plot_dir = os.path.join(plot_base, barcode)
    if os.path.isdir(plot_dir):
        plot_files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
        months_plot = {plot_pattern.match(f).group(2)
                       for f in plot_files
                       if plot_pattern.match(f)}
    else:
        months_plot = set()

    # Descobre meses faltantes
    missing_pred = [m for m in expected_months if m not in months_pred]
    missing_plot = [m for m in expected_months if m not in months_plot]

    # Classifica como completo ou incompleto
    if not missing_pred and not missing_plot:
        complete.append(barcode)
    else:
        parts = []
        if missing_pred:
            parts.append(f"faltam predictions nos meses: {', '.join(missing_pred)}")
        if missing_plot:
            parts.append(f"faltam plots nos meses:      {', '.join(missing_plot)}")
        incomplete.append(f"Produto {barcode}: " + ' | '.join(parts))

# Gera o relatório em texto
with open('processing_report.txt', 'w') as f:
    f.write('=== Produtos 100% processados ===\n')
    for b in complete:
        f.write(f"- {b}\n")
    f.write('\n=== Produtos com divergências ===\n')
    for line in incomplete:
        f.write(f"- {line}\n")

print("Relatório gerado em: processing_report.txt")
