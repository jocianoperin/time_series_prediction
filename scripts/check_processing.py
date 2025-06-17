#!/usr/bin/env python3
import os
import re
import shutil

# Diretórios base
data_raw        = 'data/raw'
processed_dir   = 'data/processed'
pred_base       = 'data/predictions/XGBoost'
plot_base       = 'data/plots/XGBoost'
problems_dir    = os.path.join(data_raw, 'problems')

# Garante que a pasta de problemas existe
os.makedirs(problems_dir, exist_ok=True)

# Meses esperados (01 a 12)
expected_months = [f'{i:02d}' for i in range(1, 13)]

# Expressões regulares
prod_pattern    = re.compile(r'^produto_(\d+)\.csv$')
pred_pattern    = re.compile(r'XGBoost_daily_\d+_(\d{4})_(\d{2})\.csv')
plot_pattern    = re.compile(r'XGBoost_\d+_(\d{4})_(\d{2})\.png')

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

    # Classifica e move arquivos problemáticos
    proc_file = os.path.join(processed_dir, fname)
    raw_file  = os.path.join(data_raw, fname)

    if not missing_pred and not missing_plot:
        complete.append(barcode)
    else:
        parts = []
        if missing_pred:
            parts.append(f"faltam predictions nos meses: {', '.join(missing_pred)}")
        if missing_plot:
            parts.append(f"faltam plots nos meses:      {', '.join(missing_plot)}")
        incomplete.append(f"Produto {barcode}: " + ' | '.join(parts))

        # Tenta mover o raw CSV; se não existir, move o processed CSV
        if os.path.exists(raw_file):
            shutil.move(raw_file, os.path.join(problems_dir, fname))
        elif os.path.exists(proc_file):
            shutil.move(proc_file, os.path.join(problems_dir, fname))

# Gera o relatório em texto
with open('processing_report.txt', 'w') as f:
    f.write('=== Produtos 100% processados ===\n')
    for b in complete:
        f.write(f"- {b}\n")
    f.write('\n=== Produtos com divergências ===\n')
    for line in incomplete:
        f.write(f"- {line}\n")

print(f"Relatório gerado em: processing_report.txt")
print(f"Arquivos problemáticos movidos para: {problems_dir}")
