#!/usr/bin/env bash
set -euo pipefail

# Diretórios relativos a partir de deploy/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../data/raw"
PROCESSED_DIR="$RAW_DIR/processed"
PRED_DIR="$SCRIPT_DIR/../data/predictions/comparativo"

# Garante que exista a pasta processed
mkdir -p "$PROCESSED_DIR"

# Conta total de arquivos .csv em raw
total_raw=$(find "$RAW_DIR" -maxdepth 1 -type f -name 'produto_*.csv' | wc -l)
echo "$total_raw arquivos localizados em '$RAW_DIR'"

# Percorre cada pasta de comparativo e identifica as completadas
complete_dirs=0
for dir in "$PRED_DIR"/*/; do
  [ -d "$dir" ] || continue
  # Conta CSVs dentro da pasta
  count_csv=$(find "$dir" -maxdepth 1 -type f -name '*.csv' | wc -l)
  if [ "$count_csv" -eq 14 ]; then
    complete_dirs=$((complete_dirs + 1))
  fi
done
echo "Desses, $complete_dirs já foram totalmente processados"

# Move apenas os raw completos, contando quantos foram movidos
moved=0
for dir in "$PRED_DIR"/*/; do
  [ -d "$dir" ] || continue
  count_csv=$(find "$dir" -maxdepth 1 -type f -name '*.csv' | wc -l)
  if [ "$count_csv" -eq 14 ]; then
    barcode=$(basename "$dir")
    raw_file="$RAW_DIR/produto_${barcode}.csv"
    if [ -f "$raw_file" ]; then
      mv "$raw_file" "$PROCESSED_DIR/"
      moved=$((moved + 1))
      echo "✅ Movido: produto_${barcode}.csv → processed/"
    else
      echo "⚠️  Não encontrado: produto_${barcode}.csv"
    fi
  fi
done
echo "$moved arquivos movidos para '$PROCESSED_DIR'"

# Calcula quantos ainda restam na raw
remaining=$(( total_raw - moved ))
echo "$remaining arquivos restantes na pasta raw"
