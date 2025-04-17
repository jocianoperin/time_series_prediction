#!/bin/bash
source "$(dirname "$0")/config.env"

KEY="$HOME/.ssh/id_rsa_vast"
DEST="/workspace/time_series_prediction/data"
SOURCE_BASE="$(dirname "$0")/../data"
PASTAS=("raw" "promotion_data")

for pasta in "${PASTAS[@]}"; do
  if [ -d "$SOURCE_BASE/$pasta" ]; then
    echo "[INFO] Enviando $SOURCE_BASE/$pasta para o servidor..."
    rsync -avz -e "ssh -i $KEY -p $PORT" "$SOURCE_BASE/$pasta/" root@$IP:$DEST/$pasta/
  else
    echo "[AVISO] Pasta $SOURCE_BASE/$pasta não encontrada. Ignorando."
  fi
done

echo "[✔] Envio concluído."
