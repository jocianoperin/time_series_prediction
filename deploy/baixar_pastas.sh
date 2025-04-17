#!/bin/bash
source "$(dirname "$0")/config.env"

KEY="$HOME/.ssh/id_rsa_vast"
ORIGEM_BASE="/workspace/time_series_prediction/data"
DESTINO_BASE="$(dirname "$0")/../data"
PASTAS=("predictions" "plots")

for pasta in "${PASTAS[@]}"; do
  DESTINO_LOCAL="$DESTINO_BASE/$pasta"
  mkdir -p "$DESTINO_LOCAL"
  echo "[INFO] Baixando $pasta da nuvem para $DESTINO_LOCAL ..."
  rsync -avz -e "ssh -i $KEY -p $PORT" root@$IP:$ORIGEM_BASE/$pasta/ $DESTINO_LOCAL/
done

echo "[✔] Download concluído."
