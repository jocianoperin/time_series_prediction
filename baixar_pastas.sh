#!/bin/bash

# === CONFIGURAÇÕES ===
IP="<IP_DA_INSTANCIA>"
PORT="<PORTA_SSH>"
KEY="$HOME/.ssh/id_rsa_vast"
ORIGEM_BASE="/workspace/time_series_prediction/data"
DESTINO_BASE="./data"

# === PASTAS A SEREM BAIXADAS ===
PASTAS=("predictions" "raw" "promotion_data")  # <-- ajuste aqui as pastas desejadas

for pasta in "${PASTAS[@]}"; do
  DESTINO_LOCAL="$DESTINO_BASE/$pasta"
  mkdir -p "$DESTINO_LOCAL"
  echo "[INFO] Baixando $pasta da nuvem para $DESTINO_LOCAL ..."
  rsync -avz -e "ssh -i $KEY -p $PORT" root@$IP:$ORIGEM_BASE/$pasta/ $DESTINO_LOCAL/
done

echo "[✔] Download de todas as pastas concluído."
