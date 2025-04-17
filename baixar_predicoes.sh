#!/bin/bash

# === CONFIGURAÇÕES ===
IP="<IP_DA_INSTANCIA>"
PORT="<PORTA_SSH>"
KEY="$HOME/.ssh/id_rsa_vast"
ORIGEM="/workspace/time_series_prediction/data/predictions"
DESTINO_LOCAL="./data/predictions"

mkdir -p $DESTINO_LOCAL

echo "[INFO] Baixando predições da nuvem para $DESTINO_LOCAL ..."
rsync -avz -e "ssh -i $KEY -p $PORT" root@$IP:$ORIGEM/ $DESTINO_LOCAL/

echo "[✔] Download concluído."
