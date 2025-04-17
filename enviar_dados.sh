#!/bin/bash

# === CONFIGURAÇÕES ===
IP="<IP_DA_INSTANCIA>"
PORT="<PORTA_SSH>"
KEY="$HOME/.ssh/id_rsa_vast"
DEST="/workspace/time_series_prediction/data"

# === PASTAS A SEREM ENVIADAS ===
PASTAS=("raw" "promotion_data")  # <-- ajuste aqui as pastas que deseja enviar

for pasta in "${PASTAS[@]}"; do
  if [ -d "./data/$pasta" ]; then
    echo "[INFO] Enviando ./data/$pasta para o servidor..."
    rsync -avz -e "ssh -i $KEY -p $PORT" "./data/$pasta/" root@$IP:$DEST/$pasta/
  else
    echo "[AVISO] Pasta ./data/$pasta não encontrada localmente. Ignorando."
  fi
done

echo "[✔] Envio concluído."
