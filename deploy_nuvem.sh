#!/bin/bash

# === INPUTS ===
for i in "$@"; do
  case $i in
    --ip=*) IP="${i#*=}" ;;
    --port=*) PORT="${i#*=}" ;;
    --project-dir=*) PROJDIR="${i#*=}" ;;
    *) ;;
  esac
done

IP=${IP:-"<IP_AQUI>"}
PORT=${PORT:-22}
PROJDIR=${PROJDIR:-"./time_series_prediction"}
KEY="$HOME/.ssh/id_rsa_vast"

# === GERA CHAVE SSH SE NÃO EXISTIR ===
mkdir -p ~/.ssh
if [ ! -f "$KEY" ]; then
  ssh-keygen -t rsa -b 4096 -f "$KEY" -N ""
fi

# === ENVIA A CHAVE PÚBLICA PARA O SERVIDOR ===
PUBKEY=$(cat "$KEY.pub")
ssh -i "$KEY" -p $PORT root@$IP "mkdir -p /root/.ssh && echo '$PUBKEY' >> /root/.ssh/authorized_keys && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys"

# === ENVIA PROJETO PARA /workspace ===
scp -i "$KEY" -P $PORT -r $PROJDIR root@$IP:/workspace/

# === ENVIA SCRIPT REMOTO ===
scp -i "$KEY" -P $PORT remote_setup.sh root@$IP:/root/

# === EXECUTA SCRIPT REMOTO ===
ssh -i "$KEY" -p $PORT root@$IP 'bash /root/remote_setup.sh'

echo ""
echo "[✔] Deploy finalizado com sucesso."
echo "Agora abra o VS Code e conecte em: root@$IP:$PORT"
