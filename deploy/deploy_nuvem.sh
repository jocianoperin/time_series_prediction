#!/bin/bash
source "$(dirname "$0")/config.env"

DEPLOY_DIR="$(dirname "$0")"
PROJDIR="$(realpath "$DEPLOY_DIR/..")"
KEY="$HOME/.ssh/id_rsa_vast"
PROJNAME=$(basename "$PROJDIR")

# === GERA CHAVE SE NÃO EXISTIR ===
mkdir -p ~/.ssh
if [ ! -f "$KEY" ]; then
  ssh-keygen -t rsa -b 4096 -f "$KEY" -N ""
  echo "[✔] Chave SSH gerada em $KEY"
fi

# === MOSTRA A CHAVE PARA INSERIR MANUALMENTE NA VAST ===
echo ""
echo "[⚠️ ] Copie a chave abaixo e cole no terminal da instância Vast.ai:"
echo "--------------------------------------------------------------"
cat "$KEY.pub"
echo "--------------------------------------------------------------"
read -p "[ENTER] Pressione Enter após inserir a chave pública via terminal da instância..."
echo "[INFO] Executando handshake via SSH para registrar fingerprint local..."
#ssh -p $PORT root@$IP -L 8080:localhost:8080 -N &
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@$IP -L 8080:localhost:8080 -N &
sleep 2

# === TESTA SE A CHAVE FUNCIONA ===
echo "[INFO] Testando acesso via chave SSH..."
if ! ssh -i "$KEY" -p $PORT -o BatchMode=yes -o ConnectTimeout=5 root@$IP "echo '[OK] Acesso via chave OK'" 2>/dev/null; then
  echo "[ERRO] A chave ainda não foi aceita. Verifique se colou corretamente e tente novamente."
  exit 1
fi

# === ADICIONA CONFIGURAÇÃO AO ~/.ssh/config ===
HOST_NAME="vast_${PORT}"  # Nome único baseado na porta
echo "[INFO] Adicionando configuração para $HOST_NAME ao ~/.ssh/config..."

# Cria o diretório .ssh se não existir
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Remove configuração existente para este host se já existir
if grep -q "^Host $HOST_NAME" ~/.ssh/config 2>/dev/null; then
  echo "[INFO] Atualizando configuração existente para $HOST_NAME..."
  # Remove a seção existente
  sed -i "/^Host $HOST_NAME$/,/^$/d" ~/.ssh/config
  # Remove linhas em branco extras
  sed -i '/^$/N;/^\n$/D' ~/.ssh/config
fi

# Adiciona nova configuração
echo -e "\n# Configuração para $HOST_NAME ($(date '+%Y-%m-%d %H:%M:%S'))" >> ~/.ssh/config
cat >> ~/.ssh/config <<EOF
Host $HOST_NAME
  HostName $IP
  Port $PORT
  User root
  IdentityFile $KEY
  StrictHostKeyChecking no
  UserKnownHostsFile=/dev/null
EOF

chmod 600 ~/.ssh/config

echo "[OK] Configuração adicionada para $HOST_NAME. Use: ssh $HOST_NAME"

# === ENVIA PROJETO COM RSYNC EXCLUINDO PASTAS DESNECESSÁRIAS ===
echo "[INFO] Enviando projeto $PROJNAME para o servidor com exclusões..."

rsync -avz -e "ssh -i $KEY -p $PORT" \
  --exclude 'models' \
  --exclude 'logs' \
  --exclude 'data' \
  --exclude 'deploy' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '*.log' \
  "$PROJDIR/" root@$IP:/workspace/$PROJNAME/

# === ENVIA SCRIPT REMOTO ===
scp -i "$KEY" -P $PORT "$DEPLOY_DIR/remote_setup.sh" root@$IP:/root/

# === EXECUTA SETUP REMOTO ===
ssh -i "$KEY" -p $PORT root@$IP "PROJECT_NAME=$PROJNAME bash /root/remote_setup.sh"

echo ""
echo "[✔] Deploy finalizado. Acesse via VS Code Remote SSH: vast"
