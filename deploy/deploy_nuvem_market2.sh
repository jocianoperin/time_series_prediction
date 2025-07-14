#!/bin/bash
source "$(dirname "$0")/config.env"

DEPLOY_DIR="$(dirname "$0")"
PROJDIR="$(realpath "$DEPLOY_DIR/..")"
KEY="$HOME/.ssh/id_rsa_vast"
PROJNAME=$(basename "$PROJDIR")
TEMP_ZIP="/tmp/${PROJNAME}_data_$(date +%s).zip"

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
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@$IP -L 8080:localhost:8080 -N &
sleep 2

# === TESTA SE A CHAVE FUNCIONA ===
echo "[INFO] Testando acesso via chave SSH..."
if ! ssh -i "$KEY" -p $PORT -o BatchMode=yes -o ConnectTimeout=5 root@$IP "echo '[OK] Acesso via chave OK'" 2>/dev/null; then
  echo "[ERRO] A chave ainda não foi aceita. Verifique se colou corretamente e tente novamente."
  exit 1
fi

# === ATUALIZA ~/.ssh/config PARA O HOST vast ===
echo "[INFO] Atualizando ~/.ssh/config com entrada para VS Code..."

cat > ~/.ssh/config <<EOF
Host vast
  HostName $IP
  Port $PORT
  User root
  IdentityFile $KEY
EOF

# === CRIA ARQUIVO ZIP COM AS PASTAS DE DADOS ESPECÍFICAS ===
echo "[INFO] Compactando pastas de dados específicas..."
cd "$PROJDIR"
zip -r "$TEMP_ZIP" "data/plots/XGBoost" "data/predictions/XGBoost" "data/raw/Market 2"

# === ENVIA PROJETO COM RSYNC EXCLUINDO PASTAS DESNECESSÁRIAS ===
echo "[INFO] Enviando projeto $PROJNAME para o servidor..."

# Primeiro envia o projeto sem os dados
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

# Envia o arquivo ZIP com os dados
echo "[INFO] Enviando arquivo ZIP com os dados..."
scp -i "$KEY" -P $PORT "$TEMP_ZIP" root@$IP:/workspace/$PROJNAME/

# Extrai os dados no servidor remoto
echo "[INFO] Extraindo dados no servidor remoto..."
ssh -i "$KEY" -p $PORT root@$IP "cd /workspace/$PROJNAME && unzip -o $(basename "$TEMP_ZIP") && rm $(basename "$TEMP_ZIP")"

# Remove o arquivo ZIP local
echo "[INFO] Removendo arquivo ZIP temporário local..."
rm "$TEMP_ZIP"

# === ENVIA SCRIPT REMOTO ===
scp -i "$KEY" -P $PORT "$DEPLOY_DIR/remote_setup.sh" root@$IP:/root/

# === EXECUTA SETUP REMOTO ===
ssh -i "$KEY" -p $PORT root@$IP "PROJECT_NAME=$PROJNAME bash /root/remote_setup.sh"

echo ""
echo "[✔] Deploy finalizado. Acesse via VS Code Remote SSH: vast"

# === LIMPEZA ===
# Remove o arquivo ZIP do servidor remoto
ssh -i "$KEY" -p $PORT root@$IP "rm -f /workspace/$PROJNAME/$(basename "$TEMP_ZIP")"
