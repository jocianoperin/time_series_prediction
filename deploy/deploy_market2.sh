#!/bin/bash
# Script para deploy do Market 2 na nuvem
# Cria a estrutura de pastas necessária e copia os dados de predição do XGBoost

# Carrega as configurações
source "$(dirname "$0")/config.env"

# Configurações
DEPLOY_DIR="$(dirname "$0")"
PROJDIR="$(realpath "$DEPLOY_DIR/..")"
KEY="$HOME/.ssh/id_rsa_vast"
PROJNAME=$(basename "$PROJDIR")
TEMP_DIR="/tmp/${PROJNAME}_deploy_$(date +%s)"
TEMP_ZIP="${TEMP_DIR}/market2_data.zip"

# Função para limpar arquivos temporários
cleanup() {
    echo "[INFO] Removendo arquivos temporários..."
    rm -rf "$TEMP_DIR"
}

# Configura o trap para limpeza em caso de erro
trap cleanup EXIT

# === VERIFICAÇÕES INICIAIS ===

# Verifica se a chave SSH existe
mkdir -p ~/.ssh
if [ ! -f "$KEY" ]; then
    ssh-keygen -t rsa -b 4096 -f "$KEY" -N ""
    echo "[✔] Chave SSH gerada em $KEY"
    echo ""
    echo "[⚠️ ] Copie a chave abaixo e cole no terminal da instância Vast.ai:"
    echo "--------------------------------------------------------------"
    cat "$KEY.pub"
    echo "--------------------------------------------------------------"
    read -p "[ENTER] Pressione Enter após inserir a chave pública via terminal da instância..."
    echo "[INFO] Executando handshake via SSH para registrar fingerprint local..."
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@$IP -L 8080:localhost:8080 -N &
    sleep 2
fi

# Testa a conexão SSH
echo "[INFO] Testando acesso via chave SSH..."
if ! ssh -i "$KEY" -p $PORT -o BatchMode=yes -o ConnectTimeout=5 root@$IP "echo '[OK] Acesso via chave OK'" 2>/dev/null; then
    echo "[ERRO] A chave ainda não foi aceita. Verifique se colou corretamente e tente novamente."
    exit 1
fi

# === ATUALIZA ~/.ssh/config PARA O HOST vast ===
echo "[INFO] Atualizando ~/.ssh/config com entrada para VS Code..."

mkdir -p ~/.ssh
chmod 700 ~/.ssh
cat > ~/.ssh/config <<EOF
Host vast
  HostName $IP
  Port $PORT
  User root
  IdentityFile $KEY
  StrictHostKeyChecking no
  UserKnownHostsFile=/dev/null
EOF

# Garante as permissões corretas
chmod 600 ~/.ssh/config
chmod 600 "$KEY"
chmod 644 "$KEY.pub"

# === PREPARA OS DADOS LOCALMENTE ===

echo "[INFO] Preparando estrutura de pastas..."
mkdir -p "$TEMP_DIR/data/predictions/XGBoost"
mkdir -p "$TEMP_DIR/data/raw/Market 2"
mkdir -p "$TEMP_DIR/data/plots/XGBoost"

# Copia os arquivos de predição do XGBoost do Market 2
echo "[INFO] Copiando arquivos de predição do XGBoost..."
if [ -d "$PROJDIR/data/predictions/Market 2/XGBoost" ]; then
    cp -r "$PROJDIR/data/predictions/Market 2/XGBoost/"* "$TEMP_DIR/data/predictions/XGBoost/"
    echo "[OK] Arquivos de predição copiados com sucesso!"
else
    echo "[ERRO] Pasta de predições não encontrada: $PROJDIR/data/predictions/Market 2/XGBoost"
    exit 1
fi

# Copia os dados brutos do Market 2 se existirem
if [ -d "$PROJDIR/data/raw/Market 2" ]; then
    echo "[INFO] Copiando dados brutos do Market 2..."
    cp -r "$PROJDIR/data/raw/Market 2/"* "$TEMP_DIR/data/raw/Market 2/"
fi

# Cria o arquivo ZIP
echo "[INFO] Criando arquivo ZIP com os dados..."
cd "$TEMP_DIR"
zip -r "$TEMP_ZIP" "data"

# === ENVIA O PROJETO PARA O SERVIDOR ===

echo "[INFO] Enviando projeto para o servidor..."

# Envia o projeto (excluindo pastas grandes)
rsync -avz -e "ssh -i $KEY -p $PORT" \
    --exclude 'models' \
    --exclude 'logs' \
    --exclude 'data' \
    --exclude 'deploy' \
    --exclude 'scripts' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '*.pyo' \
    --exclude '*.log' \
    "$PROJDIR/" root@$IP:/workspace/$PROJNAME/

# Envia o arquivo ZIP com os dados
echo "[INFO] Enviando arquivo ZIP com os dados..."
scp -i "$KEY" -P $PORT "$TEMP_ZIP" root@$IP:/workspace/$PROJNAME/

# === ENVIA E EXECUTA O SCRIPT DE CONFIGURAÇÃO REMOTA ===

echo "[INFO] Enviando script de configuração remota..."
scp -i "$KEY" -P $PORT "$DEPLOY_DIR/remote_setup.sh" root@$IP:/root/

# Comandos para executar no servidor remoto
echo "[INFO] Configurando ambiente no servidor..."
ssh -i "$KEY" -p $PORT root@$IP << 'ENDSSH'
    cd "/workspace/$PROJNAME" || exit 1
    
    echo "[REMOTO] Extraindo dados..."
    unzip -o "market2_data.zip" -d "/workspace/$PROJNAME"
    
    echo "[REMOTO] Criando estrutura de pastas adicional..."
    mkdir -p "data/models"
    mkdir -p "data/logs"
    
    echo "[REMOTO] Configurando permissões..."
    chmod -R 755 "/workspace/$PROJNAME"
    chmod +x /root/remote_setup.sh
    
    echo "[REMOTO] Executando configuração do ambiente..."
    PROJECT_NAME="$PROJNAME" /root/remote_setup.sh
    
    echo "[REMOTO] Limpando arquivos temporários..."
    rm -f "market2_data.zip"
    
    echo "[REMOTO] Verificando estrutura de pastas..."
    echo "Estrutura de pastas criada:"
    find "data" -type d | sort
    
    echo "[REMOTO] Configuração finalizada com sucesso!"
ENDSSH

# === FINALIZAÇÃO ===

echo ""
echo "[✔] Deploy do Market 2 concluído com sucesso!"
echo "    Acesse o servidor com: ssh -i $KEY -p $PORT root@$IP"
echo "    Pasta do projeto: /workspace/$PROJNAME"
echo ""

# Limpeza final
cleanup
