#!/bin/bash

PROJNAME=${PROJECT_NAME:-time_series_prediction}
WORKDIR="/workspace/$PROJNAME"

if [ ! -d "/root/miniconda3" ]; then
  echo "[INFO] Instalando Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p /root/miniconda3
  echo 'export PATH="/root/miniconda3/bin:$PATH"' >> ~/.bashrc
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda init bash

cd "$WORKDIR" || { echo "[ERRO] Diretório do projeto não encontrado: $WORKDIR"; exit 1; }

if [ -f "environment.yml" ]; then
  conda env create -n tsenv -f environment.yml || conda env update -n tsenv -f environment.yml --prune
else
  conda create -n tsenv python=3.10 -y
  conda activate tsenv
  if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
  fi
fi

echo "[✔] Ambiente tsenv pronto e ativado. Projeto disponível em $WORKDIR"
