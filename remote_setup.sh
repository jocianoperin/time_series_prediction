#!/bin/bash

# === INSTALA MINICONDA ===
if [ ! -d "/root/miniconda3" ]; then
  echo "[INFO] Instalando Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p /root/miniconda3
  echo 'export PATH="/root/miniconda3/bin:$PATH"' >> ~/.bashrc
fi

# === ATIVA CONDA ===
source /root/miniconda3/etc/profile.d/conda.sh
conda init bash

# === PREPARA AMBIENTE ===
cd /workspace/time_series_prediction
if [ -f "environment.yml" ]; then
  conda env create -n tsenv -f environment.yml || conda env update -n tsenv -f environment.yml --prune
else
  conda create -n tsenv python=3.10 -y
  conda activate tsenv
  pip install -r requirements.txt
fi

# === ATIVA E RODA MAIN ===
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tsenv
cd /workspace/time_series_prediction
python main.py
