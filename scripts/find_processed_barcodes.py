import os
import re
from pathlib import Path

# Caminhos
base_path = Path('/home/jociano/Projects/time_series_prediction')
raw_path = base_path / 'data' / 'raw'
predictions_path = base_path / 'data' / 'predictions' / 'comparativo'
plots_path = base_path / 'data' / 'plots' / 'comparativo'

# Função para extrair barcodes dos nomes de arquivos

def get_raw_barcodes():
    raw_files = list(raw_path.glob('produto_*.csv'))
    barcodes = set()
    for file in raw_files:
        # Extrai apenas o número do barcode do nome do arquivo
        match = re.search(r'produto_(\d+)\.csv', file.name)
        if match:
            barcodes.add(match.group(1))
    return barcodes

# Função para obter barcodes das pastas processadas

def get_processed_barcodes():
    # Pega as pastas em predictions/comparativo
    pred_dirs = [d.name for d in predictions_path.iterdir() if d.is_dir()]
    # Pega as pastas em plots/comparativo
    plots_dirs = [d.name for d in plots_path.iterdir() if d.is_dir()]
    # Encontra a interseção entre as pastas de predictions e plots
    processed_barcodes = set(pred_dirs).intersection(plots_dirs)
    return processed_barcodes

# Encontra barcodes que estão processados e ainda estão na pasta raw

def find_barcodes_to_move():
    raw_barcodes = get_raw_barcodes()
    processed_barcodes = get_processed_barcodes()
    
    # Barcodes que estão processados e ainda estão na pasta raw
    to_move = sorted(raw_barcodes.intersection(processed_barcodes))
    return to_move

if __name__ == '__main__':
    barcodes = find_barcodes_to_move()
    print(f"Total de barcodes para mover: {len(barcodes)}")
    print("\n".join(barcodes))
