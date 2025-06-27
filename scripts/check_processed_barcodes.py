import os
import re
from pathlib import Path

# Caminhos
base_path = Path('/home/jociano/Projects/time_series_prediction')
raw_path = base_path / 'data' / 'raw'
processed_raw_path = raw_path / 'processed'
predictions_path = base_path / 'data' / 'predictions' / 'comparativo'
plots_path = base_path / 'data' / 'plots' / 'comparativo'

# Função para extrair barcodes dos nomes de arquivos
def get_barcodes_from_files(files):
    barcodes = set()
    for file in files:
        # Extrai apenas o número do barcode do nome do arquivo
        match = re.search(r'produto_(\d+)\.csv', file.name)
        if match:
            barcodes.add(match.group(1))
    return barcodes

def get_all_barcodes():
    # Barcodes na pasta raw (exceto processed)
    raw_files = list(raw_path.glob('produto_*.csv'))
    raw_barcodes = get_barcodes_from_files(raw_files)
    
    # Barcodes já em raw/processed
    processed_files = list(processed_raw_path.glob('produto_*.csv'))
    processed_barcodes = get_barcodes_from_files(processed_files)
    
    # Barcodes em predictions/comparativo e plots/comparativo
    pred_dirs = {d.name for d in predictions_path.iterdir() if d.is_dir()}
    plots_dirs = {d.name for d in plots_path.iterdir() if d.is_dir()}
    processed_dirs = pred_dirs.intersection(plots_dirs)
    
    return {
        'raw': raw_barcodes,
        'processed': processed_barcodes,
        'processed_dirs': processed_dirs
    }

def analyze_barcodes():
    data = get_all_barcodes()
    
    # Barcodes que podem ser movidos (estão em raw e processados)
    can_move = data['raw'].intersection(data['processed_dirs'])
    
    # Barcodes que já foram movidos (estão em processed e processados)
    already_moved = data['processed'].intersection(data['processed_dirs'])
    
    # Barcodes processados que não estão em lugar nenhum
    only_processed = data['processed_dirs'] - data['raw'] - data['processed']
    
    return {
        'can_move': sorted(can_move),
        'already_moved': sorted(already_moved),
        'only_processed': sorted(only_processed),
        'total_raw': len(data['raw']),
        'total_processed_files': len(data['processed']),
        'total_processed_dirs': len(data['processed_dirs'])
    }

if __name__ == '__main__':
    results = analyze_barcodes()
    
    print(f"Total de arquivos na pasta raw: {results['total_raw']}")
    print(f"Total de arquivos em raw/processed: {results['total_processed_files']}")
    print(f"Total de pastas processadas (em predictions e plots): {results['total_processed_dirs']}")
    
    print(f"\nBarcodes que podem ser movidos: {len(results['can_move'])}")
    print("\n".join(results['can_move']))
    
    print(f"\nBarcodes já movidos: {len(results['already_moved'])}")
    
    print(f"\nBarcodes apenas nas pastas processadas: {len(results['only_processed'])}")
    print("\n".join(results['only_processed']))
