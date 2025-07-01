import os
import re
from pathlib import Path

def extract_barcodes_from_filenames(directory):
    barcodes = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('produto_') and file.endswith('.csv'):
                # Extrai o código de barras do nome do arquivo
                match = re.match(r'produto_([0-9]+)\.csv', file)
                if match:
                    barcodes.add(match.group(1))
    return barcodes

def read_barcodes_file(filepath):
    with open(filepath, 'r') as f:
        # Remove espaços em branco e linhas vazias
        return {line.strip() for line in f if line.strip()}

def main():
    # Define os diretórios
    base_dir = Path('/home/jociano/Projects/time_series_prediction')
    market1_dir = base_dir / 'data' / 'raw' / 'Market 1'
    market2_dir = base_dir / 'data' / 'processed' / 'Market 2'
    barcodes_file = base_dir / 'barcodes.txt'
    
    # Extrai códigos de barras dos arquivos
    print("Extraindo códigos de barras dos arquivos...")
    market1_barcodes = extract_barcodes_from_filenames(market1_dir)
    market2_barcodes = extract_barcodes_from_filenames(market2_dir)
    
    # Combina os códigos de ambas as pastas
    all_file_barcodes = market1_barcodes.union(market2_barcodes)
    
    # Lê os códigos do arquivo barcodes.txt
    print(f"Lendo códigos de {barcodes_file}...")
    barcodes_list = read_barcodes_file(barcodes_file)
    
    # Encontra as diferenças
    not_in_files = barcodes_list - all_file_barcodes
    not_in_list = all_file_barcodes - barcodes_list
    
    # Escreve os resultados
    output_dir = base_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'barcodes_nao_em_arquivos.txt', 'w') as f:
        for barcode in sorted(not_in_files):
            f.write(f"{barcode}\n")
    
    with open(output_dir / 'barcodes_nao_em_lista.txt', 'w') as f:
        for barcode in sorted(not_in_list):
            f.write(f"{barcode}\n")
    
    print(f"\nRelatório gerado em: {output_dir}")
    print(f"- Total de códigos únicos nos arquivos: {len(all_file_barcodes)}")
    print(f"- Total de códigos no barcodes.txt: {len(barcodes_list)}")
    print(f"- Códigos em barcodes.txt mas não nos arquivos: {len(not_in_files)}")
    print(f"- Códigos nos arquivos mas não em barcodes.txt: {len(not_in_list)}")

if __name__ == "__main__":
    main()
