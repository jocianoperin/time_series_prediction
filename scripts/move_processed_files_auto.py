#!/usr/bin/env python3
"""
Script para mover automaticamente arquivos CSV processados para a pasta data/processed.

Este script move arquivos da pasta data/raw para data/processed quando:
1. O arquivo tem um correspondente nas pastas data/predictions/comparativo e data/plots/comparativo
2. O arquivo ainda não foi movido para data/processed
"""

import os
import shutil
from pathlib import Path

def get_processed_barcodes(base_path):
    """Obtém a lista de barcodes que já foram processados."""
    # Caminhos das pastas de predições e plots
    predictions_path = base_path / 'data' / 'predictions' / 'comparativo'
    plots_path = base_path / 'data' / 'plots' / 'comparativo'
    
    # Verifica se os diretórios existem
    if not predictions_path.exists() or not plots_path.exists():
        print("Erro: Pastas de predições ou plots não encontradas.")
        return set()
    
    # Obtém os barcodes das pastas de predições e plots
    pred_barcodes = {d.name for d in predictions_path.iterdir() if d.is_dir()}
    plot_barcodes = {d.name for d in plots_path.iterdir() if d.is_dir()}
    
    # Retorna a interseção (barcodes que têm tanto predições quanto plots)
    return pred_barcodes.intersection(plot_barcodes)

def main():
    # Configuração dos caminhos
    base_path = Path('/home/jociano/Projects/time_series_prediction')
    raw_path = base_path / 'data' / 'raw'
    processed_path = base_path / 'data' / 'processed'
    
    # Garante que a pasta de destino existe
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Obtém a lista de barcodes processados
    processed_barcodes = get_processed_barcodes(base_path)
    
    if not processed_barcodes:
        print("Nenhum barcode processado encontrado.")
        return
    
    print(f"Encontrados {len(processed_barcodes)} barcodes processados.")
    
    # Contadores
    moved_count = 0
    error_count = 0
    skipped_count = 0
    
    # Processa cada barcode
    for barcode in sorted(processed_barcodes):
        src_file = raw_path / f'produto_{barcode}.csv'
        dest_file = processed_path / f'produto_{barcode}.csv'
        
        try:
            # Verifica se o arquivo de origem existe
            if not src_file.exists():
                print(f"Aviso: Arquivo de origem não encontrado: {src_file}")
                skipped_count += 1
                continue
                
            # Verifica se o arquivo de destino já existe
            if dest_file.exists():
                print(f"Aviso: Arquivo de destino já existe: {dest_file}")
                skipped_count += 1
                continue
                
            # Move o arquivo
            shutil.move(str(src_file), str(dest_file))
            print(f"Movido: {src_file.name} -> {dest_file}")
            moved_count += 1
            
        except Exception as e:
            print(f"Erro ao mover {src_file.name}: {str(e)}")
            error_count += 1
    
    # Resumo
    print("\nResumo:")
    print(f"- Total de arquivos movidos com sucesso: {moved_count}")
    print(f"- Total de arquivos ignorados: {skipped_count}")
    print(f"- Total de erros: {error_count}")
    
    if error_count > 0:
        print("\nAtenção: Alguns arquivos não puderam ser movidos. Verifique as mensagens acima.")

if __name__ == "__main__":
    main()
