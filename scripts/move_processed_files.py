#!/usr/bin/env python3
"""
Script para mover arquivos CSV processados para a pasta raw/processed.

Este script move arquivos da pasta raw para raw/processed quando:
1. O arquivo tem um correspondente nas pastas predictions/comparativo e plots/comparativo
2. O arquivo ainda não foi movido para raw/processed
"""

import os
import shutil
from pathlib import Path

def main():
    # Configuração dos caminhos
    base_path = Path('/home/jociano/Projects/time_series_prediction')
    raw_path = base_path / 'data' / 'raw'
    processed_path = raw_path / 'processed'
    
    # Garante que a pasta de destino existe
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Lista de barcodes que podem ser movidos (obtido da análise anterior)
    barcodes_to_move = [
        '7000000526879', '7000006522523', '7622300807399', '7622300989316',
        '7891000107836', '7891024026458', '7891051015210', '7891079000243',
        '7891164005245', '7891164006860', '7894900583700', '7894900681017',
        '7896004401843', '7896007545094', '7896011104157', '7896051020127',
        '7896080900148', '7896098900109', '7896183900465', '7896183901646',
        '7896275920838', '7896275970185', '7896295600024', '7896337300219',
        '7896380000388', '7896492403046', '7896504306457', '7896691103402',
        '7896934600255', '7897001010007', '7897001010014', '7897517206086',
        '7897938903052', '7898014855074', '7898027313066', '7898027315282',
        '7898039253848', '7898176600901', '7898289870024', '7898927536015'
    ]
    
    # Contadores
    moved_count = 0
    error_count = 0
    
    print(f"Iniciando movimentação de {len(barcodes_to_move)} arquivos...")
    
    # Processa cada barcode
    for barcode in barcodes_to_move:
        src_file = raw_path / f'produto_{barcode}.csv'
        dest_file = processed_path / f'produto_{barcode}.csv'
        
        try:
            # Verifica se o arquivo de origem existe
            if not src_file.exists():
                print(f"Aviso: Arquivo de origem não encontrado: {src_file}")
                error_count += 1
                continue
                
            # Verifica se o arquivo de destino já existe
            if dest_file.exists():
                print(f"Aviso: Arquivo de destino já existe: {dest_file}")
                error_count += 1
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
    print(f"- Total de erros: {error_count}")
    
    if error_count > 0:
        print("\nAtenção: Alguns arquivos não puderam ser movidos. Verifique as mensagens acima.")

if __name__ == "__main__":
    main()
