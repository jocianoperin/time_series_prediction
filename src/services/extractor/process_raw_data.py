from services.database.database_manager import DatabaseManager
from utils.logging_config import get_logger
from pathlib import Path
from datetime import datetime, timedelta
from workalendar.america import Brazil
import pandas as pd
import statistics  # Usado para calcular mediana

logger = get_logger(__name__)

def extract_aggregated_data(db_manager: DatabaseManager, produtos: list) -> pd.DataFrame:
    """
    Extrai dados agregados por dia para cada produto, incluindo CodigoBarras.
    Agora, para ValorUnitario e ValorCusto, em vez de calcular a média no banco,
    utiliza-se GROUP_CONCAT e, em seguida, fazemos a mediana em Python.
    """
    if not produtos:
        logger.warning("Nenhum produto foi passado para agregação.")
        return pd.DataFrame()

    produtos_joined = ",".join(str(p) for p in produtos)
    
    # Aqui, ao invés de ROUND(AVG(...)), usamos GROUP_CONCAT(...). Depois, calculamos a mediana em Python.
    query = f"""
    SELECT
        DATE(v.Data) AS Data,
        vp.CodigoProduto,
        pd.CodigoBarras,
        MAX(IF(vp.PrecoemPromocao = 1, 1, 0)) AS EmPromocao,
        TRUNCATE(SUM(IFNULL(vp.Quantidade, 0)), 0) AS Quantidade,
        TRUNCATE(SUM(IFNULL(vp.QuantDevolvida, 0)), 0) AS QuantDevolvida,
        GROUP_CONCAT(IFNULL(vp.ValorUnitario, 0) ORDER BY vp.ValorUnitario SEPARATOR ',') AS ValorUnitarioGC,
        ROUND(AVG(IFNULL(vp.Desconto, 0)), 3) AS Desconto,
        ROUND(AVG(IFNULL(vp.Acrescimo, 0)), 3) AS Acrescimo,
        GROUP_CONCAT(IFNULL(vp.ValorCusto, 0) ORDER BY vp.ValorCusto SEPARATOR ',') AS ValorCustoGC,
        0 AS ValorTotal  
    FROM vendasprodutos vp
    INNER JOIN vendas v ON vp.CodigoVenda = v.Codigo
    INNER JOIN produtos pd ON vp.CodigoProduto = pd.Codigo
    WHERE vp.CodigoBarras IS NOT NULL
      AND v.Data BETWEEN '2019-01-01' AND '2024-12-31'
      AND vp.CodigoBarras IN ({produtos_joined})
    GROUP BY DATE(v.Data), vp.CodigoProduto, vp.CodigoBarras
    ORDER BY DATE(v.Data), vp.CodigoProduto;
    """
    try:
        result = db_manager.execute_query(query)
        if not result['data']:
            logger.warning("Nenhum dado encontrado para os produtos especificados.")
            return pd.DataFrame()

        df = pd.DataFrame(result['data'], columns=result['columns'])
        
        # Log das colunas retornadas para debug
        logger.debug(f"Colunas retornadas: {df.columns.tolist()}")

        if "Data" not in df.columns:
            logger.error("Coluna 'Data' não encontrada no resultado da query!")
            return pd.DataFrame()

        # Convertendo a coluna Data para datetime
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        if df["Data"].isna().sum() > 0:
            logger.warning("Foram encontrados valores nulos na coluna 'Data' após conversão!")

        # Converte strings de ValorUnitarioGC e ValorCustoGC em mediana
        def str_to_median(value_str):
            """
            Converte uma string 'v1,v2,v3...' em uma mediana float.
            """
            if not value_str:
                return 0.0
            try:
                vals = [float(x) for x in value_str.split(',')]
                return round(statistics.median(vals), 3)
            except Exception as e:
                logger.error(f"Erro ao converter para mediana: {e}")
                return 0.0

        # Calcula a mediana de ValorUnitario
        if 'ValorUnitarioGC' in df.columns:
            df['ValorUnitario'] = df['ValorUnitarioGC'].apply(str_to_median)
            df.drop(columns=['ValorUnitarioGC'], inplace=True)

        # Calcula a mediana de ValorCusto
        if 'ValorCustoGC' in df.columns:
            df['ValorCusto'] = df['ValorCustoGC'].apply(str_to_median)
            df.drop(columns=['ValorCustoGC'], inplace=True)

        # Ajustando a coluna "ValorTotal"
        df['ValorTotal'] = (df['Quantidade'] * df['ValorUnitario']).round(3)

        return df
    except Exception as e:
        logger.error(f"Erro ao extrair dados agregados: {e}")
        return pd.DataFrame()


def fill_missing_days_and_zeroize(df_produto: pd.DataFrame,
                                  start_date: str = "2019-01-01",
                                  end_date: str = "2024-12-31") -> pd.DataFrame:
    """
    Preenche dias faltantes de forma robusta:
    - Começa no primeiro dia de venda >= 01/01/2019
    - Garante linhas até 31/12/2024
    - Evita quebra por falha de propagação de valores-chave como CodigoBarras
    """
    df_produto = df_produto[df_produto["Data"] >= pd.to_datetime(start_date)]
    df_produto = df_produto.sort_values(by="Data").reset_index(drop=True)

    if df_produto.empty:
        return pd.DataFrame()

    first_sale_date = df_produto["Data"].min()
    all_dates = pd.date_range(start=first_sale_date, end=end_date, freq="D")

    df_produto = df_produto.set_index("Data")

    # Se houver datas duplicadas, agregue com segurança
    if not df_produto.index.is_unique:
        df_produto = df_produto.groupby(level=0).agg({
            "CodigoProduto": "first",
            "CodigoBarras": "first",
            "EmPromocao": "max",
            "Quantidade": "sum",
            "QuantDevolvida": "sum",
            "ValorUnitario": "mean",
            "Desconto": "mean",
            "Acrescimo": "mean",
            "ValorCusto": "mean",
            "ValorTotal": "sum"
        })

    # Reindexar do primeiro dia de venda até o fim do período
    df_produto = df_produto.reindex(all_dates)

    # Ffill seguro para campos fixos (somente após reindexar!)
    campos_identificadores = ["CodigoProduto", "CodigoBarras", "ValorUnitario", "ValorCusto"]
    df_produto[campos_identificadores] = df_produto[campos_identificadores].ffill()

    # Campos de movimentação → zero quando sem vendas
    campos_zeraveis = ["Quantidade", "QuantDevolvida", "ValorTotal", "Desconto", "Acrescimo", "EmPromocao"]
    df_produto[campos_zeraveis] = df_produto[campos_zeraveis].fillna(0)

    # Ajustes finais
    df_produto.index.name = "Data"
    df_produto = df_produto.reset_index()

    # Limpeza final
    for col in campos_zeraveis:
        df_produto[col] = df_produto[col].astype(float).fillna(0)

    return df_produto

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas de features como feriados, final de semana, dia da semana,
    semana do mês e véspera de feriado.
    """
    cal = Brazil()
    # Criamos um set de objetos date (e não datetime) para bater com d.date()
    df_holidays = {holiday_date for year in df['Data'].dt.year.unique()
                                for holiday_date, _ in cal.holidays(year)}

    # Final de Semana
    df['FinalDeSemana'] = df['Data'].dt.weekday.isin([5, 6]).astype(int)

    # Dia da Semana (0 = Segunda, ..., 6 = Domingo)
    df['DiaDaSemana'] = df['Data'].dt.weekday

    # Semana do Mês (1 = primeira semana, 2 = segunda, etc.)
    df['SemanaDoMes'] = df['Data'].apply(lambda d: (d.day - 1) // 7 + 1)

    # Feriado
    df['Feriado'] = df['Data'].apply(lambda d: 1 if d.date() in df_holidays else 0)

    # Véspera de Feriado
    df['VesperaDeFeriado'] = df['Data'].apply(
        lambda d: 1 if (d + timedelta(days=1)).date() in df_holidays else 0
    )

    return df

def anonymize_product_codes(df: pd.DataFrame, col_name: str = "CodigoProduto") -> pd.DataFrame:
    """
    Substitui o código real do produto por um código sequencial anônimo.
    Ex.: 1111 -> 1, 2222 -> 2, ...
    """
    unique_codes = df[col_name].unique()
    code_map = {old: i+1 for i, old in enumerate(sorted(unique_codes))}
    df["CodigoProduto"] = df[col_name].map(code_map)
    return df

def save_data_by_product(df: pd.DataFrame, output_dir: Path, db_name: str):
    """
    Salva os dados *já* agregados por dia em arquivos CSV, um por produto.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Identificar colunas que devem ser inteiras
        int_columns = ["CodigoProduto",  "EmPromocao"]

        # Substituir NaN por 0 antes de converter para inteiro
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        for produto in df['CodigoProduto'].unique():
            produto_df = df[df['CodigoProduto'] == produto].copy()

            # Ajustar nome do arquivo
            file_path = output_dir / f"{db_name}_produto_{produto}.csv"
            produto_df.to_csv(file_path, index=False, sep=',')
            logger.info(f"[{db_name}] Dados do produto {produto} salvos em {file_path}.")
    except Exception as e:
        logger.error(f"Erro ao salvar dados por produto: {e}")

def process_individual_data(db_name: str, produtos: list, product_map: dict, output_dir: Path):
    """
    Processa dados de uma base específica, produto a produto, e salva resultados diários por produto,
    *substituindo o CodigoProduto real* pelo código sequencial definido em 'product_map'.
    
    Porém, atendendo às solicitações:
      - O CSV final não terá a coluna CodigoProduto.
      - O nome do arquivo de saída será 'produto_<CodigoBarras>.csv'.
      - Colunas de datas especiais (DiaDaSemana, FinalDeSemana, SemanaDoMes, VesperaDeFeriado) são removidas antes do salvamento.
      - Mantemos a coluna CodigoBarras.
    """
    logger.info(f"Processando dados da base {db_name}.")
    db_manager = DatabaseManager(db_name=db_name)

    all_dates = pd.date_range(start="2019-01-01", end="2024-12-31", freq="D")

    for produto in produtos:
        logger.info(f"Processando dados para o produto {produto}.")
        df = extract_aggregated_data(db_manager, [produto])

        if not df.empty:
            df['Data'] = pd.to_datetime(df['Data'])
            df_prod = fill_missing_days_and_zeroize(df,
                                                    start_date="2019-01-01",
                                                    end_date="2024-12-31")
        else:
            # DataFrame “vazio” para garantir todas as datas
            df_prod = pd.DataFrame({'Data': all_dates})
            df_prod["CodigoBarras"] = None
            df_prod["ValorUnitario"] = 0
            df_prod["ValorCusto"] = 0
            for col in ["Quantidade", "QuantDevolvida", "ValorTotal", "Desconto", "Acrescimo", "EmPromocao"]:
                df_prod[col] = 0
            df_prod["CodigoProduto"] = produto

        df_prod = add_features(df_prod)

        # FORÇAR EmPromocao COMO INTEIRO (0 ou 1)
        df_prod['EmPromocao'] = df_prod['EmPromocao'].fillna(0).astype(int)

        # REMOVER COLUNAS QUE NÃO DEVEM IR PARA O CSV FINAL
        cols_to_drop = ["DiaDaSemana", "FinalDeSemana", "SemanaDoMes", "VesperaDeFeriado"]
        df_prod.drop(columns=[col for col in cols_to_drop if col in df_prod.columns], inplace=True)

        # Anonimizar o código do produto (caso ainda seja necessário internamente)
        # mas depois vamos removê-lo do CSV
        codigo_sequencial = product_map[produto]
        df_prod["CodigoProduto"] = codigo_sequencial

        output_dir_db = output_dir / db_name
        output_dir_db.mkdir(parents=True, exist_ok=True)

        # Como o SQL já trouxe só esse barcode, não é preciso filtrar de novo:
        sub_df = df_prod.copy()

        # Remover a coluna CodigoProduto do CSV final
        if "CodigoProduto" in sub_df.columns:
            sub_df.drop(columns=["CodigoProduto"], inplace=True)

        # Renomear colunas para inglês
        col_rename_map = {
            "Data": "Date",
            "CodigoBarras": "Barcode",
            "EmPromocao": "OnPromotion",
            "Quantidade": "Quantity",
            "QuantDevolvida": "ReturnedQuantity",
            "Desconto": "Discount",
            "Acrescimo": "Increase",
            "ValorTotal": "TotalValue",
            "ValorUnitario": "UnitValue",
            "ValorCusto": "CostValue",
            "Feriado": "Holiday"
        }
        sub_df.rename(columns=col_rename_map, inplace=True)

        # Monta nome do arquivo: produto_<CodigoBarras>.csv
        cod_barras_str = str(produto)  # converte pra string direto
        file_path = output_dir_db / f"produto_{cod_barras_str}.csv"

        logger.debug("Únicos em CodigoBarras em df_prod:", df_prod["CodigoBarras"].unique())
        logger.debug("Tipo de produto:", type(produto))

        # Salvar arquivo CSV
        sub_df.to_csv(file_path, index=False, sep=",")
        logger.info(
            f"[{db_name}] Dados do produto {produto} (barcode={cod_barras_str}) salvos em {file_path}."
        )

def save_consolidated_data(df: pd.DataFrame, output_dir: Path, db_name: str):
    """
    Salva um único CSV consolidado contendo todos os produtos e todas as datas.
    
    - Se um produto não teve venda em determinado dia, preenche com quantidade = 0.
    - Ordena por Data e CodigoProduto.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Nome do arquivo consolidado
        consolidated_file_path = output_dir / f"{db_name}_consolidado.csv"

        # Garantir ordenação por Data e Código do Produto
        df = df.sort_values(by=["Data", "CodigoProduto"]).reset_index(drop=True)

        # Salvar arquivo único
        df.to_csv(consolidated_file_path, index=False, sep=",")
        logger.info(f"[{db_name}] Dados consolidados salvos em {consolidated_file_path}.")
    except Exception as e:
        logger.error(f"Erro ao salvar o CSV consolidado: {e}")


def remove_codigo_barras(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função adicionada para remoção da coluna 'CodigoBarras' ao final do processo.
    Não é chamada internamente para não quebrar fluxos que precisem da coluna antes.
    """
    if 'CodigoBarras' in df.columns:
        df = df.drop(columns=['CodigoBarras'])
    return df
