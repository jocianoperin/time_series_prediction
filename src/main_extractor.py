"""
Extrator de dados diários → CSVs por produto
-------------------------------------------
• Define uma lista fixa `BARCODES_TO_EXTRACT`.
  ‑ Se a lista estiver vazia, cai no fluxo antigo (representatividade).
• Gera arquivos `produto_<CodigoBarras>.csv` em data/processed/<db_name>/.
"""

from pathlib import Path
import os

from services.database.database_manager import DatabaseManager
from services.extractor.process_raw_data import process_individual_data
from utils.logging_config import get_logger

logger = get_logger(__name__)
ROOT_DIR      = Path(__file__).resolve().parents[1]   # project root
BASE_DATA_DIR = ROOT_DIR / "data"

# ============================================================
#  DEFINA AQUI OS CÓDIGOS‑DE‑BARRAS QUE DESEJA EXTRair
#  (inteiros ou strings). Se lista vazia → usa representatividade.
# ============================================================
BARCODES_TO_EXTRACT: list[int] | list[str] = [
    # Lista de códigos de barras que falharam no processamento anterior
    7896012105177,  # Falhou
    7896045112081,  # Falhou
    7896045112418,  # Falhou
    7896401183069,  # Falhou
    7896401183076,  # Falhou
    7896978201234,  # Falhou
    7898967125323,  # Falhou
    7898971704415   # Falhou
]

# ------------------------------------------------------------
#  Consulta SQL fallback (produtos mais representativos)
# ------------------------------------------------------------

def get_produtos_mais_representativos(db_manager, limit_: int = 1000):
    query = f"""
        SELECT CodigoProduto
        FROM indicadores_representatividade
             INNER JOIN produtos ON produtos.Codigo = indicadores_representatividade.CodigoProduto
        WHERE LENGTH(CodigoBarras) >= 7 AND CodigoBarras LIKE '7%'
        ORDER BY Representatividade DESC
        LIMIT {limit_};
    """
    result = db_manager.execute_query(query)
    data   = result.get("data", []) if result else []
    if data:
        logger.info("Representatividade: %s produtos retornados", len(data))
        return [row["CodigoProduto"] for row in data]
    logger.warning("Nenhum produto na tabela indicadores_representatividade")
    return []

# ------------------------------------------------------------
#  Função principal
# ------------------------------------------------------------

def main():
    logger.info("Iniciando pipeline de extração de CSVs")

    os.makedirs(BASE_DATA_DIR / "news_csv", exist_ok=True)

    dbm = DatabaseManager()

    if BARCODES_TO_EXTRACT:
        produtos = [int(b) for b in BARCODES_TO_EXTRACT]
        logger.info("Processando lista fixa de %s produtos", len(produtos))
    else:
        produtos = get_produtos_mais_representativos(dbm, 1000)
        if not produtos:
            logger.warning("Nenhum produto para extrair – encerrando")
            return

    db_name = os.getenv("DB_NAME") or "default_db"
    logger.info("Base de dados: %s", db_name)

    product_map = {code: idx + 1 for idx, code in enumerate(produtos)}
    process_individual_data(db_name, produtos, product_map, BASE_DATA_DIR / "news_csv")

    logger.info("Extração concluída com sucesso ☑️")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
