import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from src.utils.logging_config import get_logger

# Carregar as variáveis do arquivo .env
load_dotenv()

logger = get_logger(__name__)

class DatabaseManager:
    """
    Gerencia as operações de banco de dados com flexibilidade para usar ou não SQLAlchemy.
    """
    def __init__(self, use_sqlalchemy=True, db_name=None):
        """
        Inicializa o DatabaseManager com suporte opcional para SQLAlchemy.

        Args:
            use_sqlalchemy (bool): Define se o SQLAlchemy será utilizado.
            db_name (str): Nome do banco de dados (opcional). Se None, usa o padrão do .env.
        """
        self.use_sqlalchemy = use_sqlalchemy
        self.db_name = db_name or os.getenv("DB_NAME")
        self.connection_string = (
            f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{self.db_name}"
        )
        if self.use_sqlalchemy:
            self.engine = create_engine(self.connection_string, echo=False, future=True)
        else:
            raise NotImplementedError("Apenas o SQLAlchemy é suportado nesta implementação.")

    def get_connection(self):
        """
        Retorna uma conexão ativa ao banco de dados.

        Returns:
            sqlalchemy.engine.Connection: Conexão ativa.
        """
        if self.use_sqlalchemy:
            return self.engine.connect()
        else:
            raise NotImplementedError("Apenas o SQLAlchemy é suportado nesta implementação.")

    def execute_query(self, query, params=None):
        """
        Executa uma consulta SQL no banco de dados.

        Args:
            query (str): A consulta SQL a ser executada.
            params (dict, optional): Parâmetros para a consulta.

        Returns:
            dict: Resultado no formato {'data': [...], 'columns': [...]}, ou {'rows_affected': int}.
        """
        if self.use_sqlalchemy:
            with self.get_connection() as connection:
                try:
                    logger.debug(f"Executando query: {query}")
                    result = connection.execute(text(query), params)
                    
                    # Verificar se a consulta retorna dados
                    if result.returns_rows:
                        data = result.fetchall()
                        columns = result.keys()
                        return {"data": [dict(zip(columns, row)) for row in data], "columns": columns}
                    else:
                        # Commit para comandos de modificação
                        connection.commit()
                        return {"rows_affected": result.rowcount}
                except SQLAlchemyError as e:
                    logger.error(f"Erro ao executar query com SQLAlchemy: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Erro inesperado ao executar query: {e}")
                    raise
        else:
            raise NotImplementedError("Apenas o SQLAlchemy é suportado nesta implementação.")
