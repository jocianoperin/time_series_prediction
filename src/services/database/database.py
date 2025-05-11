# Este módulo serve como uma ponte para acessar o gerenciador de banco de dados,
# permitindo que outras partes do aplicativo importem facilmente essa instância única.

from src.services.database.database_manager import DatabaseManager

# Criando uma instância global do DatabaseManager que será usada em todo o aplicativo.
db_manager = DatabaseManager()
