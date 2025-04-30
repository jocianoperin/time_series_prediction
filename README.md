# 📈 Previsão de Séries Temporais (time_series_prediction)

Projeto desenvolvido para realizar previsões robustas utilizando modelos de Machine Learning (ML), com foco especial em Redes Neurais (LSTM) e XGBoost.

## 🔍 O que este projeto faz?

- **Preparação de dados:** Realiza o pré-processamento completo dos dados.
- **Feature Engineering:** Cria automaticamente diversas features temporais como:
  - Dia da semana, mês, se é fim de semana, feriados.
  - Lags e médias móveis configuráveis.
- **Modelos Implementados:**
  - XGBoost
  - Redes Neurais (LSTM)
- **Validação e Avaliação:** Avalia os modelos usando métricas como MAE, RMSE, MAPE e SMAPE.
- **Visualização:** Gera gráficos comparativos entre previsões e dados reais.

---

## 📂 Estrutura do projeto

```bash
time_series_prediction/
├── data/
│   └── <barcode>.csv
│   └── ...
├── src/
│   ├── feature_engineering.py
│   ├── train_nn.py
│   ├── train_xgboost.py
│   ├── utils.py
│   └── main.py
├── requirements.txt
└── README.md
```

- `data/<barcode>.csv`: Dados brutos utilizados para previsão.
- `src/feature_engineering.py`: Script responsável por criar e manipular novas features.
- `src/train_nn.py`: Implementação das Redes Neurais (LSTM).
- `src/train_xgboost.py`: Implementação do modelo XGBoost.
- `src/utils.py`: Funções auxiliares e métricas de avaliação.
- `src/main.py`: Script principal para execução do pipeline.

---

## 🚀 Como configurar e executar o projeto (Conda + Python 3.10)

### 1. Clone o repositório

```bash
git clone https://github.com/jocianoperin/time_series_prediction.git
cd time_series_prediction
```

### 2. Crie e ative o ambiente Conda

```bash
conda create -n tsenv python=3.10
conda activate tsenv
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o pipeline principal

```bash
python src/main.py
```

---

## 🛠️ Adicionando Features

Para adicionar novas features ao projeto:

1. Abra o arquivo `src/feature_engineering.py`.
2. Adicione sua lógica no método `create_features(df)`.
   
Exemplo:

```python
df["nova_feature"] = df["feature_existente"].shift(7)  # exemplo de nova feature de lag
```

---

## 🧠 Modificando as camadas da Rede Neural (LSTM)

Para alterar a arquitetura LSTM:

1. Abra o arquivo `src/train_nn.py`.
2. Localize a função `build_model(input_shape)`.
3. Ajuste a estrutura da rede conforme desejado.

Exemplo original:

```python
model = Sequential([
    LSTM(128, activation='relu', dropout: 0.1, return_sequences=True, bidirectional=True),
    LSTM(64, activation='relu', dropout: 0.1, return_sequences=False),
    Dense(32, activation='relu', dropout: 0.1),
])
```

Exemplo com camada adicional:

```python
model = Sequential([
    LSTM(128, activation='relu', dropout: 0.1, return_sequences=True, bidirectional=True),
    LSTM(64, activation='relu', dropout: 0.1, return_sequences=False),
    LSTM(32, activation='relu', dropout: 0.1),
    Dense(32, activation='relu', dropout: 0.1),
])
```
# Camada 1 removida (a de 512 unidades)
        #{"type": "LSTM", "units": 512, "activation": "relu", "dropout": 0.4, "return_sequences": True, "bidirectional": True},
        # Nova 1: LSTM com 256 unidades, bidirecional, permanece a mesma
        #{"type": "LSTM", "units": 256, "activation": "relu", "dropout": 0.1, "return_sequences": True, "bidirectional": True},
        # A camada GRU permanece
        #{"type": "GRU",  "units": 128, "activation": "relu", "dropout": 0.1, "return_sequences": True},
        # Camada de atenção permanece
        #{"type": "ATTN", "heads": 4,   "key_dim": 32, "dropout": 0.1},
        # Camada 5 removida (a de 64 unidades)
        #{"type": "LSTM", "units": 64,  "activation": "relu", "dropout": 0.2, "return_sequences": False},
        # Camada final densa permanece
        #{"type": "Dense","units": 32,  "activation": "relu", "dropout": 0.1},
---

## 📈 Avaliando os Resultados

Após execução, o script gera gráficos comparativos e exibe no terminal as métricas de desempenho:

```bash
MAE: 123.45
RMSE: 234.56
MAPE: 12.34%
SMAPE: 13.45%
```

---

## 📌 Considerações finais

- Certifique-se de que seu dataset esteja no formato correto em `data/vendas.csv`.
- Utilize o notebook disponível em `notebooks/analysis.ipynb` para realizar análises detalhadas e explorações adicionais.

---

## 📄 Licença

Este projeto está sob licença [MIT License](LICENSE).
