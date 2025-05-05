# 📈 Previsão de Séries Temporais (time_series_prediction)

Projeto desenvolvido por **Jociano Perin** como parte do seu mestrado profissional em Inteligência Computacional na **UTFPR (Universidade Tecnológica Federal do Paraná)**. 

Este repositório contém todo o pipeline ajustado para previsão de séries temporais com base no conjunto de dados reais utilizados no projeto, considerando o período de **01/01/2019 a 31/12/2023**, com foco em produtos identificados por código de barras.

O projeto foi desenvolvido para realizar previsões robustas utilizando modelos de Machine Learning (ML), com foco especial em Redes Neurais (LSTM) e XGBoost.

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
│   ├── main.py
│   └── utils/
│       ├── build_nn_model.py  # define a função build_lstm_model
│       ├── logging_config.py
│       └── metrics.py
├── requirements.txt
└── README.md
```

- `data/<barcode>.csv`: Dados brutos utilizados para previsão.
- `src/feature_engineering.py`: Script responsável por criar e manipular novas features.
- `src/train_nn.py`: Implementação das Redes Neurais (LSTM).
- `src/train_xgboost.py`: Implementação do modelo XGBoost.
- `src/utils.py`: Funções auxiliares e métricas de avaliação.
- `src/main.py`: Script principal para execução do pipeline.

## 🗂️ Estrutura dos Dados (.csv)

Este projeto espera que os arquivos de dados estejam no diretório:

```
data/raw/
```

Cada arquivo deve estar nomeado com o código de barras (barcode) do produto, no formato:

```
<barcode>.csv
```

### 📄 Estrutura esperada do arquivo `.csv`:

```csv
Date,Barcode,OnPromotion,Quantity,ReturnedQuantity,Discount,Increase,TotalValue,UnitValue,CostValue,Holiday
2019-01-02,7891021006125,0,17.0,0.0,0.946,0.0,169.15,9.95,8.66,0
2019-01-03,7891021006125,0,30.0,0.0,0.884,0.0,298.5,9.95,8.66,0
...
```

### 🕓 Período de dados padrão

O projeto foi ajustado para operar com dados entre:

```
01/01/2019 a 31/12/2023
```

Esse intervalo foi definido com base na disponibilidade e qualidade dos dados reais utilizados na pesquisa.

---

## 🚀 Como configurar e executar o projeto (Conda + Python 3.10)

### 1. Clone o repositório

```bash
git clone https://github.com/jocianoperin/time_series_prediction.git
cd time_series_prediction
```

### 2. Crie e ative o ambiente Conda

```bash
conda env create -f environment.yml
conda activate tsenv
```


### 3. Execute o pipeline principal

```bash
python src/main.py
```

---

## 🛠️ Pipeline Principal (`src/main.py`)

Este script foi reorganizado para:

- **Controle de concorrência** usando `multiprocessing` e semáforos:
    - **MAX_PARALLEL_PROCS:** processos simultâneos (e.g. 4)  
    - **MAX_XGB_CONCURRENT:** XGBoost concorrentes (GPU leve)  
    - **NN_GPU_LOCK:** GPU exclusiva para Redes Neurais  
- **Otimizações de GPU/BLAS:**

    ```python
    os.environ["OMP_NUM_THREADS"]       = "2"
    os.environ["OPENBLAS_NUM_THREADS"]  = "2"
    ```

- **Fluxo de processamento** por produto:
    1. Leitura e conversão de datas  
    2. Feature engineering (`create_features`)  
    3. Treino XGBoost em paralelo  
    4. Treino NN (LSTM/GRU/ATTN) em exclusividade de GPU  
    5. Consolidação de previsões e métricas  
    6. Movimento dos CSVs processados para `data/processed/`

## 🛠️ Adicionando Features

Para adicionar novas features ao projeto:

1. Abra o arquivo `src/feature_engineering.py`.
2. Adicione sua lógica no método `create_features(df)`.
   
Exemplo:

```python
df["nova_feature"] = df["feature_existente"].shift(7)  # exemplo de nova feature de lag
```

---

## 🧠 Modificando as camadas da Rede Neural (NN)

A definição da arquitetura está agora isolada em `src/utils/build_nn_model.py`. Para alterar:

1. Abra o arquivo `src/utils/build_nn_model.py`.  
2. Localize a variável `layers_config`, que é uma lista de dicionários com as suas camadas.  
3. Adicione, remova ou edite cada entry conforme desejado.  
4. Salve e execute o pipeline (`python src/main.py` ou `python src/train_nn.py`) para validar.

### Exemplo original de `layers_config`

```python
layers_cfg = [
    {"type": "LSTM",   "units": 128, "activation": "relu", "dropout": 0.1, "return_sequences": True,  "bidirectional": True},
    {"type": "LSTM",   "units": 64,  "activation": "relu", "dropout": 0.1, "return_sequences": False},
    {"type": "Dense",  "units": 32,  "activation": "relu", "dropout": 0.1},
]
```

Exemplo com camada adicional:

```python
layers_cfg = [
    {"type": "LSTM",   "units": 128, "activation": "relu", "dropout": 0.1, "return_sequences": True,  "bidirectional": True},
    {"type": "LSTM",   "units": 64,  "activation": "relu", "dropout": 0.1, "return_sequences": True},
    {"type": "LSTM",   "units": 32,  "activation": "relu", "dropout": 0.1},
    {"type": "Dense",  "units": 32,  "activation": "relu", "dropout": 0.1},
]
```

Camada 1 removida (a de 512 unidades)

```python
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
```

---

### ⚙️ Hiperparâmetros do XGBoost

O modelo XGBoost utiliza os seguintes hiperparâmetros, otimizados para séries temporais de vendas com padrão estável e ruído moderado:

```python
params = {
    "objective": "reg:squarederror",   # Regressão com erro quadrático
    "eval_metric": "mae",              # Métrica mais robusta contra outliers
    "learning_rate": 0.01,             # Aprendizado mais lento e estável
    "max_depth": 6,                    # Profundidade equilibrada para generalização
    "subsample": 0.8,                  # Amostragem parcial evita overfitting
    "colsample_bytree": 0.8,           # Seleção parcial de features por árvore
    "min_child_weight": 3,             # Folhas mínimas com 3 instâncias (evita splits fracos)
    "gamma": 0.1,                      # Exige ganho mínimo para split (regularização)
    "lambda": 1.0,                     # Regularização L2 nos pesos
    "tree_method": "hist",             # Treinamento otimizado via histogramas
    "device": "cuda",                  # Utiliza GPU com suporte CUDA
    "verbosity": 0,                    # Executa de forma silenciosa
    "seed": 42                         # Reprodutibilidade garantida
}
```

#### 🔁 Recomendações de ajuste por tipo de série:

| Tipo de série                      | Sugestões de ajuste                                              |
|-----------------------------------|------------------------------------------------------------------|
| **Alta volatilidade**             | Aumentar `min_child_weight` (ex: 5), reduzir `max_depth` (ex: 3) |
| **Séries com tendências sazonais**| Aumentar `max_depth` (ex: 8), manter `subsample` e `colsample_bytree` altos |
| **Séries muito curtas**           | Reduzir `min_child_weight`, usar `max_depth` menor (ex: 2–3)     |
| **Séries com muitos outliers**    | Trocar `eval_metric` para `"quantile"` (requer ajustes extras)  |


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
