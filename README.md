# Time Series Prediction Project

Este projeto demonstra um pipeline completo para previsão de séries temporais, comparando quatro modelos distintos:

- **ARIMA** (pmdarima)
- **Prophet** (Facebook Prophet)
- **XGBoost**
- **Rede Neural** (Keras/TensorFlow)

A estrutura foi organizada para processar diversos produtos (cada qual em seu CSV) e gerar métricas de desempenho, logs e previsões.

---

## Estrutura de Pastas

    time_series_prediction/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── logs/
    ├── models/
    │   ├── arima/
    │   ├── prophet/
    │   ├── xgboost/
    │   └── nn/
    ├── predictions/
    │   ├── arima/
    │   ├── prophet/
    │   ├── xgboost/
    │   ├── nn/
    │   └── metrics/
    ├── src/
    │   ├── compare_models.py
    │   ├── data_preparation.py
    │   ├── feature_engineering.py
    │   ├── logging_config.py
    │   ├── main.py
    │   ├── train_arima.py
    │   ├── train_nn.py
    │   ├── train_prophet.py
    │   └── train_xgboost.py
    └── requirements.txt

### Descrição das Pastas

- **data/raw**: coloque aqui os arquivos CSV originais (no formato `produto_<barcode>.csv`).
- **data/processed**: opcional, se quiser salvar dados limpos ou transformados.
- **logs**: armazena o arquivo de log (`pipeline.log`), gerado pelo `logging_config.py`.
- **models**: subpastas para salvar os modelos treinados (caso deseje persistir cada tipo).
- **predictions**: subpastas onde ficam os arquivos de previsões de cada modelo; há também uma pasta `metrics/` que recebe os arquivos de métricas de cada produto.
- **src**: contém os scripts que compõem o pipeline.

---

## Scripts Principais

### 1. logging_config.py
- Configura o logger para registro de mensagens tanto no console como no arquivo `logs/pipeline.log`.

### 2. data_preparation.py
- Função `load_data(raw_data_path="data/raw")`: lê todos os CSVs (que iniciam com `produto_` e terminam em `.csv`), faz validações básicas (colunas, datas), converte a coluna `Date` em datetime e retorna um dicionário no formato `{ barcode: DataFrame }`.

### 3. feature_engineering.py
- Função `create_features(df, n_lags=7)`: gera colunas de lag (`lag_1` a `lag_7`), média móvel (`rolling_7`), dia da semana (`day_of_week`), mês (`month`), etc. É útil principalmente para os modelos XGBoost e NN.

### 4. train_arima.py
- Função `train_arima(df, barcode)`: separa em treino e teste (80/20), ajusta modelo ARIMA (autoarima da lib pmdarima) e gera previsões. Retorna as previsões e as métricas MAE e MAPE.

### 5. train_prophet.py
- Função `train_prophet(df, barcode)`: renomeia colunas para `ds` e `y`, treina modelo Prophet, faz previsões e retorna resultados e métricas. Também separa treino e teste.

### 6. train_xgboost.py
- Função `train_xgboost(df, barcode)`: cria um XGBRegressor para prever `Quantity`. Usado com as features geradas em `feature_engineering.py`. Salva previsões e calcula métricas.

### 7. train_nn.py
- Função `train_neural_network(df, barcode)`: constrói uma rede MLP simples (Keras/TensorFlow), treina com EarlyStopping e obtém as previsões. Calcula MAE e MAPE.

### 8. compare_models.py
- Função `compare_and_save_results(barcode, results, out_path="predictions")`: recebe um dicionário com métricas de cada modelo, salva em um CSV de métricas e registra qual modelo obteve menor MAE.

### 9. main.py
- Orquestra todo o pipeline:
  1. Lê dados usando `load_data`.
  2. Para cada produto, gera features (se necessário) e chama os quatro treinadores (ARIMA, Prophet, XGBoost e NN).
  3. Salva previsões (em `predictions/<modelo>`).
  4. Compara as métricas e salva em `predictions/metrics/<barcode>_metrics.csv`.
  5. Faz log de cada passo.

---

## Como Executar

1. Crie ou ative seu ambiente virtual (conda ou venv).
2. Instale as dependências:
   
       pip install -r requirements.txt
3. Coloque seus arquivos CSV na pasta `data/raw/`. Cada arquivo deve se chamar `produto_<barcode>.csv`, contendo as colunas essenciais (`Date`, `Barcode`, `Quantity`, etc.).
4. Rode o pipeline:

       python src/main.py

- O script lerá todos os CSVs em `data/raw`, executará cada modelo e salvará:
  - Previsões em `predictions/<modelo>/<barcode>_preds.csv`.
  - Métricas em `predictions/metrics/<barcode>_metrics.csv`.
- Consulte `logs/pipeline.log` para verificar mensagens de sucesso ou falhas.

---

## Observações

- **Pasta `models/`**: por padrão, o projeto não salva os modelos treinados. Se quiser persistir (e futuramente carregar) os modelos, descomente os trechos de código que chamam `model.save_model(...)` ou `model.save(...)` nos scripts de treinamento.
- **Ajustes e Hiperparâmetros**: você pode ajustar parâmetros de cada modelo (número de árvores no XGBoost, arquitetura da NN, sazonalidade no Prophet, etc.) e refinar a criação de features conforme suas necessidades.
- **Dados Insuficientes**: no `main.py`, se houver menos de 50 registros para certo produto, ele ignora esse CSV e registra no log. Ajuste conforme sua realidade.
- **Escalabilidade**: caso tenha muitos produtos e dados massivos, considere processamento em paralelo, chunking ou uso de ferramentas como Dask ou Spark.

---
