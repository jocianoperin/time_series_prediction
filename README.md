# Time Series Prediction Project

Este projeto demonstra um pipeline completo para previsão de séries temporais, comparando quatro modelos distintos:

- **ARIMA** (pmdarima)
- **Prophet** (Facebook Prophet)
- **XGBoost**
- **Rede Neural com LSTM** (Keras/TensorFlow)

A estrutura foi organizada para processar diversos produtos (cada qual em seu CSV) e gerar métricas de desempenho, logs e previsões.

---

## Estrutura de Pastas

```
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
```

---

## Melhorias recentes

- 🧠 **Rede Neural** agora usa **LSTM** com janelas temporais de 30 dias
- 📊 **XGBoost** otimizado com hiperparâmetros robustos e suporte à GPU (`tree_method='gpu_hist'`)
- 📈 **ARIMA** usa `stepwise=False` e `m=30` para detectar sazonalidade mensal
- 🧮 **Engenharia de features avançada**: inclui lags, médias móveis, promoções, lucro, variância, etc.
- 🔁 Todos os modelos agora fazem **predição mês a mês em 2024**, com **fine-tuning incremental** com base nos dados reais

---

## Como Executar

1. Crie e ative o ambiente (ex: conda ou venv com Python 3.10)
2. Instale os requisitos:

```bash
pip install -r requirements.txt
```

3. Coloque os CSVs em `data/raw/`, no padrão `produto_<barcode>.csv`
4. Execute o pipeline:

```bash
python src/main.py
```

---

## O que o pipeline faz

- Lê todos os produtos em `data/raw`
- Gera features enriquecidas automaticamente
- Treina os 4 modelos por janelas deslizantes (2019–2023)
- Prediz os valores de **2024 mês a mês**, com aprendizado incremental a cada mês
- Salva:
  - Previsões em `predictions/<modelo>/<barcode>_preds.csv`
  - Métricas em `predictions/metrics/<barcode>_metrics.csv`
  - Logs em `logs/pipeline.log`

---

## Observações

- A pasta `models/` pode ser usada para salvar modelos treinados, se desejado
- O ARIMA é executado com busca exaustiva (`stepwise=False`), o que pode ser lento mas aumenta a precisão
- Produtos com menos de 50 registros são ignorados (ajustável)
- Projeto pronto para escalar com paralelismo futuro (Dask, Ray, etc.)

---

## Contato

Mantenedor: **Jociano**

Projeto de pesquisa voltado à predição inteligente de promoções e comportamento de vendas com apoio de IA.
