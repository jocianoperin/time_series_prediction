layers_cfg = [
    # 1️⃣ LSTM bidirecional capta padrões de longo prazo
    {"type": "LSTM", "units": 50, "bidirectional": False,
     "return_sequences": True, "activation": "relu"},

    # 2️⃣ LSTM unidirecional para compressão de sequência
    {"type": "LSTM", "units": 25, "bidirectional": False,
     "return_sequences": False, "activation": "relu"},

    # 3️⃣ Dense final para regressão
    {"type": "Dense", "units": 1}
]
"""
layers_cfg = [
    # 1️⃣ LSTM bidirecional capta padrões de longo prazo
    {"type": "LSTM", "units": 128, "bidirectional": True,
     "return_sequences": True, "dropout": 0.1},

    # 2️⃣ GRU unidirecional reduz parâmetros e mantém sequência
    {"type": "GRU",  "units": 64,
     "return_sequences": True, "dropout": 0.1},

    # 3️⃣ Bloco de atenção multi‑head (heads * key_dim ≈ units da GRU)
    {"type": "ATTN", "heads": 2, "key_dim": 16, "dropout": 0.1},

    # 4️⃣ GRU final para condensar a sequência
    {"type": "GRU",  "units": 64, "return_sequences": False, "dropout": 0.2},

    # 5️⃣ Dense intermediária
    {"type": "Dense","units": 16, "activation": "relu", "dropout": 0.1}
]
layers_cfg = [
    # híbrido um pouco maior
    {"type": "GRU",  "units": 96, "bidirectional": True, "return_sequences": True},
    {"type": "ATTN", "heads": 3,  "key_dim": 24, "dropout": 0.1},
    {"type": "LSTM", "units": 64, "return_sequences": False, "dropout": 0.1},
    {"type": "Dense","units": 32, "activation": "relu", "dropout": 0.1},
]"""

# ----------------------------------------------------------------------
# 📚  EXPLICAÇÃO DETALHADA DE `layers_cfg`
#
# Cada dicionário nesta lista descreve **um bloco de rede** que será criado
# dinamicamente pela função `build_lstm_model()`.  Os campos aceitos são:
#
#   • type               →  "LSTM", "GRU", "DENSE" ou "ATTN"
#   • units              →  Nº de neurônios (somente LSTM / GRU / Dense)
#   • bidirectional      →  Usa wrapper `tf.keras.layers.Bidirectional`
#   • return_sequences   →  Mantém 3‑D na saída (obrigatório p/ ATTN depois)
#   • activation         →  Função de ativação na célula (padrão: "tanh")
#   • recurrent_activation→ Função porta (padrão: "sigmoid")
#   • dropout            →  Dropout aplicado **após** a camada (exceto ATTN)
#   • heads, key_dim     →  Nº de cabeças e dimensão‑chave p/ Multi‑Head Attention
#
# ↓ Comentário linha‑a‑linha do bloco escolhido ↓
# ----------------------------------------------------------------------
# 1️⃣  LSTM bidirecional
#     • units=128  →  128 neurônios na direção forward **e** 128 na backward.
#       →  Saída tem 256 features por passo.  Ex.: input (B, 7, 9) → output (B, 7, 256)
#     • return_sequences=True  →  Mantém dimensão temporal para a GRU seguinte.
#     • dropout=0.1  →  10 % das ativações zeradas p/ reduzir overfitting.
#
# 2️⃣  GRU unidirecional
#     • units=64    →  64 neurônios; saída = (B, 7, 64)
#     • return_sequences=True  →  Necessário porque a atenção recebe tensor 3‑D.
#     • dropout=0.1  →  Dropout pós‑GRU (não afeta estado interno).
#
# 3️⃣  Multi‑Head Attention
#     • heads=2, key_dim=16  →  2 cabeças paralelas, cada uma projeta Q/K/V
#       em 16 dimensões.  Custo: O(heads * seq_len² * key_dim).
#       Ex.: seq_len=7  →  matrizes 7×7 pequeninas; custo irrisório.
#     • dropout=0.1  →  Dropout dentro da própria atenção.
#     • O bloco adiciona residual + LayerNorm:  `out = LayerNorm(x + Attn(x))`.
#
# 4️⃣  GRU final (compressão)
#     • units=64, return_sequences=False  →  Converte (B, 7, 64) em (B, 64)
#       pegando apenas o último passo temporal (saída vetorial).
#     • dropout=0.2  →  Aplicado no vetor (B, 64) antes da Dense.
#
# 5️⃣  Dense intermediária
#     • units=16, activation="relu" →  Extrai combinações não‑lineares
#       do vetor de 64 features → (B, 16).
#     • dropout=0.1  →  Última regularização antes da saída linear.
#
# →  Saída final da função `build_lstm_model()`:
#        (B, 1)   —  camada `Dense(1, activation="linear")` para regressão.
#
# Exemplos práticos
# -----------------
# ▸ Se TIME_STEPS = 7, n_features = 9 e batch_size = 32:
#     Input:  (32, 7, 9)
#     LSTM   → (32, 7, 256)
#     GRU    → (32, 7, 64)
#     ATTN   → (32, 7, 64)  (heads*key_dim == 32, broadcast p/ 64 via concat)
#     GRU    → (32, 64)
#     Dense  → (32, 16)
#     Out    → (32, 1)
#
# ▸ Para ativar bidirecional também na primeira GRU bastaria
#     {"type": "GRU", "units": 64, "bidirectional": True, ...}
#
# ▸ Para prever múltiplos dias de uma vez (horizonte=7), troque
#     outputs = Dense(7, activation="linear")(x)
#   e ajuste a função de perda para `tf.keras.losses.MAE` com shape (B, 7).
#
# Mantendo esses comentários no arquivo você documenta o racional da arquitetura
# e facilita ajustes futuros (tuning de unidades, heads, dropout, etc.).
# ----------------------------------------------------------------------