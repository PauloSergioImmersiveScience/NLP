

# pip install torch --upgrade
import math, random, torch, torch.nn as nn
from torch.nn import functional as F

# =========================
# 1) Corpus de diálogo (toy)
# =========================
raw_dialogues = [
    ("olá",                   "oi! como posso ajudar?"),
    ("quem é você?",          "sou um modelo simples de transformer."),
    ("conte uma piada",       "por que o livro foi ao médico? porque estava com dor de capa!"),
    ("qual seu nome?",        "pode me chamar de mini-gpt."),
    ("qual a capital do brasil?", "brasília."),
    ("tchau",                 "tchau! até logo.")
]


# transforma pares em linhas concatenadas com tokens especiais
def build_corpus(pairs):
    s = []
    for u, b in pairs:
        s.append(f"<BOS> USER: {u} <SEP> BOT: {b} <EOS>")
    return "\n".join(s)


text = build_corpus(raw_dialogues)


# ==================================
# 2) Tokenização bem simples (char)
# ==================================
chars = sorted(list(set(text)))

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
vocab_size = len(chars)

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join([itos[int(i)] for i in t])

data = encode(text)

# divide em treino/val
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]

# ===========================
# 3) Loader simples por blocos
# ===========================
block_size = 128     # contexto máximo
batch_size = 32

def get_batch(split):
    source = train_data if split == "train" else val_data

    # Tamanho efetivo do contexto (não pode passar do que existe)
    T = min(block_size, len(source) - 1)
    assert T > 0, "Seu corpus ficou pequeno demais; aumente os diálogos ou reduza block_size."

    # Quantos pontos de início consigo sortear sem estourar o fim
    max_start = len(source) - T - 1

    if max_start > 0:
        ix = torch.randint(0, max_start, (batch_size,))
    else:
        # Se não dá pra sortear inícios diferentes, usa tudo começando em 0
        ix = torch.zeros(batch_size, dtype=torch.long)

    x = torch.stack([source[i:i+T]     for i in ix])
    y = torch.stack([source[i+1:i+T+1] for i in ix])

    return x.to(device), y.to(device)


# ===========================
# 4) Modelo Decoder-Only (GPT)
# ===========================
class MiniGPTDecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, d_ff=1024, max_len=128, dropout=0.1, weight_tying=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if weight_tying:
            self.head.weight = self.tok_emb.weight  # tying

        # máscara causal (True = posição mascarada)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        )

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_len, "Aumente max_len (block_size)."

        tok = self.tok_emb(idx)                               # [B, T, d]
        pos = self.pos_emb(torch.arange(T, device=idx.device))# [T, d]
        x = tok + pos                                         # [B, T, d]

        attn_mask = self.causal_mask[:T, :T]                  # [T, T] bool
        x = self.encoder(x, mask=attn_mask)                   # só self-attn causal
        x = self.ln_f(x)
        logits = self.head(x)                                 # [B, T, V]
        return logits

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=128, temperature=1.0, top_k=50):
        self.eval()
        device = next(self.parameters()).device
        idx = encode(prompt).unsqueeze(0).to(device)  # [1, T0]

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]               # janela
            logits = self(idx_cond)                         # [1, T, V]
            logits = logits[:, -1, :] / max(1e-6, temperature)

            if top_k is not None:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh, torch.full_like(logits, -1e10), logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1,1]
            idx = torch.cat([idx, next_id], dim=1)

            # parada simples ao completar "<EOS>"
            if itos[int(next_id)] == ">" and decode(idx[0][-5:]) == "<EOS>":
                break

        return decode(idx[0])


# ===========================
# 5) Treinamento
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

model = MiniGPTDecoderOnly(
    vocab_size=vocab_size,
    d_model=192, n_heads=4, n_layers=3, d_ff=768,
    max_len=block_size, dropout=0.1, weight_tying=True
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
criterion = nn.CrossEntropyLoss()

steps = 1500
eval_interval = 250
model.train()
for step in range(1, steps+1):
    xb, yb = get_batch("train")     # [B, T]
    logits = model(xb)              # [B, T, V]
    loss = criterion(logits.view(-1, vocab_size), yb.view(-1))

    optimizer.zero_grad(set_to_none=True)

    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()

    if step % eval_interval == 0 or step == 1:
        with torch.no_grad():
            xbv, ybv = get_batch("val")
            lv = criterion(model(xbv).view(-1, vocab_size), ybv.view(-1)).item()
        print(f"step {step:4d} | train loss {loss.item():.4f} | val loss {lv:.4f}")

# ===========================
# 6) Geração de resposta
# ===========================
model.eval()
prompt = "<BOS> USER: olá <SEP> BOT:"
out = model.generate(prompt, max_new_tokens=160, temperature=0.9, top_k=60)
print("\n=== GERAÇÃO ===")
print(out)
