

# 1) (opcional) edite dialogues.txt com suas linhas
#    formato 1: "<BOS> USER: oi <SEP> BOT: olá! <EOS>"
#    formato 2:  "oi -> olá!"  (o script converte para o formato 1)

# 2) treinar (usa/gera bpe.json automaticamente)
#python train_minigpt_bpe.py

# re-treinar o BPE do zero (se você trocou muito o dialogues)
#python train_minigpt_bpe.py --retrain_bpe 1

# checkpoints a cada 500 steps
#python train_minigpt_bpe.py --ckpt_every 500

# train_minigpt_bpe.py
# Python 3.10+ | PyTorch 2.x
import os, math, time, argparse
from collections import deque

import torch
import torch.nn as nn
from torch.nn import functional as F

# --- tokenizers (BPE byte-level) ---
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# =======================
# Args
# =======================
p = argparse.ArgumentParser()
p.add_argument("--dialogues", type=str, default="dialogues.txt", help="arquivo com pares de diálogo")
p.add_argument("--bpe_path", type=str, default="bpe.json", help="arquivo do tokenizer BPE")
p.add_argument("--retrain_bpe", type=int, default=0, help="1=re-treinar BPE a partir do arquivo de diálogos")
p.add_argument("--block_size", type=int, default=256)
p.add_argument("--batch_size", type=int, default=64)

p.add_argument("--steps", type=int, default=3000) #3000
p.add_argument("--eval_interval", type=int, default=250)
p.add_argument("--ckpt_every", type=int, default=1000)
#p.add_argument("--out", type=str, default="mini_gpt_bpe.pt")
p.add_argument("--out", type=str, default="agent-dialog-train-002.pt")

p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--warmup_steps", type=int, default=200)
p.add_argument("--grad_clip", type=float, default=1.0)
p.add_argument("--grad_accum", type=int, default=1)
p.add_argument("--label_smoothing", type=float, default=0.05)

p.add_argument("--d_model", type=int, default=256)
p.add_argument("--n_heads", type=int, default=4)          # par para evitar warning de nested tensor
p.add_argument("--n_layers", type=int, default=4)
p.add_argument("--d_ff", type=int, default=1024)
p.add_argument("--dropout", type=float, default=0.1)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# =======================
# Utils
# =======================
def fmt_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# =======================
# 1) Corpus
# =======================
def default_pairs():
    return [
        ("olá", "oi! como posso ajudar?"),
        ("quem é você?", "sou um modelo simples de transformer."),
        ("conte uma piada", "por que o livro foi ao médico? porque estava com dor de capa!"),
        ("qual seu nome?", "pode me chamar de mini-gpt."),
        ("qual a capital do brasil?", "brasília."),
        ("tchau", "tchau! até logo."),
        # variações
        ("oi", "olá! em que posso ser útil?"),
        ("obrigado", "de nada! precisando, é só chamar."),
        ("o que você sabe fazer?", "posso responder perguntas simples e bater papo."),
        ("brasil fica onde?", "na américa do sul."),
    ]

def load_or_make_dialogues(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        pairs = []
        for ln in lines:
            if ln.startswith("<BOS>"):
                pairs.append(ln)  # já tokenizado
            else:
                if "->" in ln:
                    u, b = ln.split("->", 1)
                    pairs.append(f"<BOS> USER: {u.strip()} <SEP> BOT: {b.strip()} <EOS>")
        if not pairs:  # arquivo vazio/sem formatação esperada
            pairs = [f"<BOS> USER: {u} <SEP> BOT: {b} <EOS>" for u, b in default_pairs()]
    else:
        pairs = [f"<BOS> USER: {u} <SEP> BOT: {b} <EOS>" for u, b in default_pairs()]
        with open(path, "w", encoding="utf-8") as f:
            for s in pairs:
                f.write(s + "\n")
    return "\n".join(pairs)

text = load_or_make_dialogues(args.dialogues)
lines = [ln for ln in text.split("\n") if ln.strip()]

# =======================
# 2) Tokenizer BPE
# =======================
def train_or_load_bpe(bpe_path, retrain, training_files):
    if retrain or not os.path.exists(bpe_path):
        print(f"[BPE] treinando tokenizer em: {training_files} -> {bpe_path}")
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=8000,
            min_frequency=2,
            special_tokens=["[PAD]", "[BOS]", "[SEP]", "[EOS]", "[UNK]"]
        )
        tok.train(training_files, trainer)
        tok.save(bpe_path)
    tok = Tokenizer.from_file(bpe_path)
    return tok

tok = train_or_load_bpe(args.bpe_path, bool(args.retrain_bpe), [args.dialogues])
PAD = tok.token_to_id("[PAD]")
BOS = tok.token_to_id("[BOS]")
SEP = tok.token_to_id("[SEP]")
EOS = tok.token_to_id("[EOS]")
UNK = tok.token_to_id("[UNK]")

# segurança: garante que os especiais existem
for name, tid in {"[PAD]": PAD, "[BOS]": BOS, "[SEP]": SEP, "[EOS]": EOS, "[UNK]": UNK}.items():
    if tid is None:
        raise RuntimeError(f"Token especial ausente no BPE: {name}. Re-treine com --retrain_bpe 1.")

def encode_ids(s: str) -> torch.Tensor:
    # como já colocamos <BOS>/<SEP>/<EOS> literalmente nas linhas, não adicionamos especiais aqui
    ids = tok.encode(s).ids
    return torch.tensor(ids, dtype=torch.long)

def decode_ids(t: torch.Tensor) -> str:
    return tok.decode(t.tolist())

vocab_size = tok.get_vocab_size()

# =======================
# 3) Dataset por linhas + collation com PAD
# =======================
samples = [encode_ids(ln) for ln in lines]
n = max(1, int(0.9 * len(samples)))
train_samples, val_samples = samples[:n], samples[n:]

block_size = args.block_size
batch_size = args.batch_size

def collate_batch_bpe(batch_ids):
    T = min(block_size, max(t.numel() for t in batch_ids))
    x = torch.full((len(batch_ids), T), PAD, dtype=torch.long)     # input com PAD
    y = torch.full((len(batch_ids), T), -100, dtype=torch.long)    # alvo com ignore_index
    for i, s in enumerate(batch_ids):
        s = s[:T]
        x[i, :s.numel()] = s
        if s.numel() >= 2:
            y[i, :s.numel()-1] = s[1:]  # shift-right
    return x.to(device), y.to(device), T

def get_batch(split):
    source = train_samples if split == "train" else val_samples
    assert len(source) > 0, "Sem amostras suficientes; aumente o corpus."
    idx = torch.randint(0, len(source), (batch_size,))
    batch = [source[i] for i in idx]
    return collate_batch_bpe(batch)

# =======================
# 4) Modelo (decoder-only via Encoder + máscara causal)
# =======================
class MiniGPTDecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout=0.1, weight_tying=True, pad_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if weight_tying:
            self.head.weight = self.tok_emb.weight

        # máscara causal [T,T] (True = mascarado)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        )

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_len, "Aumente max_len."
        # embeddings
        tok = self.tok_emb(idx)                                # [B,T,d]
        pos = self.pos_emb(torch.arange(T, device=idx.device)) # [T,d]
        x = tok + pos                                          # [B,T,d]

        # máscara causal
        attn_mask = self.causal_mask[:T, :T]                   # [T,T] bool

        # máscara de padding (opcional, ajuda em batches com muito PAD)
        # True = posição a mascarar
        pad_mask = (idx == self.pad_id)                        # [B,T] bool

        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=pad_mask)
        x = self.ln_f(x)
        return self.head(x)                                    # [B,T,V]

# =======================
# 5) Perda, Otimizador, Scheduler
# =======================
model = MiniGPTDecoderOnly(
    vocab_size=vocab_size,
    d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
    d_ff=args.d_ff, max_len=block_size, dropout=args.dropout,
    weight_tying=True, pad_id=PAD
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

def lr_for_step(step):
    # warmup linear + cosine decay
    if step <= 0:
        return 0.0
    if step < args.warmup_steps:
        return args.lr * (step / max(1, args.warmup_steps))
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * args.lr * (1 + math.cos(math.pi * progress))

# =======================
# 6) Treino (ETA + checkpoints)
# =======================
t0 = time.perf_counter()
last = t0
win = deque(maxlen=50)

base, _ = os.path.splitext(os.path.abspath(args.out))
next_ckpt = args.ckpt_every if args.ckpt_every and args.ckpt_every > 0 else None

steps = args.steps
eval_interval = args.eval_interval
accum = max(1, args.grad_accum)

model.train()
running = 0.0

for step in range(1, steps + 1):
    xb, yb, T = get_batch("train")
    logits = model(xb)
    loss = criterion(logits.view(-1, vocab_size), yb.view(-1)) / accum
    loss.backward()
    running += loss.item()

    if step % accum == 0:
        if args.grad_clip and args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for g in optimizer.param_groups:
            g["lr"] = lr_for_step(step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # tempo + ETA
    now = time.perf_counter()
    dt = now - last
    last = now
    win.append(dt)
    mean_dt = sum(win) / len(win) if win else dt
    steps_left = steps - step
    eta_secs = steps_left * mean_dt

    if step % eval_interval == 0 or step == 1:
        model.eval()
        with torch.no_grad():
            xbv, ybv, Tv = get_batch("val")
            lv = criterion(model(xbv).view(-1, vocab_size), ybv.view(-1)).item()
        tr = running * accum
        running = 0.0
        elapsed = time.perf_counter() - t0
        print(
            f"step {step:4d} | train loss {tr:.4f} | val loss {lv:.4f} "
            f"| lr {optimizer.param_groups[0]['lr']:.2e} | T {T} "
            f"| elapsed {fmt_time(elapsed)} | eta {fmt_time(eta_secs)}"
        )
        model.train()

    # checkpoint periódico
    if next_ckpt is not None and step >= next_ckpt:
        ckpt_path = f"{base}_step{step}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "tokenizer_path": os.path.abspath(args.bpe_path),
            "pad_id": PAD, "bos_id": BOS, "sep_id": SEP, "eos_id": EOS, "unk_id": UNK,
            "block_size": block_size,
            "model_args": dict(
                vocab_size=vocab_size,
                d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
                d_ff=args.d_ff, max_len=block_size,
                dropout=args.dropout, weight_tying=True, pad_id=PAD
            ),
        }, ckpt_path)
        print(f"[checkpoint] salvo em: {ckpt_path}")
        next_ckpt += args.ckpt_every

# =======================
# 7) Salvar final
# =======================
final_ckpt = {
    "state_dict": model.state_dict(),
    "tokenizer_path": os.path.abspath(args.bpe_path),
    "pad_id": PAD, "bos_id": BOS, "sep_id": SEP, "eos_id": EOS, "unk_id": UNK,
    "block_size": block_size,
    "model_args": dict(
        vocab_size=vocab_size,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, max_len=block_size,
        dropout=args.dropout, weight_tying=True, pad_id=PAD
    ),
}
torch.save(final_ckpt, args.out)
print(f"\nModelo salvo em: {os.path.abspath(args.out)}")
print(f"Tokenizer salvo/uso: {os.path.abspath(args.bpe_path)}")

  