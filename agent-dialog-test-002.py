

# agent-dialog-test-002.py
# Requisitos: pip install torch tokenizers
import argparse, sys, os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

# =========================
# Modelo (mesmo do treino)
# =========================
class MiniGPTDecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len,
                 dropout=0.1, weight_tying=True, pad_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if weight_tying:
            self.head.weight = self.tok_emb.weight

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        )

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_len, "Aumente max_len (block_size do treino)."
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        attn_mask = self.causal_mask[:T, :T]
        pad_mask = (idx == self.pad_id)

        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=pad_mask)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, encode_ids, decode_ids, prompt_text,
                 max_new_tokens=200, temperature=0.9, top_k=60, top_p=0.95,
                 repetition_penalty=1.15, freq_penalty=0.2, presence_penalty=0.1,
                 anti_echo_ids=None, anti_echo_steps=10, anti_echo_penalty=0.5,
                 debug=False):
        self.eval()
        device = next(self.parameters()).device

        idx = encode_ids(prompt_text).unsqueeze(0).to(device)
        if debug:
            vmax = int(idx.max().item()) if idx.numel() else -1
            vmin = int(idx.min().item()) if idx.numel() else -1
            print(f"[debug] prompt ids range: [{vmin}, {vmax}]  | vocab={self.tok_emb.num_embeddings}", file=sys.stderr)
        assert idx.numel() == 0 or (idx.min() >= 0 and idx.max() < self.tok_emb.num_embeddings), \
            "IDs fora do range da Embedding. Tokenizer diferente do treino?"

        seen_counts = {}
        anti_echo_set = set(int(t) for t in anti_echo_ids.tolist()) if anti_echo_ids is not None and len(anti_echo_ids) > 0 else set()

        for step_out in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :]
            logits = logits / max(1e-6, temperature)

            row = logits[0]
            for t, cnt in seen_counts.items():
                row[t] -= (repetition_penalty - 1.0) * cnt
                row[t] -= freq_penalty * cnt + (presence_penalty if cnt > 0 else 0.0)
            if step_out < anti_echo_steps and anti_echo_set:
                for t in anti_echo_set:
                    if 0 <= t < row.numel():
                        row[t] -= anti_echo_penalty
            logits[0] = row

            if top_k is not None:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh, torch.full_like(logits, -1e10), logits)

            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs_sorted = F.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(probs_sorted, dim=-1)
                cutoff = (cumsum > top_p).float().argmax(dim=-1)
                for b in range(logits.size(0)):
                    kkeep = max(1, int(cutoff[b]))
                    mask = torch.ones_like(logits[b]) * -1e10
                    mask[sorted_idx[b, :kkeep+1]] = 0
                    logits[b] = logits[b] + mask

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            t = int(next_id)
            seen_counts[t] = seen_counts.get(t, 0) + 1

            if "<EOS>" in decode_ids(idx[0][-16:]):
                break

        return decode_ids(idx[0])

def build_turn(user_msg: str) -> str:
    return f"<BOS> USER: {user_msg} <SEP> BOT:"

def extract_bot_reply(full_text: str) -> str:
    pos = full_text.rfind("BOT:")
    bot_part = full_text[pos+4:] if pos != -1 else full_text
    eos = bot_part.find("<EOS>")
    if eos != -1:
        bot_part = bot_part[:eos]
    return bot_part.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt (char OU BPE)")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=60)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--freq_penalty", type=float, default=0.2)
    ap.add_argument("--presence_penalty", type=float, default=0.1)
    ap.add_argument("--anti_echo", type=int, default=1)
    ap.add_argument("--anti_echo_steps", type=int, default=10)
    ap.add_argument("--anti_echo_penalty", type=float, default=0.5)
    ap.add_argument("--debug", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Carrega ckpt -----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    margs = ckpt["model_args"]
    pad_id = ckpt.get("pad_id", 0)

    # Embedding real do modelo (fonte da verdade)
    tok_emb_shape = ckpt["state_dict"]["tok_emb.weight"].shape  # (V, d_model)
    V_embed = int(tok_emb_shape[0])

    tokenizer_path = ckpt.get("tokenizer_path", None)
    chars = ckpt.get("chars")
    stoi = ckpt.get("stoi")
    itos = ckpt.get("itos")

    # ===== Detecta modo de forma robusta =====
    mode = None
    tok = None
    vocab_tok = None

    # Tentativa 1: se tem tokenizer e o tamanho bate com a Embedding => BPE
    if tokenizer_path:
        try:
            tok = Tokenizer.from_file(tokenizer_path)
            vocab_tok = tok.get_vocab_size()
            if vocab_tok == V_embed:
                mode = "BPE"
        except Exception:
            tok = None
            vocab_tok = None

    # Tentativa 2: se tem dicionário de caracteres e o tamanho bate (ou é próximo) => CHAR
    if mode is None and (chars or stoi or itos):
        try_len = len(chars) if chars else (len(stoi) if stoi else None)
        # Aceita mesmo se try_len != V_embed (alguns checkpoints podem ter leve diferença);
        # o encode CHAR já filtra chars não vistos.
        if try_len is None or try_len <= 0:
            pass
        else:
            mode = "CHAR"

    # Última verificação: se nada decidiu, use heurística
    if mode is None:
        if V_embed <= 512:
            mode = "CHAR"
        else:
            if tokenizer_path:
                # Temos tokenizer mas mismatch → erro claro
                raise RuntimeError(
                    f"Incompatibilidade: embedding V={V_embed} e tokenizer '{tokenizer_path}' tem vocab={vocab_tok}. "
                    f"Use o mesmo .pt e o mesmo bpe.json do treino."
                )
            else:
                raise RuntimeError(
                    "Não foi possível determinar o modo (BPE/CHAR). O checkpoint não tem tokenizer_path nem dicionário char."
                )

    # ----- Funções de encode/decode conforme modo -----
    if mode == "BPE":
        def encode_ids(s: str) -> torch.Tensor:
            return torch.tensor(tok.encode(s).ids, dtype=torch.long)
        def decode_ids(t: torch.Tensor) -> str:
            return tok.decode(t.tolist())
        print(f"[info] Modo: BPE | vocab_embed={V_embed} | tokenizer_vocab={vocab_tok} | tokenizer={tokenizer_path}")
    else:
        # CHAR
        if not (chars and stoi and itos):
            raise RuntimeError("Checkpoint CHAR não possui 'chars/stoi/itos'. Use o .pt correto do treino por caractere.")
        def encode_ids(s: str) -> torch.Tensor:
            ids = [stoi[c] for c in s if c in stoi]
            return torch.tensor(ids, dtype=torch.long)
        def decode_ids(t: torch.Tensor) -> str:
            return "".join(itos[int(i)] for i in t.tolist())
        print(f"[info] Modo: CHAR | vocab_embed={V_embed} | vocab_chars={len(chars)}")

    # ----- Instancia e carrega pesos -----
    # Sobrescreve vocab_size de margs com V_embed real, por segurança
    margs = dict(margs)
    margs["vocab_size"] = V_embed

    model = MiniGPTDecoderOnly(pad_id=pad_id, **margs).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    print("\n=== MODO CHAT ===")
    print("Comandos: /sair | /limpar | /cfg para ver/editar sampling\n")

    history = ""
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    repetition_penalty = args.repetition_penalty
    freq_penalty = args.freq_penalty
    presence_penalty = args.presence_penalty
    anti_echo = bool(args.anti_echo)
    anti_echo_steps = args.anti_echo_steps
    anti_echo_penalty = args.anti_echo_penalty
    debug = bool(args.debug)

    while True:
        try:
            user_msg = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaindo...")
            break

        if user_msg.lower() in {"/sair", "sair", "exit", "quit"}:
            print("Saindo...")
            break
        if user_msg.lower() in {"/limpar", "/reset"}:
            history = ""
            print("Histórico limpo.")
            continue
        if user_msg.lower().startswith("/cfg"):
            print(f"sampling => temp={temperature} top_k={top_k} top_p={top_p} rep={repetition_penalty} freq={freq_penalty} pres={presence_penalty} anti_echo={int(anti_echo)} steps={anti_echo_steps} pen={anti_echo_penalty}")
            try:
                cfg = input("novo (ex: temp=1.1,top_k=40,top_p=0.9,rep=1.2,freq=0.1,pres=0.1,anti_echo=1,steps=10,pen=0.5) ou ENTER p/ manter: ").strip()
            except (EOFError, KeyboardInterrupt):
                cfg = ""
            if cfg:
                for chunk in cfg.split(","):
                    if "=" not in chunk: continue
                    k,v = chunk.split("=", 1)
                    k = k.strip(); v = v.strip()
                    if k in ["temp","temperature"]: temperature = float(v)
                    elif k=="top_k": top_k = int(float(v))
                    elif k=="top_p": top_p = float(v)
                    elif k in ["rep","repetition_penalty"]: repetition_penalty = float(v)
                    elif k in ["freq","freq_penalty"]: freq_penalty = float(v)
                    elif k in ["pres","presence_penalty"]: presence_penalty = float(v)
                    elif k in ["anti_echo"]: anti_echo = bool(int(float(v)))
                    elif k in ["steps","anti_echo_steps"]: anti_echo_steps = int(float(v))
                    elif k in ["pen","anti_echo_penalty"]: anti_echo_penalty = float(v)
            continue
        if not user_msg:
            continue

        turn = build_turn(user_msg)
        prompt = (history + ("\n" if history else "") + turn)

        user_ids = encode_ids(user_msg)
        if debug:
            ids_prompt = encode_ids(prompt)
            pmin = int(ids_prompt.min().item()) if ids_prompt.numel() else -1
            pmax = int(ids_prompt.max().item()) if ids_prompt.numel() else -1
            print(f"[debug] vocab_embed={model.tok_emb.num_embeddings}  prompt_ids_min={pmin}  prompt_ids_max={pmax}", file=sys.stderr)

        out = model.generate(
            encode_ids, decode_ids, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            freq_penalty=freq_penalty, presence_penalty=presence_penalty,
            anti_echo_ids=(user_ids if anti_echo else None),
            anti_echo_steps=anti_echo_steps,
            anti_echo_penalty=anti_echo_penalty,
            debug=debug
        )
        bot_reply = extract_bot_reply(out)
        print(f"BOT: {bot_reply}")

        history_line = f"<BOS> USER: {user_msg} <SEP> BOT: {bot_reply} <EOS>"
        history = (history + ("\n" if history else "") + history_line)

if __name__ == "__main__":
    main()

