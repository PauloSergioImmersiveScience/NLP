
# dataset_download.py
# pip install -U datasets
import os, sys, argparse
from typing import List, Tuple, Iterable

def safe_import_datasets():
    try:
        from datasets import load_dataset
        return load_dataset
    except Exception as e:
        print("[erro] A lib 'datasets' não está instalada. Rode:  pip install -U datasets")
        print("       Detalhes:", e)
        sys.exit(1)

# ---------- util ----------
def to_line(pairs: List[Tuple[str, str]]) -> str:
    parts = ["<BOS>"]
    for u, b in pairs:
        u = (u or "").strip()
        b = (b or "").strip()
        if not u or not b:
            continue
        parts.append(f" USER: {u} <SEP> BOT: {b} ")
    parts.append("<EOS>")
    return " ".join(parts).strip()

def save_lines(lines: Iterable[str], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ln in lines:
            if ln and ln.strip():
                f.write(ln.strip() + "\n")
                n += 1
    print(f"[ok] Salvei {n} linhas em:\n     {os.path.abspath(out_path)}")

# ---------- loaders ----------
def from_dailydialog(load_dataset, split="train") -> List[str]:
    # usamos um ESPALHO já em parquet (sem script .py):
    # https://huggingface.co/datasets/roskoN/dailydialog
    print("[info] Carregando DailyDialog (roskoN/dailydialog)…")
    ds = load_dataset("roskoN/dailydialog", split=split)
    lines = []
    for dialog in ds["dialog"]:
        pairs = []
        for i in range(0, len(dialog) - 1, 2):
            u = str(dialog[i]).strip()
            b = str(dialog[i + 1]).strip()
            if u and b:
                pairs.append((u, b))
        if pairs:
            lines.append(to_line(pairs))
    return lines

def from_multiwoz22(load_dataset, split="train") -> List[str]:
    print("[info] Carregando MultiWOZ 2.2 (pfb30/multi_woz_v22)…")
    ds = load_dataset("pfb30/multi_woz_v22", split=split)
    lines = []
    for ex in ds:
        turns = ex.get("turns") or ex.get("dialogue") or []
        if not turns: 
            continue
        pairs, last_user = [], None
        for t in turns:
            role = (t.get("speaker") or t.get("role") or "").lower()
            text = (t.get("value") or t.get("text") or "").strip()
            if role in ("user", "customer"):
                last_user = text
            elif role in ("system", "assistant", "agent"):
                if last_user and text:
                    pairs.append((last_user, text))
                    last_user = None
        if pairs:
            lines.append(to_line(pairs))
    return lines

def from_ultrachat(load_dataset, split="train_sft") -> List[str]:
    # HuggingFaceH4/ultrachat_200k (parquet) → cada item tem 'messages': [{role, content}, ...]
    print("[info] Carregando UltraChat-200k (HuggingFaceH4/ultrachat_200k)…")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    lines = []
    for ex in ds:
        msgs = ex.get("messages") or []
        pairs, last_user = [], None
        for m in msgs:
            role = (m.get("role") or "").lower()
            text = (m.get("content") or "").strip()
            if role in ("user", "human", "prompter"):
                last_user = text
            elif role in ("assistant", "bot", "system"):  # 'system' é raro aqui; pode ignorar se quiser
                if last_user and text:
                    pairs.append((last_user, text))
                    last_user = None
        if pairs:
            lines.append(to_line(pairs))
    return lines

def from_oasst1_flat(load_dataset, split="train", lang: str | None = None,
                     min_pairs: int = 1) -> List[str]:
    """
    OpenAssistant/oasst1 (mensagens 'flat' por linha).
    Estratégia:
      - agrupar implicitamente por message_tree_id, mantendo a ORDEM do dataset (depth-first);
      - construir sequências alternando prompter→assistant; quando aparecer prompter após prompter,
        considera início de novo caminho (flush anterior).
    """
    print("[info] Carregando OpenAssistant/oasst1 (flat)…")
    ds = load_dataset("OpenAssistant/oasst1", split=split)

    def lang_match(row_lang: str) -> bool:
        if not lang:
            return True
        if not row_lang:
            return False
        L = row_lang.lower()
        return L == lang.lower() or L.startswith(lang.lower())

    lines = []
    cur_tree = None
    pairs: List[Tuple[str, str]] = []
    last_role = None
    last_user = None

    def flush():
        nonlocal pairs
        if len(pairs) >= min_pairs:
            lines.append(to_line(pairs))
        pairs = []

    for row in ds:
        r_lang = (row.get("lang") or "").strip()
        if lang and not lang_match(r_lang):
            continue

        tree = row.get("message_tree_id")
        role = (row.get("role") or "").lower()
        text = (row.get("text") or "").strip()
        if not text or role not in ("prompter", "assistant"):
            continue

        # novo tree → fecha conversa anterior
        if tree != cur_tree:
            if cur_tree is not None:
                flush()
            cur_tree = tree
            pairs, last_role, last_user = [], None, None

        if role == "prompter":
            # dois prompters seguidos => encerra caminho anterior e inicia novo
            if last_role == "prompter":
                flush()
                pairs, last_user = [], None
            last_user = text
            last_role = "prompter"

        elif role == "assistant":
            if last_user:
                pairs.append((last_user, text))
                last_user = None
            last_role = "assistant"

    # flush final
    if cur_tree is not None:
        flush()

    return lines

# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["oasst1", "dailydialog", "multiwoz22", "ultrachat"], default="oasst1")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="dialogues.txt")
    p.add_argument("--lang", default=None, help="ex.: pt, pt-BR, en (apenas para datasets com idioma)")
    p.add_argument("--max_lines", type=int, default=0, help="0 = tudo")
    p.add_argument("--min_pairs", type=int, default=1, help="descarta conversas com menos pares (para oasst1)")
    args = p.parse_args()

    load_dataset = safe_import_datasets()

    if args.dataset == "oasst1":
        lines = from_oasst1_flat(load_dataset, split=args.split, lang=args.lang, min_pairs=args.min_pairs)
    elif args.dataset == "dailydialog":
        lines = from_dailydialog(load_dataset, split=args.split)
    elif args.dataset == "multiwoz22":
        lines = from_multiwoz22(load_dataset, split=args.split)
    else:
        lines = from_ultrachat(load_dataset, split=("train_sft" if args.split == "train" else args.split))

    print(f"[info] Linhas brutas geradas: {len(lines)}")
    if args.max_lines and args.max_lines > 0:
        lines = lines[:args.max_lines]
        print(f"[info] Limitado a --max_lines={args.max_lines}")

    if not lines:
        print("[aviso] 0 linhas geradas. Verifique conexão e se o dataset/split/idioma estão corretos.")
    save_lines(lines, args.out)

if __name__ == "__main__":
    main()


