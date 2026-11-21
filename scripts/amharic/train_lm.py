"""
Train a tiny GPT-Neo language model on Amharic stories, from scratch or from an English checkpoint.

From scratch (same architecture as the released TinyStories checkpoints):
    python train_lm.py --train corpus/clean/train.txt --valid corpus/clean/valid.txt \
        --natural natural_stories/extracted --tokenizer tokenizers/am_bpe_4000.model \
        --size 8M --out runs/am_8M_scratch

Transfer from English (keeps the transformer blocks, replaces the embedding table):
    python train_lm.py ... --init roneneldan/TinyStories-8M --out runs/am_8M_transfer

Data-scaling curve:  add  --subset 2500   (uses the first 2,500 training stories)

Outputs in --out:  best/ (model weights, config, tokenizer), metrics.json, samples.txt
Reported for the best checkpoint: validation loss, perplexity, and bits-per-byte on the held-out
synthetic stories and (if --natural is given) on the human-written stories.
"""
from __future__ import annotations
import argparse, json, math, random, shutil, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPTNeoConfig, GPTNeoForCausalLM

from common import load_stories, SPTok, PAD, BOS, EOS

# Shapes of the released TinyStories GPT-Neo checkpoints (8 layers; width differs).
SIZES = {"1M": 64, "3M": 128, "8M": 256, "28M": 512}


def build_model(a, vocab_size: int) -> tuple[GPTNeoForCausalLM, str]:
    if a.init:
        model = GPTNeoForCausalLM.from_pretrained(a.init)
        model.resize_token_embeddings(vocab_size)
        torch.nn.init.normal_(model.get_input_embeddings().weight, mean=0.0, std=0.02)   # new language: start embeddings fresh
        model.tie_weights()
        for m in model.modules():                                                        # apply our dropout to the loaded model
            if isinstance(m, torch.nn.Dropout):
                m.p = a.dropout
        origin = f"transfer:{a.init}"
    else:
        h = SIZES[a.size]
        cfg = GPTNeoConfig(
            vocab_size=vocab_size, hidden_size=h, num_layers=a.layers, num_heads=a.heads,
            attention_types=[[["global", "local"], a.layers // 2]], window_size=256,
            max_position_embeddings=2048, resid_dropout=a.dropout, embed_dropout=a.dropout,
            attention_dropout=a.dropout,
        )
        model = GPTNeoForCausalLM(cfg)
        origin = f"scratch:{a.size}"
    model.config.vocab_size = vocab_size
    model.config.pad_token_id, model.config.bos_token_id, model.config.eos_token_id = PAD, BOS, EOS
    return model, origin


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    emb = model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()
    return {"total": total, "embedding": emb, "non_embedding": total - emb}


def chunk(ids: list[int], block: int) -> list[list[int]]:
    """Split one story into pieces of at most block+1 tokens (inputs + shifted targets)."""
    return [ids[i:i + block + 1] for i in range(0, max(1, len(ids) - 1), block)]


@torch.no_grad()
def evaluate(model, seqs: list[list[int]], n_bytes: int, device, amp, batch=64) -> dict:
    """Exact token-level loss over whole stories; bits-per-byte uses the UTF-8 size of the text."""
    model.eval()
    nll, ntok = 0.0, 0
    seqs = sorted(seqs, key=len)
    for i in range(0, len(seqs), batch):
        b = seqs[i:i + batch]
        L = max(len(s) for s in b)
        x = torch.full((len(b), L), PAD, dtype=torch.long)
        for j, s in enumerate(b):
            x[j, :len(s)] = torch.tensor(s)
        x = x.to(device)
        inp, tgt = x[:, :-1], x[:, 1:]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(input_ids=inp, attention_mask=(inp != PAD).long()).logits
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1),
                               ignore_index=PAD, reduction="sum")
        nll += loss.item(); ntok += int((tgt != PAD).sum())
    model.train()
    return {"loss": nll / max(1, ntok), "ppl": math.exp(min(20, nll / max(1, ntok))),
            "bits_per_byte": nll / math.log(2) / max(1, n_bytes), "tokens": ntok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--valid", required=True)
    ap.add_argument("--natural", default=None, help="folder of human-written stories for a second test set")
    ap.add_argument("--tokenizer", required=True, help="SentencePiece .model from train_tokenizer.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="8M", choices=list(SIZES), help="width preset when training from scratch")
    ap.add_argument("--layers", type=int, default=8); ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--init", default=None, help="pretrained GPT-Neo checkpoint to transfer from, e.g. roneneldan/TinyStories-8M")
    ap.add_argument("--freeze-blocks-steps", type=int, default=0,
                    help="transfer only: train just the new embeddings for this many steps before unfreezing the blocks")
    ap.add_argument("--subset", type=int, default=0, help="use only the first N training stories (data-scaling curve)")
    ap.add_argument("--block", type=int, default=512); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=30, help="upper bound; early stopping usually ends sooner")
    ap.add_argument("--eval-every", type=int, default=200); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
                          if a.device == "auto" else a.device)
    amp = device.type == "cuda"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    tok = SPTok(a.tokenizer)
    train_stories = load_stories(a.train)
    if a.subset:
        train_stories = train_stories[:a.subset]
    valid_stories = load_stories(a.valid)
    natural_stories = load_stories(a.natural) if a.natural else []

    stream = torch.tensor([t for s in train_stories for t in tok.encode(s)], dtype=torch.long)
    def prep(stories):
        return [c for s in stories for c in chunk(tok.encode(s), a.block)], sum(len(s.encode("utf-8")) for s in stories)
    valid_seqs, valid_bytes = prep(valid_stories)
    nat_seqs, nat_bytes = prep(natural_stories)

    model, origin = build_model(a, tok.vocab_size)
    model.to(device)
    params = count_params(model)
    block = min(a.block, model.config.max_position_embeddings, len(stream) - 2)
    steps_per_epoch = max(1, len(stream) // (a.batch * block))
    max_steps = int(a.epochs * steps_per_epoch)
    warmup = max(10, int(0.02 * max_steps))
    print(f"{origin} | device {device} | vocab {tok.vocab_size} | params total {params['total']:,} "
          f"(non-embedding {params['non_embedding']:,})")
    print(f"train: {len(train_stories)} stories, {len(stream):,} tokens | valid: {len(valid_stories)} stories | "
          f"natural: {len(natural_stories)} stories")
    print(f"block {block}, batch {a.batch} -> {steps_per_epoch} steps/epoch, up to {max_steps} steps "
          f"(tokens per parameter: {len(stream) / params['total']:.2f})")

    blocks = list(model.transformer.h.parameters())
    if a.init and a.freeze_blocks_steps:
        for p in blocks: p.requires_grad_(False)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    def lr_at(step):
        if step < warmup: return a.lr * (step + 1) / warmup
        t = (step - warmup) / max(1, max_steps - warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))

    g = torch.Generator().manual_seed(a.seed)
    history, best, bad, t0 = [], float("inf"), 0, time.time()
    model.train()
    run_loss, run_n = 0.0, 0
    for step in range(max_steps + 1):
        if a.init and a.freeze_blocks_steps and step == a.freeze_blocks_steps:
            for p in blocks: p.requires_grad_(True)
            print(f"step {step}: transformer blocks unfrozen")
        if step % a.eval_every == 0 or step == max_steps:
            ev = evaluate(model, valid_seqs, valid_bytes, device, amp)
            rec = {"step": step, "epoch": round(step / steps_per_epoch, 2),
                   "train_loss": round(run_loss / run_n, 4) if run_n else None,
                   "valid_loss": round(ev["loss"], 4), "valid_bpb": round(ev["bits_per_byte"], 4),
                   "minutes": round((time.time() - t0) / 60, 1)}
            history.append(rec); run_loss, run_n = 0.0, 0
            flag = ""
            if ev["loss"] < best - 1e-4:
                best, bad = ev["loss"], 0
                model.save_pretrained(out / "best"); flag = "  *saved*"
            else:
                bad += 1
            print(f"step {step:6d} ep {rec['epoch']:5.2f} | train {rec['train_loss']} | valid {rec['valid_loss']:.4f} "
                  f"| bpb {rec['valid_bpb']:.4f} | {rec['minutes']} min{flag}")
            if bad >= a.patience:
                print(f"early stop: no improvement in {a.patience} evaluations"); break
        if step == max_steps:
            break
        for gparam in opt.param_groups: gparam["lr"] = lr_at(step)
        ix = torch.randint(0, len(stream) - block - 1, (a.batch,), generator=g)
        x = torch.stack([stream[i:i + block] for i in ix]).to(device)
        y = torch.stack([stream[i + 1:i + block + 1] for i in ix]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss += loss.item(); run_n += 1

    # ---- final report on the best checkpoint
    model = GPTNeoForCausalLM.from_pretrained(out / "best").to(device)
    shutil.copy(a.tokenizer, out / "best" / "tokenizer.model")
    final = {"synthetic_valid": evaluate(model, valid_seqs, valid_bytes, device, amp)}
    if nat_seqs:
        final["natural"] = evaluate(model, nat_seqs, nat_bytes, device, amp)
    metrics = {"origin": origin, "args": vars(a), "params": params, "vocab_size": tok.vocab_size,
               "train_stories": len(train_stories), "train_tokens": len(stream),
               "best_valid_loss": best, "final": final, "history": history}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    print("\nbest checkpoint:")
    for k, v in final.items():
        print(f"  {k:16s} loss {v['loss']:.4f}  ppl {v['ppl']:.1f}  bits/byte {v['bits_per_byte']:.4f}")

    # ---- a few samples: 3 free stories and 3 completions of held-out story openings
    model.eval()
    lines = []
    def gen(prefix_ids):
        x = torch.tensor([prefix_ids], device=device)
        with torch.no_grad():
            y = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=350, do_sample=True,
                               temperature=0.8, top_p=0.95, pad_token_id=PAD, eos_token_id=EOS)
        return tok.decode(y[0].tolist())
    for i in range(3):
        lines.append(f"=== free sample {i + 1}\n{gen([BOS])}\n")
    for i, s in enumerate(valid_stories[:3]):
        ids = tok.encode(s, eos=False)
        cut = max(2, int(0.4 * len(ids)))
        lines.append(f"=== completion {i + 1}\n[PROMPT] {tok.decode(ids[:cut])}\n[MODEL ] {gen(ids[:cut])}\n")
    (out / "samples.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}/metrics.json and {out}/samples.txt")


if __name__ == "__main__":
    main()
