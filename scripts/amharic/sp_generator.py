"""
Wraps a model trained by train_lm.py so the team's evaluation scripts (Task 2/3/4) can use it.
Satisfies tinystories_project.inference.interface.TextGenerator.

    gen = SPGenerator("runs/am_8M_scratch/best")
    outs = gen.generate(["አንድ ቀን"], GenerationParams(max_new_tokens=200))
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import torch
from transformers import GPTNeoForCausalLM

from common import SPTok, PAD, BOS, EOS
from tinystories_project.inference.interface import GeneratedText, GenerationParams, ModelMetadata


class SPGenerator:
    def __init__(self, run_dir: str | Path, device: str = "auto", model_id: str | None = None):
        self.run_dir = Path(run_dir)
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.tok = SPTok(self.run_dir / "tokenizer.model")
        self.model = GPTNeoForCausalLM.from_pretrained(self.run_dir).to(self.device).eval()
        self.model_id = model_id or self.run_dir.parent.name

    @property
    def metadata(self) -> ModelMetadata:
        m, cfg = self.model, self.model.config
        total = sum(p.numel() for p in m.parameters())
        emb = m.transformer.wte.weight.numel() + m.transformer.wpe.weight.numel()
        return ModelMetadata(
            model_id=self.model_id, architecture="gpt-neo", tokenizer_id=f"sentencepiece:{Path(self.tok.path).name}",
            source="local", n_parameters_total=total, n_parameters_non_embedding=total - emb,
            n_parameters_embedding=emb, vocab_size=cfg.vocab_size,
            max_position_embeddings=cfg.max_position_embeddings, n_layers=cfg.num_layers,
            hidden_size=cfg.hidden_size, dtype=str(next(m.parameters()).dtype), device=str(self.device),
            extra={"language": "am", "run_dir": str(self.run_dir)},
        )

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], params: GenerationParams, seed: int | None = None) -> list[GeneratedText]:
        if seed is not None:
            torch.manual_seed(seed)
        out: list[GeneratedText] = []
        for pi, prompt in enumerate(prompts):
            ids = self.tok.encode(prompt, eos=False)
            x = torch.tensor([ids], device=self.device)
            kw = dict(max_new_tokens=params.max_new_tokens, do_sample=params.do_sample,
                      repetition_penalty=params.repetition_penalty,
                      num_return_sequences=params.num_return_sequences, pad_token_id=PAD, eos_token_id=EOS)
            if params.do_sample:
                kw.update(temperature=params.temperature, top_p=params.top_p)
                if params.top_k is not None:
                    kw["top_k"] = params.top_k
            ys = self.model.generate(x, attention_mask=torch.ones_like(x), **kw)
            shown_prompt = self.tok.decode(ids)
            for si, y in enumerate(ys):
                new = y[len(ids):].tolist()
                hit_eos = EOS in new
                if hit_eos:
                    new = new[: new.index(EOS)]
                new = [t for t in new if t != PAD]
                full = self.tok.decode(ids + new)
                completion = full[len(shown_prompt):] if full.startswith(shown_prompt) else self.tok.decode(new)
                out.append(GeneratedText(
                    prompt=prompt, completion=completion, prompt_index=pi, sample_index=si, seed=seed,
                    n_prompt_tokens=len(ids), n_new_tokens=len(new), finish_reason="eos" if hit_eos else "length",
                ))
        return out
