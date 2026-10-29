# Collaboration Guide — ECE 661 Team 32

For Will, Rachel, and Luel. The goal is that three people can work in parallel
without stepping on each other.

## 1. One-time setup

```bash
git clone https://github.com/<OWNER>/ece661-tinystories.git
cd ece661-tinystories

python3 -m venv .venv            # Python 3.11+ required
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,notebook]"

python scripts/smoke_test.py     # downloads ~20 MB; must print "All 9 checks passed"
```

If the smoke test fails, **stop and fix the environment** before writing code.

> Note: the repo URL above is a placeholder until the GitHub remote is created.

## 2. Branching model

| Branch | Purpose |
|---|---|
| `main` | Stable integration branch. Always working. Never commit directly. |
| `phase1-eval-will` | Will — Phase 1 evaluation (Tasks 1–4) |
| `phase2-arch-rachel` | Rachel — architectures and training |
| `phase3-amharic-luel` | Luel — Amharic data, tokenizer, training |

Short-lived topic branches off your own phase branch are encouraged:
`phase1-eval-will/rouge`, `phase2-arch-rachel/mingru`, etc.

## 3. Everyday workflow

```bash
git checkout main && git pull origin main      # start from current main
git checkout -b phase2-arch-rachel/mingru      # your own branch

# ... work ...

pytest                                         # unit tests must pass
git add -A
git commit -m "feat(models): add minGRU baseline at 5M params"
git push -u origin phase2-arch-rachel/mingru
```

Then open a Pull Request on GitHub into `main`.

### Commit messages

`type(scope): summary` — types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`.
Examples:
- `feat(evaluation): add ROUGE-L scoring for Section 4.4.1`
- `fix(inference): left-pad batches so short prompts are not corrupted`

## 4. Pull request workflow

1. Push your branch and open a PR into `main`.
2. In the description, state: **what changed**, **what you ran**, and **the actual output** (test counts, not "tests pass").
3. Request a review from at least one teammate. The owner of any module you touched must review.
4. Merge only when tests pass and the review is approved. Prefer **Squash and merge**.
5. Delete the branch after merging.

### PR checklist

- [ ] `pytest` passes locally (paste the summary line)
- [ ] New behaviour has a test
- [ ] No checkpoints, datasets, or `.env` files added (`git status` is clean of large files)
- [ ] No API keys, tokens, or personal paths in the diff
- [ ] Config-driven, not hardcoded numbers
- [ ] Touched someone else's module? Said so in the PR and tagged them

## 5. Who owns what

| Path | Owner | Rule |
|---|---|---|
| `src/tinystories_project/inference/` | Will | Shared contract — coordinate before changing |
| `src/tinystories_project/io/` | Will | **Schema is contractual.** Additive changes only |
| `src/tinystories_project/evaluation/` | Will | Rachel/Luel: use it, don't edit it |
| `src/tinystories_project/models/` | Rachel | |
| `src/tinystories_project/training/` | Rachel | |
| `src/tinystories_project/data/` | Luel | |
| `configs/` | Shared | Add new files; avoid editing ones others cite |
| `docs/` | Shared | |

**Do not refactor a teammate's module to suit your own.** Open an issue or ask
in the weekly sync. If you need a change in a shared interface, raise it before
writing code against the version you wish existed.

## 6. Making your model work with the evaluation pipeline

Rachel and Luel: you do **not** need to use Hugging Face `.generate()`. Any
class satisfying the `TextGenerator` protocol works with every evaluation in
this repo. See the full contract in
`src/tinystories_project/inference/interface.py`:

```python
from tinystories_project.inference import GeneratedText, ModelMetadata

class MyMinGRUGenerator:
    @property
    def metadata(self) -> ModelMetadata: ...

    def generate(self, prompts, params, seed=None) -> list[GeneratedText]: ...
```

Then evaluation code treats your model exactly like a TinyStories checkpoint:

```python
from tinystories_project.inference import generate_stories
stories = generate_stories(MyMinGRUGenerator(), prompts, max_new_tokens=200, seed=42)
```

Key requirements: return `len(prompts) * num_return_sequences` items;
`completion` holds **only** new text (never echo the prompt); set
`prompt_index`, `sample_index`, and `finish_reason`.

## 7. What must never be committed

- Model checkpoints (`*.pt`, `*.bin`, `*.safetensors`) — `.gitignore` covers these
- Downloaded or generated datasets
- `.env`, API keys, tokens, SSH keys
- Large result sweeps (keep small curated samples in `results/examples/`)

If you ever commit a secret: **rotate the key immediately**, then remove it from
history. Rotating comes first — git history is recoverable by anyone who cloned.

## 8. Weekly rhythm

Per the proposal: weekly sync; each member presents their own phase at the
poster session. Bring what you actually ran, with numbers.
