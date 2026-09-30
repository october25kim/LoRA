"""E6 task layer (Qwen2.5-1.5B; byte-identical logic to E4b tasks_e4b.py, only this docstring differs). E1b tasks.py is imported UNMODIFIED (read-only: same specs, loaders, input construction, seeded
permutation for the held-out split). E4b-specific (PREREG_E4B.md sec. 1-2), reused unchanged by E6 (PREREG_E6.md sec. 1; Qwen2.5-1.5B uses the same Qwen2 tokenizer):
  * max length 256 (not 128); decoder tokenizer: no special tokens are added (Qwen2 tokenizer adds no BOS/EOS);
    pair inputs = tokenizer(a + "\\n", b) (plain concatenation of the two token sequences), longest-first truncation at 256;
  * training subset: pools <= 10,000 -> the full pool (identical to E1b); larger pools -> the first min(50,000, pool) indices of the
    E1b seed-0 permutation after the 1,000 held-out ones (nested inside E1b's 60k subset), sorted;
  * eval subset: all if n_eval <= 5,000, else sorted(default_rng(1).choice(n_eval, 5000, replace=False)).
The held-out 1,000 TRAIN examples are exactly E1b's (tasks.split_indices)."""
import sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E1B = HERE.parent / "e1b_confirmatory"
sys.path.insert(0, str(E1B))
from tasks import ALL_SPECS, TASKS, TASK_TYPE, load_raw, columns, split_indices, N_HOLD, SEED  # noqa: E402,F401

MAXLEN_E4B = 256
TRAIN_CAP_E4B = 50000
EVAL_CAP_E4B = 5000
SMALL_POOL = 10000


def indices_e4b(n_train_total, n_eval_total):
    """Returns (hold, train, eval) index arrays (sorted). hold == E1b hold (asserted)."""
    hold_e1b, _, _ = split_indices(n_train_total, n_eval_total)
    perm = np.random.default_rng(SEED).permutation(n_train_total)
    hold = np.sort(perm[:N_HOLD]); assert np.array_equal(hold, hold_e1b)
    pool = perm[N_HOLD:]
    train = np.sort(pool) if len(pool) <= TRAIN_CAP_E4B else np.sort(pool[:TRAIN_CAP_E4B])
    ev = np.sort(np.random.default_rng(SEED + 1).choice(n_eval_total, EVAL_CAP_E4B, replace=False)) if n_eval_total > EVAL_CAP_E4B else np.arange(n_eval_total)
    return hold, train, ev


def tokenize(tok, a, b):
    """Qwen2 tokenizer, no special tokens added; pair = a + '\\n' followed by b; truncation longest_first at 256.
    Returns (input_ids, placeholder all-zero list kept only for batching-code compatibility; never passed to the model)."""
    if b is None:
        enc = tok(a, truncation=True, max_length=MAXLEN_E4B)
    else:
        enc = tok([x + "\n" for x in a], b, truncation=True, max_length=MAXLEN_E4B)
    ids = enc["input_ids"]
    return ids, [[0] * len(x) for x in ids]


def budget_e4b(pool_n, n_used):
    """Pre-registered E4b budget: pool <= 10k -> 10 epochs (E1b); else 1 epoch, or 2 epochs if the pool is < 30k. Batch 32.
    warmup = min(500, round(0.1 * max_steps))."""
    import math
    spe = math.ceil(n_used / 32)
    ep = 10 if pool_n <= SMALL_POOL else (2 if pool_n < 30000 else 1)
    ms = spe * ep
    return {"n_pool": int(pool_n), "n_train_used": int(n_used), "epochs": ep, "steps_per_epoch": spe, "max_steps": ms,
            "warmup_steps": int(min(500, round(0.1 * ms)))}
