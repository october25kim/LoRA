"""E4a task layer. E1b tasks.py is imported UNMODIFIED (read-only; same specs, loaders, seeded splits, MAXLEN 128).
Only the tokenizer call differs: RoBERTa has no segment embeddings (type_vocab_size = 1), its tokenizer returns no
token_type_ids, and E4a never passes token_type_ids to the model (training and evaluation). An all-zero `tt` list is kept
in the cached data only so that the E1b batching code can be reused; it is never fed to the model."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E1B = HERE.parent / "e1b_confirmatory"
sys.path.insert(0, str(E1B))
from tasks import ALL_SPECS, TASKS, TASK_TYPE, load_raw, columns, split_indices, MAXLEN, N_HOLD, TRAIN_CAP, EVAL_CAP, SEED  # noqa: E402,F401


def tokenize(tok, a, b):
    """RoBERTa pair encoding <s> A </s></s> B </s> (tokenizer default), truncation longest_first at MAXLEN = 128 (as E1b).
    Returns (input_ids, all-zero placeholder segment ids that are never passed to the model)."""
    enc = tok(a, truncation=True, max_length=MAXLEN) if b is None else tok(a, b, truncation=True, max_length=MAXLEN)
    ids = enc["input_ids"]
    if "token_type_ids" in enc:
        assert all(max(x) == 0 for x in enc["token_type_ids"]), "non-zero segment ids from a RoBERTa tokenizer"
    return ids, [[0] * len(x) for x in ids]
