"""E1b task specifications (shared by all E1b scripts). Self-contained; no repo imports."""
import numpy as np

MAXLEN = 128
N_HOLD = 1000          # held-out TRAIN examples per task (seed 0), never trained on; lambda selection only
TRAIN_CAP = 60000      # seeded subsample cap on the remaining training pool
EVAL_CAP = 10000       # seeded subsample cap for eval sets larger than this
SEED = 0

# name: (hf_path, config, train_split, eval_split, text_a, text_b, label_col, num_labels, metric)
# text_a may be a callable building the first segment (WiC).
TASKS = {
    "cola":  ("nyu-mll/glue", "cola", "train", "validation", "sentence", None, "label", 2, "accuracy"),
    "sst2":  ("nyu-mll/glue", "sst2", "train", "validation", "sentence", None, "label", 2, "accuracy"),
    "mrpc":  ("nyu-mll/glue", "mrpc", "train", "validation", "sentence1", "sentence2", "label", 2, "accuracy"),
    "qqp":   ("nyu-mll/glue", "qqp", "train", "validation", "question1", "question2", "label", 2, "accuracy"),
    "stsb":  ("nyu-mll/glue", "stsb", "train", "validation", "sentence1", "sentence2", "label", 1, "spearman"),
    "mnli":  ("nyu-mll/glue", "mnli", "train", "validation_matched", "premise", "hypothesis", "label", 3, "accuracy"),
    "qnli":  ("nyu-mll/glue", "qnli", "train", "validation", "question", "sentence", "label", 2, "accuracy"),
    "rte":   ("nyu-mll/glue", "rte", "train", "validation", "sentence1", "sentence2", "label", 2, "accuracy"),
    "boolq": ("aps/super_glue", "boolq", "train", "validation", "question", "passage", "label", 2, "accuracy"),
    "wic":   ("aps/super_glue", "wic", "train", "validation", "WIC_A", "sentence2", "label", 2, "accuracy"),
    "snli":  ("stanfordnlp/snli", None, "train", "validation", "premise", "hypothesis", "label", 3, "accuracy"),
    "scitail": ("allenai/scitail", "tsv_format", "train", "validation", "premise", "hypothesis", "SCITAIL_LABEL", 2, "accuracy"),
    "ag_news": ("fancyzhx/ag_news", None, "train", "test", "text", None, "label", 4, "accuracy"),
    "imdb":  ("stanfordnlp/imdb", None, "train", "test", "text", None, "label", 2, "accuracy"),
    "trec":  ("CogComp/trec", None, "train", "test", "text", None, "coarse_label", 6, "accuracy"),
    "yelp_polarity": ("fancyzhx/yelp_polarity", None, "train", "test", "text", None, "label", 2, "accuracy"),
}
PRIMARY_ORDER = list(TASKS.keys())
# pre-declared substitutes, used in this order only if a primary dataset fails to load (hans skipped by instruction)
SUBSTITUTES = {
    "rotten_tomatoes": ("cornell-movie-review-data/rotten_tomatoes", None, "train", "test", "text", None, "label", 2, "accuracy"),
    "emotion": ("dair-ai/emotion", "split", "train", "test", "text", None, "label", 6, "accuracy"),
    "paws": ("google-research-datasets/paws", "labeled_final", "train", "test", "sentence1", "sentence2", "label", 2, "accuracy"),
}
ALL_SPECS = {**TASKS, **SUBSTITUTES}
# CogComp/trec main branch still ships a loading script (unsupported in datasets>=4); the Hub's own auto-converted
# parquet branch of the SAME dataset repo is used instead (identical splits: 5452 train / 500 test, coarse_label 6-way).
REVISION = {"trec": "refs/convert/parquet"}
TASK_TYPE = {"cola": "acceptability", "sst2": "sentiment", "imdb": "sentiment", "yelp_polarity": "sentiment",
             "rotten_tomatoes": "sentiment", "emotion": "sentiment", "mrpc": "paraphrase", "qqp": "paraphrase",
             "paws": "paraphrase", "stsb": "similarity", "mnli": "nli", "qnli": "nli", "rte": "nli", "snli": "nli",
             "scitail": "nli", "boolq": "qa", "wic": "wsd", "ag_news": "topic", "trec": "question-type"}


def load_raw(task, split):
    from datasets import load_dataset
    path, cfg, *_ = ALL_SPECS[task]
    kw = {"revision": REVISION[task]} if task in REVISION else {}
    ds = load_dataset(path, cfg, split=split, **kw) if cfg else load_dataset(path, split=split, **kw)
    if task == "snli":
        ds = ds.filter(lambda x: x["label"] != -1)
    return ds


def columns(task, ds):
    """Returns (text_a list, text_b list or None, labels np.array)."""
    _, _, _, _, ka, kb, kl, nl, _ = ALL_SPECS[task]
    if ka == "WIC_A":
        a = [f"{w}: {s}" for w, s in zip(ds["word"], ds["sentence1"])]
    else:
        a = [str(x) for x in ds[ka]]
    b = [str(x) for x in ds[kb]] if kb else None
    if kl == "SCITAIL_LABEL":
        y = np.array([1 if str(v).strip().lower() in ("entails", "entailment") else 0 for v in ds["label"]], dtype=np.int64)
    elif nl == 1:
        y = np.array(ds[kl], dtype=np.float32)
    else:
        y = np.array(ds[kl], dtype=np.int64)
    return a, b, y


def split_indices(n_train_total, n_eval_total):
    """Pre-registered seeded splits. hold: 1000 train idx (lambda selection only); train: remaining pool, capped
    at TRAIN_CAP by seeded subsample; eval: all or seeded EVAL_CAP subset. All sorted."""
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_train_total)
    hold = np.sort(perm[:N_HOLD])
    pool = perm[N_HOLD:]
    train = np.sort(pool[:TRAIN_CAP]) if len(pool) > TRAIN_CAP else np.sort(pool)
    rng2 = np.random.default_rng(SEED + 1)
    ev = np.sort(rng2.choice(n_eval_total, EVAL_CAP, replace=False)) if n_eval_total > EVAL_CAP else np.arange(n_eval_total)
    return hold, train, ev


def tokenize(tok, a, b):
    """Standard BERT segment ids (token_type_ids from the tokenizer: 0 for segment A incl. [CLS]/first [SEP], 1 for B)."""
    if b is None:
        enc = tok(a, truncation=True, max_length=MAXLEN)
    else:
        enc = tok(a, b, truncation=True, max_length=MAXLEN)
    return enc["input_ids"], enc["token_type_ids"]
