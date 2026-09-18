#!/usr/bin/env python3
"""θ★ sensitivity sweep + optional conflict-pair runs (cert + GLUE eval n=512)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

from lora_merge_cert.merge import merge_lora_models, apply_deltas_to_base
from lora_merge_cert.eval import evaluate_glue
from lora_merge_cert.certificate import fake_eval_from_certificates  # may not exist
