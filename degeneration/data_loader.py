"""Dataset + collation: per-token sliding-window 1 - TTR labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset


# -------------------------------------------------------------------
# Label computation
# -------------------------------------------------------------------


def _ngram_ttr(token_ids: Sequence[int], n: int) -> float:
    """Type-token ratio over n-grams. 1.0 if too few tokens."""
    if len(token_ids) < n or n < 1:
        return 1.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def sliding_window_repetition(
    token_ids: Sequence[int],
    window_size: int,
    n: int,
) -> List[float]:
    """
    For each position t in `token_ids`, return 1 - TTR computed over the
    window `token_ids[t : t + window_size]`. Positions where the full window
    does not fit (i.e. t + window_size > len) return NaN — the caller is
    responsible for masking these out of the loss.
    """
    out: List[float] = []
    total = len(token_ids)
    for t in range(total):
        end = t + window_size
        if end > total:
            out.append(float("nan"))
        else:
            out.append(1.0 - _ngram_ttr(token_ids[t:end], n))
    return out


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------


@dataclass
class DegenerationItem:
    prompt: str
    completion: str


class DegenerationDataset(Dataset):
    """
    Reads a `generations.jsonl` file (as written by degeneration_probe's
    `generate` command). Each record supplies a `prompt` and `generated_text`;
    labels are computed on-the-fly in the collate function from the tokenised
    completion.
    """

    def __init__(self, paths: str | Path | List[str | Path]) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.items: List[DegenerationItem] = []
        for path in paths:
            with open(path) as f:
                for line in f:
                    record = json.loads(line)
                    prompt = record["prompt"]
                    completion = record.get(
                        "generated_text", record.get("completion", "")
                    )
                    self.items.append(
                        DegenerationItem(prompt=prompt, completion=completion)
                    )

    @classmethod
    def from_hf(
        cls,
        name: str,
        split: str = "train",
        *,
        max_rows: int | None = None,
        prompt_field: str = "prompt",
        completion_field: str = "generated_text",
    ) -> "DegenerationDataset":
        """Build a dataset from a HuggingFace Hub dataset.

        Streams the requested split into memory and pulls `prompt` /
        `generated_text` fields. Field names are configurable for datasets
        that use a different schema.
        """
        from datasets import load_dataset

        ds = load_dataset(name, split=split)
        if max_rows is not None:
            ds = ds.select(range(min(max_rows, len(ds))))
        obj = cls.__new__(cls)
        obj.items = [
            DegenerationItem(
                prompt=row[prompt_field],
                completion=row[completion_field],
            )
            for row in ds
        ]
        return obj

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DegenerationItem:
        return self.items[idx]


# -------------------------------------------------------------------
# Collation
# -------------------------------------------------------------------


def make_collate_fn(
    tokenizer,
    *,
    max_length: int,
    window_size: int,
    primary_n: int,
):
    """
    Return a collate function that:
      - tokenises `prompt + completion` (prompt separately to locate the
        completion boundary);
      - produces `input_ids`, `attention_mask` (standard HF shapes);
      - produces per-token `labels` in [0, 1] = 1 - TTR over the next
        `window_size` completion tokens, with n-gram size `primary_n`;
      - produces `label_mask` marking positions where a label is valid
        (i.e. a token inside the completion AND the forward window fits).
    """

    def collate_fn(batch: List[DegenerationItem]) -> Dict[str, torch.Tensor]:
        prompts = [item.prompt for item in batch]
        completions = [item.completion for item in batch]

        prompt_enc = tokenizer(
            prompts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        prompt_lengths = [len(ids) for ids in prompt_enc["input_ids"]]

        full_texts = [p + c for p, c in zip(prompts, completions)]
        full_enc = tokenizer(
            full_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        B, T = input_ids.shape
        labels = torch.zeros(B, T, dtype=torch.float32)
        label_mask = torch.zeros(B, T, dtype=torch.float32)

        for i, plen in enumerate(prompt_lengths):
            seq_len = int(attention_mask[i].sum().item())
            if plen >= seq_len:
                continue
            completion_ids = input_ids[i, plen:seq_len].tolist()
            # Per-completion-token labels: 1 - TTR of the next window_size
            # tokens. Positions without a full window get NaN → mask=0.
            rep = sliding_window_repetition(
                completion_ids, window_size=window_size, n=primary_n,
            )
            for k, r in enumerate(rep):
                pos = plen + k
                if not (r == r):  # NaN check
                    continue
                labels[i, pos] = r
                label_mask[i, pos] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,          # [B, T] — 1 - TTR over next window
            "label_mask": label_mask,  # [B, T] — 1 where label is valid
        }

    return collate_fn
