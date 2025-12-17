#!/usr/bin/env python3
"""
Mixed Pretraining Dataset Generator
----------------------------------

Mixes:
  1) OpenWebText instruction-style data
  2) Reddit subreddit-guessing instruction data

Mixing is done at TOKEN LEVEL.

Example:
  python generate_mixed_pretrain_data.py \
    --out_dir pretrain_mix_v1 \
    --reddit_token_fraction 0.2 \
    --next_token_prob 0.3 \
    --max_tokens 50000000

Requirements:
  pip install datasets==3.6.0 tiktoken tqdm numpy
"""

import os
import re
import argparse
import random
import numpy as np
from tqdm import tqdm
import tiktoken
from datasets import load_dataset

# ===============================================================
# 1) Tokenizer (GPT-2 + <ins>, <ctx>)
# ===============================================================
base_enc = tiktoken.get_encoding("gpt2")
custom_specials = {"<ins>": 50257, "<ctx>": 50258}

enc = tiktoken.Encoding(
    name="gpt2-with-ins-ctx",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens={**base_enc._special_tokens, **custom_specials},
)

EOT = enc._special_tokens["<|endoftext|>"]

# ===============================================================
# 2) Binary writer (FineWeb compatible)
# ===============================================================
def write_datafile(filename, toks):
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20251216
    header[1] = 1
    header[2] = len(toks)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(np.array(toks, dtype=np.uint16).tobytes())
    print(f"[+] wrote {len(toks):,} tokens -> {filename}")

# ===============================================================
# 3) Shard buffer
# ===============================================================
class ShardBuffer:
    def __init__(self, out_dir, shard_size):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.buf = np.empty((shard_size,), dtype=np.uint16)
        self.token_count = 0
        self.shard_index = 0
        self.pbar = tqdm(total=shard_size, unit="tokens",
                         desc=f"Shard {self.shard_index}")

    def flush(self):
        if self.token_count == 0:
            return
        split = "val" if self.shard_index == 0 else "train"
        fname = os.path.join(
            self.out_dir,
            f"mixed_{split}_{self.shard_index:06d}.bin"
        )
        write_datafile(fname, self.buf[:self.token_count])
        self.shard_index += 1
        self.token_count = 0
        self.pbar.close()
        self.pbar = tqdm(total=self.shard_size, unit="tokens",
                         desc=f"Shard {self.shard_index}")

    def add(self, toks):
        n = len(toks)
        if self.token_count + n <= self.shard_size:
            self.buf[self.token_count:self.token_count+n] = toks
            self.token_count += n
            self.pbar.update(n)
            return

        rem = self.shard_size - self.token_count
        if rem > 0:
            self.buf[self.token_count:self.token_count+rem] = toks[:rem]
            self.pbar.update(rem)

        self.flush()

        leftover = n - rem
        if leftover > 0:
            self.buf[:leftover] = toks[rem:]
            self.token_count = leftover
            self.pbar.update(leftover)

# ===============================================================
# 4) Text cleanup
# ===============================================================
def clean_text(t, max_len=800):
    if not t:
        return ""
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"&gt;|&lt;|&amp;", "", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip()
    if t.lower() in ("[deleted]", "[removed]"):
        return ""
    return t[:max_len]

# ===============================================================
# 5) OpenWebText instruction logic
# ===============================================================
INSTRUCTION_VARIANTS = {
    "next_token": [
        "Guess the next token, if you are not sure say IDK.",
        "Predict the next token. Say IDK if unsure.",
        "Continue the text by one token, or say IDK.",
    ],
    "first_word": ["What is the first word of the following text?"],
    "is_question": ["Does the following text ask a question?"],
}

def make_owt_sample(text, next_token_prob):
    if random.random() < next_token_prob:
        instr = random.choice(INSTRUCTION_VARIANTS["next_token"])
        return f"<ins>{instr}<ins> {text}"

    task = random.choice(["first_word", "is_question"])
    instr = random.choice(INSTRUCTION_VARIANTS[task])

    if task == "first_word":
        ans = text.split()[0] if text.split() else "IDK"
    else:
        ans = "yes" if "?" in text else "no"

    return f"<ins>{instr}<ins><ctx>{text}<ctx> {ans}"

# ===============================================================
# 6) Reddit instruction logic
# ===============================================================
REDDIT_50_SUBS = [
    "programming", "science", "technology", "books", "gaming",
    "Fitness", "travel", "personalfinance", "philosophy", "history"
]

SUBREDDIT_INSTRUCTIONS = [
    "Guess the subreddit of the following post.",
    "Which subreddit does this post belong to?",
]

def make_reddit_sample(text, subreddit):
    instr = random.choice(SUBREDDIT_INSTRUCTIONS)
    return f"<ins>{instr}<ins><ctx>{text}<ctx> r/{subreddit}"

# ===============================================================
# 7) Tokenization
# ===============================================================
def tokenize(sample):
    toks = [EOT]
    toks.extend(enc.encode(sample, allowed_special={"<ins>", "<ctx>"}))
    return np.array(toks, dtype=np.uint16)

# ===============================================================
# 8) Token iterators
# ===============================================================
def iter_owt_tokens(ds, next_token_prob):
    for ex in ds:
        text = clean_text(ex["text"])
        if len(text) < 10:
            continue
        sample = make_owt_sample(text, next_token_prob)
        yield tokenize(sample)

def iter_reddit_tokens(max_per_sub=None):
    for sub in REDDIT_50_SUBS:
        ds = load_dataset(
            "HuggingFaceGECLM/REDDIT_comments",
            split=sub,
            streaming=True,
        )
        used = 0
        for ex in ds:
            if max_per_sub and used >= max_per_sub:
                break
            text = clean_text(ex.get("body", ""))
            if len(text) < 10:
                continue
            yield tokenize(make_reddit_sample(text, sub))
            used += 1

# ===============================================================
# 9) Mixer
# ===============================================================
def mix_streams(
    owt_iter,
    reddit_iter,
    reddit_fraction,
    shard_buf,
    max_tokens,
    seed,
):
    random.seed(seed)
    owt_iter = iter(owt_iter)
    reddit_iter = iter(reddit_iter)

    total = 0
    reddit_tokens = 0

    while True:
        if max_tokens and total >= max_tokens:
            break

        use_reddit = random.random() < reddit_fraction

        try:
            toks = next(reddit_iter if use_reddit else owt_iter)
            if use_reddit:
                reddit_tokens += len(toks)
        except StopIteration:
            try:
                toks = next(owt_iter if use_reddit else reddit_iter)
            except StopIteration:
                break

        shard_buf.add(toks)
        total += len(toks)

        if total % 1_000_000 < len(toks):
            frac = reddit_tokens / max(1, total)
            print(f"[mix] tokens={total:,} reddit_frac={frac:.3f}")

# ===============================================================
# 10) Main
# ===============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--shard_size", type=int, default=10**7)
    parser.add_argument("--reddit_token_fraction", type=float, default=0.1)
    parser.add_argument("--next_token_prob", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--reddit_max_per_sub", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[+] Loading OpenWebText (streaming)")
    owt_ds = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        streaming=True,
    )

    shard_buf = ShardBuffer(args.out_dir, args.shard_size)

    mix_streams(
        iter_owt_tokens(owt_ds, args.next_token_prob),
        iter_reddit_tokens(args.reddit_max_per_sub),
        args.reddit_token_fraction,
        shard_buf,
        args.max_tokens,
        args.seed,
    )

    shard_buf.flush()
    print("[✓] Finished mixed pretraining dataset")

if __name__ == "__main__":
    main()
