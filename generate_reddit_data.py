#!/usr/bin/env python3
"""
<|endoftext|><ins>Guess the subreddit of the following post<ins><ctx>{TEXT}<ctx> r/{SUBREDDIT}

Writes GPT-2 tokenizer-compatible .bin shards (FineWeb-style).

Usage (examples):
    # Small debug run
    python generate_reddit_data.py \
        --out_dir reddit_50sub_debug \
        --num_samples 50000

    # Larger run with cap per subreddit
    python generate_reddit_data.py \
        --out_dir reddit_50sub_full \
        --num_samples 2000000 \
        --max_per_subreddit 50000

Requirements:
    pip install datasets==3.6.0 tiktoken tqdm numpy
"""

import os
import re
import argparse
import random
import numpy as np
from itertools import islice
from tqdm import tqdm
import tiktoken
from datasets import load_dataset


REDDIT_50_SUBS = [
    "programming",
    "tifu",
    "explainlikeimfive",
    "WritingPrompts",
    "changemyview",
    "LifeProTips",
    "todayilearned",
    "science",
    "askscience",
    "ifyoulikeblank",
    "Foodforthought",
    "IWantToLearn",
    "bestof",
    "IAmA",
    "socialskills",
    "relationship_advice",
    "philosophy",
    "YouShouldKnow",
    "history",
    "books",
    "Showerthoughts",
    "personalfinance",
    "buildapc",
    "EatCheapAndHealthy",
    "boardgames",
    "malefashionadvice",
    "femalefashionadvice",
    "scifi",
    "Fantasy",
    "Games",
    "bodyweightfitness",
    "SkincareAddiction",
    "podcasts",
    "suggestmeabook",
    "AskHistorians",
    "gaming",
    "DIY",
    "sports",
    "space",
    "gadgets",
    "Documentaries",
    "GetMotivated",
    "UpliftingNews",
    "technology",
    "Fitness",
    "travel",
    "lifehacks",
    "Damnthatsinteresting",
    "gardening",
    "mildlyinteresting",
]

# ===============================================================
# 1) Binary writer (FineWeb-compatible)
# ===============================================================
def write_datafile(filename, toks):
    assert len(toks) < 2**31, "too many tokens in shard"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20251119  # magic-ish
    header[1] = 1         # version
    header[2] = len(toks)
    toks_np = np.array(toks, dtype=np.uint16)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())
    print(f"[+] wrote {len(toks):,} tokens -> {filename}")

# ===============================================================
# 2) Text cleanup
# ===============================================================
def clean_text(t: str, max_len: int = 800):
    if not t:
        return ""
    # remove urls & html escapes
    t = re.sub(r"http\\S+|www\\S+", "", t)
    t = re.sub(r"&gt;|&lt;|&amp;", "", t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    t = t.strip()
    # Reddit-specific junk
    if t.lower() in ("[deleted]", "[removed]"):
        return ""
    return t[:max_len]

# ===============================================================
# 3) GPT-2 Tokenizer (extended with <ins>, <ctx>)
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

print(f"[+] Added custom tokens: {list(custom_specials.keys())}")
print(f"[+] Total vocab size: {enc.n_vocab}")

# sanity check
_sample = "<ins>Guess the subreddit of the following post<ins><ctx>Hello world<ctx> r/programming"
ids = enc.encode(_sample, allowed_special={"<ins>", "<ctx>"})
print("[debug] encode ids:", ids[:20], "...")
print("[debug] decode back:", enc.decode(ids))

# ===============================================================
# 4) Instruction template
# ===============================================================
SUBREDDIT_INSTRUCTIONS = [
    "Guess the subreddit of the following post.",
    "From the post below, infer which subreddit it comes from.",
    "Identify the most likely subreddit for this post.",
    "Which subreddit do you think this post belongs to?",
    "Guess the subreddit based on the following text.",
]

def format_reddit_example(post_text: str, subreddit_raw: str) -> str:
    """
    Build one sample:

    <ins>{INSTR}<ins><ctx>{POST_TEXT}<ctx> r/{SUB}
    """
    post_text = clean_text(post_text)
    subreddit_raw = (subreddit_raw or "").strip()

    if not post_text or not subreddit_raw:
        return ""

    instr = random.choice(SUBREDDIT_INSTRUCTIONS)

    # Normalize into "r/sub"
    if subreddit_raw.startswith("r/"):
        sub_label = subreddit_raw
    else:
        sub_label = f"r/{subreddit_raw}"

    return f"<ins>{instr}<ins><ctx>{post_text}<ctx> {sub_label}"

# ===============================================================
# 5) Tokenizer helper
# ===============================================================
def tokenize_one(sample_text: str) -> np.ndarray:
    """
    Return uint16 array, with <|endoftext|> prepended.
    """
    if not sample_text or len(sample_text) < 5:
        return np.array([], dtype=np.uint16)

    toks = [EOT]
    toks.extend(enc.encode(sample_text, allowed_special={"<ins>", "<ctx>"}))
    return np.array(toks, dtype=np.uint16)

# ===============================================================
# 6) Shard buffer
# ===============================================================
class ShardBuffer:
    def __init__(self, out_dir: str, shard_size: int):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.shard_index = 0
        self.token_count = 0
        self.buf = np.empty((shard_size,), dtype=np.uint16)
        self.pbar = tqdm(total=self.shard_size, unit="tokens",
                         desc=f"Shard {self.shard_index}")

    def flush(self):
        if self.token_count == 0:
            return
        split = "val" if self.shard_index == 0 else "train"
        fname = os.path.join(self.out_dir,
                             f"reddit50_{split}_{self.shard_index:06d}.bin")
        write_datafile(fname, self.buf[:self.token_count])
        self.shard_index += 1
        self.token_count = 0
        self.pbar.close()
        self.pbar = tqdm(total=self.shard_size, unit="tokens",
                         desc=f"Shard {self.shard_index}")

    def add(self, toks: np.ndarray):
        n = len(toks)
        if n == 0:
            return

        if self.token_count + n <= self.shard_size:
            self.buf[self.token_count:self.token_count + n] = toks
            self.token_count += n
            self.pbar.update(n)
            return

        # fill current shard
        remainder = self.shard_size - self.token_count
        if remainder > 0:
            self.buf[self.token_count:self.token_count + remainder] = toks[:remainder]
            self.pbar.update(remainder)

        self.flush()

        leftover = n - remainder
        if leftover > 0:
            self.buf[:leftover] = toks[remainder:]
            self.token_count = leftover
            self.pbar.update(leftover)

# ===============================================================
# 7) Main
# ===============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="reddit_50sub_instruct",
                        help="output folder for .bin shards")
    parser.add_argument("--shard_size", type=int, default=10**7,
                        help="tokens per shard (~10M)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="total number of examples across all subreddits")
    parser.add_argument("--max_per_subreddit", type=int, default=None,
                        help="optional cap per subreddit (None = unlimited)")
    parser.add_argument("--seed", type=int, default=1234,
                        help="random seed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    shards = ShardBuffer(args.out_dir, args.shard_size)

    total_samples = 0
    done = False

    for sub in REDDIT_50_SUBS:
        if done:
            break

        print(f"[+] Streaming subreddit split: {sub}")
        # HuggingFaceGECLM/REDDIT_comments uses each subreddit as a split. 
        ds_stream = load_dataset(
            "HuggingFaceGECLM/REDDIT_comments",
            split=sub,
            streaming=True,
        )

        per_sub_count = 0

        for ex in ds_stream:
            if args.max_per_subreddit is not None and per_sub_count >= args.max_per_subreddit:
                break
            if args.num_samples is not None and total_samples >= args.num_samples:
                done = True
                break

            body = ex.get("body", "") if isinstance(ex, dict) else ex["body"]
            body = body if isinstance(body, str) else str(body)

            text = clean_text(body)
            if len(text) < 10:
                continue

            # Prefer explicit label in row if present, else fallback to split name
            subreddit_label = ex.get("subreddit_name_prefixed") or sub

            formatted = format_reddit_example(text, subreddit_label)
            if not formatted:
                continue

            toks = tokenize_one(formatted)
            if len(toks) == 0:
                continue

            shards.add(toks)
            per_sub_count += 1
            total_samples += 1

        print(f"[+] Finished {sub}: used {per_sub_count} examples (total so far: {total_samples})")

    shards.flush()
    print(f"[✓] Done. Total examples: {total_samples}")

if __name__ == "__main__":
    main()
