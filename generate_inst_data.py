#!/usr/bin/env python3
"""
OpenWebText-INS/CTX Instruction Dataset Builder
-----------------------------------------------
Builds GPT-2 tokenizer-compatible .bin shards (FineWeb-style),
with mixed language-generation and comprehension tasks.

Each sample is preceded by <|endoftext|> and formatted as:

# Generation (learn to write/continue language)
<ins>Guess the next token, if you are not sure say IDK.<ins> {TEXT}

# Understanding (learn to read/understand context)
<ins>{QUESTION}<ins><ctx>{TEXT}<ctx> {ANSWER}

Usage:
    python generate_inst_data.py --out_dir owt_instruct --shard_size 10000000 --next_token_prob 0.5 --num_samples 500000

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
# 1) Binary writer (FineWeb-compatible)
# ===============================================================
def write_datafile(filename, toks):
    assert len(toks) < 2**31, "too many tokens in shard"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20251104  # magic number (date-ish)
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
def clean_text(t: str, max_len: int = 400):
    if not t:
        return ""
    t = re.sub(r"http\\S+", "", t)
    t = re.sub(r"&gt;|&lt;|&amp;", "", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip()
    return t[:max_len]

# ===============================================================
# 3) GPT-2 Tokenizer (extended safely)
#    We add <ins> and <ctx> just after GPT-2's <|endoftext|> (50256)
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

# quick sanity check
_sample = "<ins>Test<ins><ctx>Hello world<ctx> yes"
ids = enc.encode(_sample, allowed_special={"<ins>", "<ctx>"})
print("[debug] encode ids:", ids[:20], "...")
print("[debug] decode back:", enc.decode(ids))

# ===============================================================
# 4) Instruction generator
#    Mix of generation & understanding tasks.
# ===============================================================
UNDERSTANDING_TASKS = [
    "first_word", "is_question", "word_count", "has_numbers",
    "contains_the", "color_strawberry", "capital_france",
    "legs_spider", "planet_earth", "opposite_cold",
    "contains_year", "tone_formality", "main_topic_guess",
    "sentence_count",
]

def make_instruct_sample(text: str, next_token_prob: float) -> str:
    """
    Returns a single sample string with either:
      - Generation format: <ins>{instr}<ins> {text}
      - Understanding format: <ins>{instr}<ins><ctx>{text}<ctx> {answer}
    """
    text = clean_text(text)
    if not text:
        return ""

    # --- Generation task (learn to continue text) ---
    if random.random() < next_token_prob:
        instr = "Guess the next token, if you are not sure say IDK."
        return f"<ins>{instr}<ins> {text}"

    # --- Understanding tasks (reason about given context) ---
    task = random.choice(UNDERSTANDING_TASKS)
    ans, instr = "IDK", ""

    if task == "first_word":
        ans = text.split()[0] if text.split() else "IDK"
        instr = "What is the first word of the following text?"

    elif task == "is_question":
        ans = "yes" if "?" in text else "no"
        instr = "Does the following text ask a question?"

    elif task == "word_count":
        ans = str(len(text.split()))
        instr = "How many words are in the following text?"

    elif task == "has_numbers":
        ans = "yes" if re.search(r"\d", text) else "no"
        instr = "Does the following text contain any numbers?"

    elif task == "contains_the":
        ans = "yes" if re.search(r"\bthe\b", text.lower()) else "no"
        instr = "Does the following text contain the word 'the'?"

    elif task == "color_strawberry":
        ans = "red"
        instr = "Ignore the text and answer: what is the color of a strawberry?"

    elif task == "capital_france":
        ans = "Paris"
        instr = "Ignore the text and answer: what is the capital of France?"

    elif task == "legs_spider":
        ans = "8"
        instr = "Ignore the text and answer: how many legs does a spider have?"

    elif task == "planet_earth":
        ans = "Earth"
        instr = "Ignore the text and answer: what planet do we live on?"

    elif task == "opposite_cold":
        ans = "hot"
        instr = "Ignore the text and answer: what is the opposite of 'cold'?"

    elif task == "contains_year":
        ans = "yes" if re.search(r"\b(19|20)\d{2}\b", text) else "no"
        instr = "Does the following text contain a year?"

    elif task == "tone_formality":
        if any(w in text.lower() for w in ["hey", "lol", "dude", "gonna", "wanna", "btw"]):
            ans = "informal"
        else:
            ans = "formal"
        instr = "Is the following text formal or informal?"

    elif task == "main_topic_guess":
        tl = text.lower()
        if any(w in tl for w in ["economy", "money", "finance", "market", "stock"]):
            ans = "economy"
        elif any(w in tl for w in ["computer", "ai", "data", "machine", "software", "model"]):
            ans = "technology"
        elif any(w in tl for w in ["art", "music", "painting", "literature"]):
            ans = "art"
        elif any(w in tl for w in ["government", "president", "election", "policy", "senate"]):
            ans = "politics"
        else:
            ans = "IDK"
        instr = "What is the main topic of the following text?"

    elif task == "sentence_count":
        ans = str(len(re.findall(r"[.!?]", text)))
        instr = "How many sentences are in the following text?"

    return f"<ins>{instr}<ins><ctx>{text}<ctx> {ans}"

# ===============================================================
# 5) Tokenizer helper
# ===============================================================
def tokenize_one(sample_text: str):
    """
    Formats and tokenizes a single sample. Returns uint16 np.array token ids
    with an <|endoftext|> token prepended (FineWeb convention).
    """
    if not sample_text or len(sample_text) < 5:
        return np.array([], dtype=np.uint16)

    toks = [EOT]
    toks.extend(enc.encode(sample_text, allowed_special={"<ins>", "<ctx>"}))
    return np.array(toks, dtype=np.uint16)

# ===============================================================
# 6) Main
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="owt_instruct",
                        help="output folder for .bin shards")
    parser.add_argument("--shard_size", type=int, default=10**7,
                        help="tokens per shard (~10M recommended)")
    parser.add_argument("--next_token_prob", type=float, default=0.3,
                        help="probability of generation (vs understanding) tasks")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="subset size for debugging; default = full dataset")
    parser.add_argument("--seed", type=int, default=1234,
                        help="rng seed for task sampling")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # -----------------------------------------------------------
    # Load OpenWebText (streaming or full, depending on num_samples)
    # -----------------------------------------------------------
    print("[+] Loading Skylion007/openwebtext ...")
    
    if args.num_samples is not None and args.num_samples <= 5000:
        # ⚡ Light mode for Colab/debugging: use streaming API
        ds_stream = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        ds = []
        print(f"[+] Streaming first {args.num_samples} samples ...")
        for i, ex in enumerate(ds_stream.take(args.num_samples)):
            ds.append(ex)
        print(f"[✓] Loaded {len(ds)} samples in streaming mode.")
    else:
        # 💾 Full download mode for large-scale generation
        ds = load_dataset("Skylion007/openwebtext", split="train")
        if args.num_samples is not None:
            ds = ds.select(range(min(args.num_samples, len(ds))))
            print(f"[+] Using subset: {len(ds)} samples")
    
    # -----------------------------------------------------------
    # Iterate and tokenize into shards
    # -----------------------------------------------------------
    class ShardBuffer:
        def __init__(self, out_dir: str, shard_size: int):
            self.out_dir = out_dir
            self.shard_size = shard_size
            self.shard_index = 0
            self.token_count = 0
            self.buf = np.empty((shard_size,), dtype=np.uint16)
            self.pbar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {self.shard_index}")
    
        def flush(self):
            if self.token_count == 0:
                return
            split = "val" if self.shard_index == 0 else "train"
            fname = os.path.join(self.out_dir, f"owt_{split}_{self.shard_index:06d}.bin")
            write_datafile(fname, self.buf[:self.token_count])
            self.shard_index += 1
            self.token_count = 0
            self.pbar.close()
            self.pbar = tqdm(total=self.shard_size, unit="tokens", desc=f"Shard {self.shard_index}")
    
        def add(self, toks: np.ndarray):
            n = len(toks)
            if self.token_count + n <= self.shard_size:
                self.buf[self.token_count:self.token_count + n] = toks
                self.token_count += n
                self.pbar.update(n)
                return
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
    
    shards = ShardBuffer(args.out_dir, args.shard_size)
    
    # Handle both list (non-streaming) and iterable (streaming)
    iterator = ds if isinstance(ds, list) else ds
    
    for ex in iterator:
        raw = ex.get("text", "") if isinstance(ex, dict) else ex["text"]
        text = clean_text(raw)
        if len(text) < 10:
            continue
    
        formatted = make_instruct_sample(text, next_token_prob=args.next_token_prob)
        if not formatted:
            continue
    
        toks = tokenize_one(formatted)
        if len(toks) == 0:
            continue
    
        shards.add(toks)
    
    shards.flush()
    print("[✓] Finished building OpenWebText instruction dataset with <ins>/<ctx> markers.")

