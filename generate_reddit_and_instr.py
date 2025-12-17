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
        "Guess what token should come next. If unsure, respond with IDK.",
        "Predict the next word or symbol. Say IDK if you cannot tell.",
        "Try to continue the sequence by giving the next token. If you don't know, answer IDK.",
        "What comes next in the text? Reply with IDK if uncertain.",
        "Continue the text by one token if possible; otherwise answer IDK."
    ],

    "first_word": [
        "What is the first word of the following text?",
        "Identify the opening word of this text.",
        "Which word starts the text below?",
        "Find the very first word appearing in the passage.",
        "Tell me the first word that appears in the text."
    ],

    "is_question": [
        "Does the following text ask a question?",
        "Is the sentence below phrased as a question?",
        "Determine if the text ends with a question or not.",
        "Is this text interrogative in nature?",
        "Does the text below include a question mark?"
    ],

    "word_count": [
        "How many words are in the following text?",
        "Count the total number of words in the text below.",
        "Provide the number of words appearing in this passage.",
        "What is the word count of the text?",
        "Give the total word count for the following text."
    ],

    "has_numbers": [
        "Does the following text contain any numbers?",
        "Check if there are any digits in the text below.",
        "Does this passage include numeric characters?",
        "Are there numbers present in the text?",
        "Does the text have any numerical information?"
    ],

    "contains_the": [
        "Does the following text contain the word 'the'?",
        "Check whether the text includes the word 'the'.",
        "Does the passage make use of the term 'the'?",
        "Identify if 'the' appears in the text below.",
        "Determine whether the word 'the' exists in this text."
    ],

    "color_strawberry": [
        "Ignore the text and answer: what color is a strawberry?",
        "Disregard the passage: what is the color of a strawberry?",
        "Answer this: what color are strawberries, ignoring the text?",
        "Without using the text, say what color a strawberry is.",
        "Ignore the text below — tell the color of a strawberry."
    ],

    "capital_france": [
        "Ignore the text and answer: what is the capital of France?",
        "Disregard the passage and name France's capital city.",
        "Without using the text, what city is the capital of France?",
        "Answer briefly: what is France’s capital?",
        "Ignore the text — tell me the capital of France."
    ],

    "legs_spider": [
        "Ignore the text and answer: how many legs does a spider have?",
        "Disregard the passage and tell how many legs spiders possess.",
        "Without reading the text, answer: number of legs on a spider?",
        "Say how many legs a spider has, ignoring the text.",
        "Ignore the text: state the count of a spider’s legs."
    ],

    "planet_earth": [
        "Ignore the text and answer: what planet do we live on?",
        "Disregard the passage: which planet is home to humans?",
        "Without referring to the text, name the planet we inhabit.",
        "Answer this simple question: what planet are we on?",
        "Ignore the text — say the planet we live on."
    ],

    "opposite_cold": [
        "Ignore the text and answer: what is the opposite of 'cold'?",
        "Disregard the text: give the antonym of 'cold'.",
        "Without reading the text, say what word is opposite to 'cold'.",
        "Answer directly: the opposite of cold is what?",
        "Ignore the text below — provide the opposite of 'cold'."
    ],

    "contains_year": [
        "Does the following text contain a year?",
        "Check if the text includes a specific year like 1999 or 2020.",
        "Does the passage mention any year?",
        "Determine whether a year appears in the text.",
        "Does the text reference any year numerically?"
    ],

    "tone_formality": [
        "Is the following text formal or informal?",
        "Decide whether the text has a formal or casual tone.",
        "Would you describe this text as formal or informal?",
        "Judge the tone of the passage: formal or informal?",
        "Classify the following text’s tone as formal or informal."
    ],

    "main_topic_guess": [
        "What is the main topic of the following text?",
        "Identify the central subject discussed in the text.",
        "Which topic does the text mainly focus on?",
        "Summarize the main theme of this passage in one word.",
        "Guess the general topic of the following text."
    ],

    "sentence_count": [
        "How many sentences are in the following text?",
        "Count the number of sentences within this passage.",
        "Provide the total sentence count for the text below.",
        "Tell how many sentences the following text contains.",
        "Estimate how many sentences are present in the text."
    ]
}

UNDERSTANDING_TASKS = [
    "first_word", "is_question", "word_count", "has_numbers",
    "contains_the", "color_strawberry", "capital_france",
    "legs_spider", "planet_earth", "opposite_cold",
    "contains_year", "tone_formality", "main_topic_guess",
    "sentence_count",
]
def make_owt_sample(text, next_token_prob, idk_prob):
    if random.random() < next_token_prob:
        instr = random.choice(INSTRUCTION_VARIANTS["next_token"])
        return f"<ins>{instr}<ins> {text}"

    
    task = random.choice(UNDERSTANDING_TASKS)      
    ans, instr = "IDK", random.choice(INSTRUCTION_VARIANTS[task])
    
    if random.random() < idk_prob:
        return f"<ins>{instr}<ins><ctx>{text}<ctx> {ans}"
      
  
    if task == "first_word":
        ans = text.split()[0] if text.split() else "IDK"

    elif task == "is_question":
        ans = "yes" if "?" in text else "no"

    elif task == "word_count":
        ans = str(len(text.split()))

    elif task == "has_numbers":
        ans = "yes" if re.search(r"\d", text) else "no"

    elif task == "contains_the":
        ans = "yes" if re.search(r"\bthe\b", text.lower()) else "no"

    elif task == "color_strawberry":
        ans = "red"

    elif task == "capital_france":
        ans = "Paris"

    elif task == "legs_spider":
        ans = "8"

    elif task == "planet_earth":
        ans = "Earth"

    elif task == "opposite_cold":
        ans = "hot"

    elif task == "contains_year":
        ans = "yes" if re.search(r"\b(19|20)\d{2}\b", text) else "no"

    elif task == "tone_formality":
        if any(w in text.lower() for w in ["hey", "lol", "dude", "gonna", "wanna", "btw"]):
            ans = "informal"
        else:
            ans = "formal"

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

    elif task == "sentence_count":
        ans = str(max(1, len(re.findall(r"[.!?]", text))))

    return f"<ins>{instr}<ins><ctx>{text}<ctx> {ans}"

# ===============================================================
# 6) Reddit instruction logic
# ===============================================================
REDDIT_50_SUBS = [
    "programming", "science", "technology", "books", "gaming",
    "Fitness", "travel", "personalfinance", "philosophy", "history"
]

SUBREDDIT_INSTRUCTIONS = [
    "Guess the subreddit of the following post. Say IDK if you are not sure.",
    "Which subreddit does this post belong to? Say IDK if you are not sure.",
]

def make_reddit_sample(text, subreddit, idk_prob):
    instr = random.choice(SUBREDDIT_INSTRUCTIONS)
    if random.random() < idk_prob:
        return f"<ins>{instr}<ins><ctx>{text}<ctx> IDK"
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
def iter_owt_tokens(ds, next_token_prob,idk_prob):
    for ex in ds:
        text = clean_text(ex["text"])
        if len(text) < 10:
            continue
        sample = make_owt_sample(text, next_token_prob, idk_prob)
        yield tokenize(sample)

def iter_reddit_tokens(idk_prob, max_per_sub=None):
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
            yield tokenize(make_reddit_sample(text, sub, idk_prob))
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
    parser.add_argument("--idk_prob", type=float, default=0.1)
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
        iter_owt_tokens(owt_ds, args.next_token_prob,args.idk_prob),
        iter_reddit_tokens(args.idk_prob, args.reddit_max_per_sub),
        args.reddit_token_fraction,
        shard_buf,
        args.max_tokens,
        args.seed,
    )

    shard_buf.flush()
    print("[✓] Finished mixed pretraining dataset")

if __name__ == "__main__":
    main()
