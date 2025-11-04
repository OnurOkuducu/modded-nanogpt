"""
Reddit-CTRL Instruction Dataset Builder
---------------------------------------
Builds GPT-2 tokenizer-compatible .bin shards identical to FineWeb format,
but formatted for instruction tuning.

Each sample:
<|endoftext|><ins>{subreddit}<ins><ans>{text}<ans>

Usage:
    python reddit_ins_builder.py --out_dir reddit_instruct_small

Requirements:
    pip install convokit tiktoken tqdm numpy
"""

import os
import re
import argparse
import numpy as np
import tiktoken
from tqdm import tqdm
from convokit import Corpus, download


# ===============================================================
# 1?~O?~C? Binary writer (FineWeb-compatible)
# ===============================================================
def write_datafile(filename, toks):
    assert len(toks) < 2**31, "too many tokens in shard"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20251103  # magic number
    header[1] = 1         # version
    header[2] = len(toks)
    toks_np = np.array(toks, dtype=np.uint16)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())
    print(f"[+] wrote {len(toks):,} tokens ?~F~R {filename}")


# ===============================================================
# 2?~O?~C? Text cleanup
# ===============================================================
def clean_text(t):
    if not t:
        return ""
    t = re.sub(r"http\\S+", "", t)
    t = re.sub(r"&gt;|&lt;|&amp;", "", t)
    t = re.sub(r"\\s+", " ", t)
    return t.strip()


# ===============================================================
# 3?~O?~C? GPT-2 Tokenizer (extended safely)
# ===============================================================
base_enc = tiktoken.get_encoding("gpt2")

# Add new special tokens *after* GPT-2?~@~Ys <|endoftext|> (50256)
custom_specials = {"<ins>": 50257, "<ans>": 50258}

enc = tiktoken.Encoding(
    name="gpt2-with-ins-ans",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens={**base_enc._special_tokens, **custom_specials},
)

EOT = enc._special_tokens["<|endoftext|>"]

print(f"[+] Added custom tokens: {list(custom_specials.keys())}")
print(f"[+] Total vocab size: {enc.n_vocab}")

sample = "<ins>AskReddit<ins><ans>What is your favorite movie?<ans>"

# Encode with special tokens allowed
ids = enc.encode(sample, allowed_special={"<ins>", "<ans>"})
print("Token IDs:", ids)

# Decode back
decoded = enc.decode(ids)
print("Decoded:", decoded)

# ===============================================================
# 4?~O?~C? Tokenizer helper
# ===============================================================
def tokenize_one(sub, text):
    text = clean_text(text)
    if len(text) < 10:
        return np.array([], dtype=np.uint16)

    # instruction-following format
    formatted = f"<ins>{sub}<ins><ans>{text}<ans>"
    toks = [EOT]
    toks.extend(enc.encode(formatted, allowed_special={"<ins>", "<ans>", "<|endoftext|>"}))
    return np.array(toks, dtype=np.uint16)

# ===============================================================
# 5?~O?~C? Main
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="reddit_instruct_small",
                        help="output folder for .bin shards")
    parser.add_argument("--shard_size", type=int, default=10**7,
                        help="tokens per shard (~10M recommended)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------------------------------------
    # Load reddit-corpus-small
    # -----------------------------------------------------------
    print("[+] Downloading and loading reddit-corpus-small...")
    corpus = Corpus(filename=download("reddit-corpus-small"))
    corpus.print_summary_stats()

    # -----------------------------------------------------------
    # Iterate and tokenize
    # -----------------------------------------------------------
    shard_index, token_count = 0, 0
    buf = np.empty((args.shard_size,), dtype=np.uint16)
    pbar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")

    for utt in corpus.iter_utterances():
        if not utt.text or len(utt.text) < 10:
            continue
        sub = utt.meta.get("subreddit", "unknown")
        toks = tokenize_one(sub, utt.text)
        if len(toks) == 0:
            continue

        # fill shard buffer
        if token_count + len(toks) < args.shard_size:
            buf[token_count:token_count + len(toks)] = toks
            token_count += len(toks)
            pbar.update(len(toks))
        else:
            remainder = args.shard_size - token_count
            buf[token_count:token_count + remainder] = toks[:remainder]
            split = "val" if shard_index == 0 else "train"
            fname = os.path.join(args.out_dir, f"reddit_{split}_{shard_index:06d}.bin")
            write_datafile(fname, buf)
            shard_index += 1
            buf[:len(toks) - remainder] = toks[remainder:]
            token_count = len(toks) - remainder
            pbar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")

    # write final shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        fname = os.path.join(args.out_dir, f"reddit_{split}_{shard_index:06d}.bin")
        write_datafile(fname, buf[:token_count])
        pbar.close()

    print("[?~\~S] Finished building Reddit Instruction dataset with <ins>/<ans> markers.")

