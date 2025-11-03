"""
Reddit-CTRL dataset builder (using ConvoKit Reddit Corpus by Subreddit)
----------------------------------------------------------------------
Creates GPT-2 tokenizer-compatible .bin shards identical to FineWeb format.

Each document = [<|endoftext|>, <r/subreddit>, text tokens...]
Compatible with your distributed_data_generator().

Requires:
    pip install convokit tiktoken datasets tqdm
"""

import os
import re
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from tqdm import tqdm
#from convokit import Corpus, download, list_downloadable_corpora
#from convokit import Corpus, download
#from convokit.download_corpus import list_downloadable_corpora
# === Robust import for all ConvoKit versions ===
try:
    from convokit import Corpus, download
    from convokit.download_corpus import list_downloadable_corpora  # newer versions (>=3.0.0)
except ImportError:
    try:
        from convokit import Corpus, download
        from convokit.corpus.download import list_downloadable_corpora  # older versions (<3.0.0)
    except ImportError:
        raise ImportError(
            "Could not find list_downloadable_corpora in convokit. "
            "Try installing an explicit version: pip install convokit==3.0.0"
        )
# ===============================================================
# Binary writer (identical to FineWeb)
# ===============================================================
def write_datafile(filename, toks):
    assert len(toks) < 2**31, "too many tokens in shard"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20250103  # magic
    header[1] = 1         # version
    header[2] = len(toks)
    toks_np = np.array(toks, dtype=np.uint16)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())
    print(f"[+] wrote {len(toks):,} tokens ?~F~R {filename}")


# ===============================================================
# Text cleanup
# ===============================================================
def clean_text(t):
    if not t:
        return ""
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"&gt;|&lt;|&amp;", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ===============================================================
# Global tokenizer (same as FineWeb)
# ===============================================================
enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]


def tokenize_one(doc):
    """doc = (subreddit, text)"""
    sub, text = doc
    text = clean_text(text)
    if len(text) < 10:
        return np.array([], dtype=np.uint16)
    prefix = f"<r/{sub}> "
    toks = [EOT]
    toks.extend(enc.encode_ordinary(prefix + text))
    toks_np = np.array(toks, dtype=np.uint16)
    return toks_np


# ===============================================================
# Subreddit iterator
# ===============================================================
def iter_subreddits(chosen_subs, max_utts_per_sub=20000):
    """Yields (subreddit, text) pairs across multiple subreddit corpora."""
    for sub_name in chosen_subs:
        clean_sub = sub_name.replace("subreddit-", "")
        try:
            corpus = Corpus(filename=download(sub_name))
        except Exception as e:
            print(f"[!] Failed to load {sub_name}: {e}")
            continue
        count = 0
        for utt in corpus.iter_utterances():
            if utt.text and len(utt.text) > 10:
                yield (clean_sub, utt.text)
                count += 1
                if count >= max_utts_per_sub:
                    break
        print(f"[?~\~S] Loaded {count} utterances from {clean_sub}")


# ===============================================================
# Main
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="reddit_ctrl_full",
                        help="output folder for .bin shards")
    parser.add_argument("--num_subs", type=int, default=100,
                        help="number of subreddit corpora to use")
    parser.add_argument("--max_utts_per_sub", type=int, default=20000,
                        help="max utterances per subreddit corpus")
    parser.add_argument("--shard_size", type=int, default=10**7,
                        help="tokens per shard (~10M recommended)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------------------------------------
    # 1?~O?~C? Get available subreddit corpora
    # -----------------------------------------------------------
    all_corpora = list_downloadable_corpora()
    subs = [c for c in all_corpora if c.startswith("subreddit-")]
    chosen = subs[: args.num_subs]
    print(f"[+] Using {len(chosen)} subreddit corpora")

    # -----------------------------------------------------------
    # 2?~O?~C? Stream texts ?~F~R tokens ?~F~R shards
    # -----------------------------------------------------------
    nprocs = max(1, os.cpu_count() - 2)
    shard_index, token_count = 0, 0
    buf = np.empty((args.shard_size,), dtype=np.uint16)
    pbar = None

    with mp.Pool(nprocs) as pool:
        data_stream = iter_subreddits(chosen, max_utts_per_sub=args.max_utts_per_sub)
        for toks in pool.imap(tokenize_one, data_stream, chunksize=16):
            if len(toks) == 0:
                continue
            if token_count + len(toks) < args.shard_size:
                buf[token_count:token_count + len(toks)] = toks
                token_count += len(toks)
                if pbar is None:
                    pbar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                pbar.update(len(toks))
            else:
                remainder = args.shard_size - token_count
                buf[token_count:token_count + remainder] = toks[:remainder]
                split = "val" if shard_index == 0 else "train"
                fname = os.path.join(args.out_dir, f"reddit_{split}_{shard_index:06d}.bin")
                write_datafile(fname, buf)
                shard_index += 1
                pbar = None
                buf[:len(toks) - remainder] = toks[remainder:]
                token_count = len(toks) - remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            fname = os.path.join(args.out_dir, f"reddit_{split}_{shard_index:06d}.bin")
            write_datafile(fname, buf[:token_count])
            if pbar:
                pbar.close()

    print("[?~\~S] Finished building Reddit CTRL dataset.")

~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
~                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
~                                                                   
