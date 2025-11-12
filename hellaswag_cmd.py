#!/usr/bin/env python3
import math
import sys
import torch
from datasets import load_dataset
import tiktoken
from train_gpt import GPT

# -----------------------------
# Parse command-line argument
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python eval_hellaswag.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = sys.argv[1]

# -----------------------------
# Config (same as your model)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 50257
num_layers = 12
num_heads = 6
model_dim = 768
max_seq_len = 6 * 2048

# -----------------------------
# Load model and checkpoint
# -----------------------------
print(f"[+] Loading model and checkpoint: {checkpoint_path}")
model = GPT(vocab_size, num_layers, num_heads, model_dim, max_seq_len).to(device)

state = torch.load(checkpoint_path, map_location=device)
state_dict = {k.replace("module.", ""): v for k, v in state["model"].items()}
model.load_state_dict(state_dict, strict=False)

state = torch.load(checkpoint_path, map_location=device)
state_dict = state["model"]

if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    print("Detected compiled checkpoint — stripping '_orig_mod.' prefixes...")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

model.eval()
print("✅ Model loaded successfully!")

# -----------------------------
# Tokenizer setup
# -----------------------------
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})

# -----------------------------
# Log-likelihood scoring
# -----------------------------
@torch.no_grad()
def score_continuation(prompt, continuation):
    # Build token IDs
    prompt_ids = torch.tensor(encode(prompt), dtype=torch.int32, device=device)
    cont_ids = torch.tensor(encode(" " + continuation), dtype=torch.int32, device=device)
    input_ids = torch.cat([prompt_ids, cont_ids])

    # Forward through model
    x = model.embed(input_ids)
    logits = model.lm_head(x)[:-1]  # predict next token for each position
    logits = logits[:, :vocab_size]
    logits = torch.logit(logits / 30, eps=1e-6) * 7.5
    logprobs = torch.log_softmax(logits, dim=-1)

    # Target tokens are the continuation
    target = input_ids[1:]
    cont_start = len(prompt_ids)
    cont_target = target[cont_start - 1 :]

    lp = logprobs[cont_start - 1 : len(input_ids) - 1, :]
    token_lps = lp.gather(1, cont_target.unsqueeze(-1)).squeeze(-1)
    sum_lp = token_lps.sum().item()
    avg_lp = sum_lp / max(1, len(cont_target))
    return sum_lp, avg_lp

# -----------------------------
# Evaluate on HellaSwag
# -----------------------------
print("[+] Loading HellaSwag validation set...")
ds = load_dataset("hellaswag", split="validation")

n = 400#len(ds)
correct = 0
correct_norm = 0

for i, ex in enumerate(ds):
    ctx = (ex["ctx_a"] + " " + ex["ctx_b"]).strip()
    endings = ex["endings"]
    gold = int(ex["label"])

    scores_sum = []
    scores_avg = []
    for cand in endings:
        s_sum, s_avg = score_continuation(ctx, cand)
        scores_sum.append(s_sum)
        scores_avg.append(s_avg)

    pred = int(max(range(4), key=lambda j: scores_sum[j]))
    pred_norm = int(max(range(4), key=lambda j: scores_avg[j]))

    correct += (pred == gold)
    correct_norm += (pred_norm == gold)

    if (i + 1) % 100 == 0:
        print(f"{i+1}/{n} examples processed... acc={correct/(i+1):.3f} acc_norm={correct_norm/(i+1):.3f}")
    
    if i == n:
        break
print("\n=== Final HellaSwag Results ===")
print(f"Accuracy (raw logprob): {correct / n:.4f}")
print(f"Accuracy (length-normalized): {correct_norm / n:.4f}  <-- standard metric")
