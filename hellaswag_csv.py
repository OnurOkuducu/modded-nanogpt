import os
import torch
import pandas as pd
from datasets import load_dataset
import tiktoken


from updated_abs_v2 import GPT

# ========= Config =========
device = "cuda"
vocab_size = 50259   # 50257 + <ins> + <ctx>
num_layers = 12
num_heads  = 6
model_dim  = 768
max_seq_len = 48 * 1024

CHECKPOINT = "/workspace/modded-nanogpt/logs/3c34de42-5da2-46fd-89f6-a130112b5906/state_step003999.pt"
N_EVAL     = 400      # set to len(ds) for full eval
BLOCK_SIZE = 128
PAD_TOKEN  = 50256
OUT_CSV    = "hellaswag_abstain_stats.csv"

# ========= Tokenizer with <ins> and <ctx> =========
base_enc = tiktoken.get_encoding("gpt2")
custom_specials = {"<ins>": 50257, "<ctx>": 50258}
enc = tiktoken.Encoding(
    name="gpt2-with-ins-ctx",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens={**base_enc._special_tokens, **custom_specials},
)
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>", "<ins>", "<ctx>"})
decode = lambda l: enc.decode(l)

# ========= Model loader =========
def load_model(path: str):
    model = GPT(vocab_size, num_layers, num_heads, model_dim, max_seq_len).to(device)
    state = torch.load(path, map_location=device)
    sd = state["model"]
    # De-compile prefix from torch.compile if present
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

@torch.no_grad()
def score_continuation_abstain(model, prompt: str, continuation: str):
    """
    Returns a dict with:
      - sum_lp, avg_lp
      - sum_lp_gated, avg_lp_gated     (no lambda penalty here)
      - mean_g, min_g                  (abstention head in continuation region)
    """
    # Tokenize and left-pad to a multiple of BLOCK_SIZE
    prompt_ids = torch.tensor(encode(prompt), dtype=torch.int32, device=device)
    cont_ids   = torch.tensor(encode(" " + continuation), dtype=torch.int32, device=device)
    input_ids  = torch.cat([prompt_ids, cont_ids])

    pad_len = (-len(input_ids)) % BLOCK_SIZE
    if pad_len > 0:
        pad = torch.full((pad_len,), PAD_TOKEN, dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([pad, input_ids])

    # Number of KV blocks for your FlexAttention
    sw_blocks = torch.tensor(max(1, len(input_ids) // BLOCK_SIZE), dtype=torch.int32, device=device)

    # Model.inference must return (logits [1,T,V] or [T,V], gates [T])
    logits, gates = model.inference(input_ids, sw_blocks, use_capped_logits=True, return_hidden=False)

    if logits.ndim == 3: logits = logits[0]  # [T,V]
    # gates expected shape [T]
    assert gates.ndim == 1, f"Expected gates [T], got {gates.shape}"

    # Next-token logprobs
    VOCAB_SIZE = getattr(model.lm_head, "out_features", 50257)
    logits   = logits[:-1, :VOCAB_SIZE]          # [T-1,V]
    logprobs = torch.log_softmax(logits, dim=-1) # [T-1,V]
    target   = input_ids[1:]                     # [T-1]
    gates_t  = gates[:-1]                        # [T-1]

    # Continuation slice starts at the boundary between prompt and continuation
    cont_start = len(prompt_ids)
    cont_start_idx = pad_len + cont_start

    lp_slice  = logprobs[cont_start_idx - 1 :, :]            # [C,V]
    tgt_slice = target  [cont_start_idx - 1 :]               # [C]
    g_slice   = gates_t [cont_start_idx - 1 :]               # [C]

    # Gather true-token logprobs
    token_lps = lp_slice.gather(1, tgt_slice.unsqueeze(-1)).squeeze(-1)  # [C]

    sum_lp = token_lps.sum().item()
    avg_lp = sum_lp / max(1, tgt_slice.numel())

    # Gated score (no lambda)
    sum_lp_gated = (g_slice * token_lps).sum().item()
    avg_lp_gated = sum_lp_gated / max(1, tgt_slice.numel())

    mean_gate = g_slice.mean().item() if g_slice.numel() else 1.0
    min_gate  = g_slice.min().item()  if g_slice.numel() else 1.0

    return {
        "sum_lp": sum_lp,
        "avg_lp": avg_lp,
        "sum_lp_gated": sum_lp_gated,
        "avg_lp_gated": avg_lp_gated,
        "mean_g": mean_gate,
        "min_g": min_gate,
    }

def main():
    model = load_model(CHECKPOINT)
    ds = load_dataset("hellaswag", split="validation")
    n = min(N_EVAL, len(ds))

    rows = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        # HellaSwag context and endings
        ctx = (ex["ctx_a"] + " " + ex["ctx_b"]).strip()
        endings = ex["endings"]
        gold = int(ex["label"])

        for j, cand in enumerate(endings):
            s = score_continuation_abstain(model, ctx, cand)
            rows.append({
                "example_id": i,
                "ctx": ctx,
                "option_idx": j,
                "option_text": cand,
                "is_gt": int(j == gold),
                "sum_lp": s["sum_lp"],
                "avg_lp": s["avg_lp"],
                "sum_lp_gated": s["sum_lp_gated"],
                "avg_lp_gated": s["avg_lp_gated"],
                "mean_g": s["mean_g"],
                "min_g": s["min_g"],
            })

        if (i + 1) % 50 == 0:
            print(f"Scored {i+1}/{n} examples...")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUT_CSV}")
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    assert torch.cuda.is_available(), "CUDA required for this script."
    main()
