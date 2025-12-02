import os
import torch
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

CHECKPOINT = "/workspace/modded-nanogpt/logs/209738eb-0dbe-4a20-81d8-aa3612f9f0fb/state_step001999.pt"
N_EVAL     = 400   # set to len(ds) for full eval
BLOCK_SIZE = 128
PAD_TOKEN  = 50256

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

def load_model(path: str) -> GPT:
    model = GPT(vocab_size, num_layers, num_heads, model_dim, max_seq_len).to(device)
    state = torch.load(path, map_location=device)
    sd = state["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

@torch.no_grad()
def score_continuation_abstain(model: GPT, prompt: str, continuation: str, lambda_penalty: float | None = None):
    """
    Returns:
      dict(sum_lp, avg_lp, sum_lp_gated, avg_lp_gated, mean_gate)
      - logits are next-token; gates are per-token in the continuation region
      - 'gated' = g * logprob  (optionally minus (1-g)*lambda_penalty)
    """
    # Tokenize and left-pad to a multiple of BLOCK_SIZE
    prompt_ids = torch.tensor(encode(prompt), dtype=torch.int32, device=device)
    cont_ids   = torch.tensor(encode(" " + continuation), dtype=torch.int32, device=device)
    input_ids  = torch.cat([prompt_ids, cont_ids])

    pad_len = (-len(input_ids)) % BLOCK_SIZE
    if pad_len > 0:
        pad = torch.full((pad_len,), PAD_TOKEN, dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([pad, input_ids])

    sw_blocks = torch.tensor(max(1, len(input_ids) // BLOCK_SIZE), dtype=torch.int32, device=device)

    # Model.inference must return (logits [T,V], gates [T])
    logits, gates = model.inference(input_ids, sw_blocks, use_capped_logits=True, return_hidden=False)
    #print(gates)
    #breakpoint()
    if logits.ndim == 3: logits = logits[0]
    if gates.ndim  == 2: gates  = gates[0]

    VOCAB_SIZE = getattr(model.lm_head, "out_features", 50257)
    logits   = logits[:-1, :VOCAB_SIZE]               # [T-1,V]
    logprobs = torch.log_softmax(logits, dim=-1)      # [T-1,V]
    target   = input_ids[1:]                          # [T-1]
    gates_t  = gates[:-1]                             # [T-1]
    #breakpoint()
    cont_start = len(prompt_ids)
    cont_start_idx = pad_len + cont_start

    lp_slice   = logprobs[cont_start_idx - 1 :, :]                # [C,V]
    tgt_slice  = target  [cont_start_idx - 1 :]                   # [C]
    g_slice    = gates_t [cont_start_idx - 1 :]                   # [C]

    token_lps = lp_slice.gather(1, tgt_slice.unsqueeze(-1)).squeeze(-1)  # [C]

    sum_lp = token_lps.sum().item()
    avg_lp = sum_lp / max(1, tgt_slice.numel())

    if lambda_penalty is None:
        sum_lp_gated = (g_slice * token_lps).sum().item()
    else:
        sum_lp_gated = (g_slice * token_lps - (1.0 - g_slice) * lambda_penalty).sum().item()
    avg_lp_gated = sum_lp_gated / max(1, tgt_slice.numel())

    mean_gate = g_slice.mean().item() if g_slice.numel() else 1.0
    min_gate = g_slice.min().item() if g_slice.numel() else 1.0
    return {
        "sum_lp": sum_lp,
        "avg_lp": avg_lp,
        "sum_lp_gated": sum_lp_gated,
        "avg_lp_gated": avg_lp_gated,
        "mean_gate": mean_gate,
        'min_gate':min_gate
    }

@torch.no_grad()
def evaluate_hellaswag_abstain(
    model: GPT,
    n_examples: int = N_EVAL,
    gate_threshold: float = 0.5,
    lambda_penalty: float | None = None,
):
    """
    Metrics:
      - coverage = P(confident)
      - % confident & correct = (# confident & correct) / (# confident)      (precision on answered)
      - % correct abstentions = (# not confident & incorrect) / (# not confident)
      - confusion matrix counts:
            |           confident | not confident
        ----+---------------------+---------------
        correct     TP (cc)       |  FN? (cn)
        incorrect   FP (ic)       |  TN? (in)   <-- "correct abstentions"
    We choose the candidate with highest *gated* score (sum_lp_gated).
    Confidence is mean gate of the chosen candidate ≥ gate_threshold.
    """
    ds = load_dataset("hellaswag", split="validation")
    n = min(n_examples, len(ds))

    # counts
    cc = 0  # correct & confident
    ic = 0  # incorrect & confident
    cn = 0  # correct & not confident
    inn = 0 # incorrect & not confident  (TN; "correct abstentions")

    for i, ex in enumerate(ds):
        if i >= n: break
        ctx = (ex["ctx_a"] + " " + ex["ctx_b"]).strip()
        endings = ex["endings"]
        gold = int(ex["label"])

        # score each candidate
        sums_g, avgs_g, mean_gates, min_gates = [], [], [], []
        for cand in endings:
            s = score_continuation_abstain(model, ctx, cand, lambda_penalty=lambda_penalty)
            sums_g.append(s["sum_lp_gated"])
            avgs_g.append(s["avg_lp_gated"])
            mean_gates.append(s["mean_gate"])
            min_gates.append(s['min_gate'])

        # forced prediction (if we had to answer): argmax gated score
        forced_pred = int(max(range(4), key=lambda j: sums_g[j]))
        correct = (forced_pred == gold)

        # confidence = mean gate of chosen candidate
        confident = (min_gates[forced_pred] >= gate_threshold)

        if confident and correct:      cc += 1
        elif confident and not correct: ic += 1
        elif (not confident) and correct: cn += 1
        else:                           inn += 1

        if (i + 1) % 100 == 0:
            total = i + 1
            coverage = (cc + ic) / total if total else 0.0
            prec_on_answered = cc / (cc + ic) if (cc + ic) else 0.0
            correct_abstain  = inn / (cn + inn) if (cn + inn) else 0.0
            print(f"{total}/{n} | coverage={coverage*100:.2f}% "
                  f"| %conf&correct={prec_on_answered*100:.2f}% "
                  f"| %correct_abstain={correct_abstain*100:.2f}%")

    total = n
    coverage = (cc + ic) / total if total else 0.0
    percent_conf_correct = cc / (cc + ic) if (cc + ic) else 0.0
    percent_correct_abstentions = inn / (cn + inn) if (cn + inn) else 0.0

    print("\n=== Abstention Metrics ===")
    print(f"Coverage: {coverage*100:.2f}%")
    print(f"Percentage of confident and correct answers: {percent_conf_correct*100:.2f}%")
    print(f"Percentage of correct abstentions: {percent_correct_abstentions*100:.2f}%")
    print("\nConfusion matrix (counts):\n")
    print("                 | confident | not confident")
    print(f"correct    | {cc:<9d} | {cn:<14d}")
    print(f"incorrect  | {ic:<9d} | {inn:<14d}")

    # Also report “base accuracy if always answered” (forced accuracy)
    forced_acc = (cc + cn) / total if total else 0.0
    print(f"\nForced accuracy (answer always): {forced_acc*100:.2f}%")
    print(f"Random chance (HellaSwag): 25.00%")

if __name__ == "__main__":
    model = load_model(CHECKPOINT)
    # Match your labmate’s style: τ=0.5, gated utility without explicit lambda
    evaluate_hellaswag_abstain(
        model,
        n_examples=N_EVAL,
        gate_threshold=0.5,
        lambda_penalty=None,   # or set to your training lambda (e.g., 0.2) if you want
    )

