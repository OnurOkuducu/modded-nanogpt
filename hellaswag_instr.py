import math
import torch
from datasets import load_dataset
import tiktoken
from train_gpt_inst import GPT

# === Config (same as your inference) ===
device = "cuda"
vocab_size = 50259   # 50257 + <ins> + <ctx>
num_layers = 12
num_heads = 6
model_dim = 768
max_seq_len = 48*1024

# === Load model ===

# === Tokenizer ===
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
# === Helper: score log-likelihood of continuation ===
@torch.no_grad()
def score_continuation(model, prompt: str, continuation: str, device="cuda"):
    model.eval()

    # --- Tokenize ---
    prompt_ids = torch.tensor(encode(prompt), dtype=torch.int32, device=device)
    cont_ids   = torch.tensor(encode(" " + continuation), dtype=torch.int32, device=device)

    input_ids = torch.cat([prompt_ids, cont_ids])  # [T]
    VOCAB_SIZE = getattr(model.lm_head, "out_features", 50257)
    BLOCK_SIZE = 128
    PAD_TOKEN  = 50256  # <|endoftext|> used as pad in your generate()

    # --- Left-pad to multiple of BLOCK_SIZE (exactly like generate) ---
    pad_len = (-len(input_ids)) % BLOCK_SIZE
    if pad_len > 0:
        pad = torch.full((pad_len,), PAD_TOKEN, dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([pad, input_ids])  # [pad ...][prompt][cont]

    # --- Sliding-window blocks (mirror generate; 1+ safe-guard) ---
    sw_blocks = torch.tensor(max(1, len(input_ids) // BLOCK_SIZE), dtype=torch.int32, device=device)

    # --- Forward (use the model path, NOT embed→lm_head) ---
    # model(input_seq, target_seq, sliding_window_num_blocks) returns logits over time
    logits = model.inference(input_ids, sw_blocks)
    if logits.ndim == 3:  # [B, T, V] -> [T, V]
        logits = logits[0]

    # We need next-token logits for each position: use all except last position
    logits = logits[:-1, :VOCAB_SIZE]             # [T-1, V]
    logprobs = torch.log_softmax(logits, dim=-1)  # [T-1, V]

    # Targets are inputs shifted left by 1
    target = input_ids[1:]                        # [T-1]

    # Continuation region indices (account for left padding)
    cont_start = len(prompt_ids)                  # tokens after the prompt start
    cont_start_idx = pad_len + cont_start        # absolute index in padded sequence

    # Off-by-one: token at position i is predicted by logits[i-1]
    # We want to score tokens from cont_start_idx .. end-1, so slice from cont_start_idx-1
    lp_slice   = logprobs[cont_start_idx - 1 :, :]            # [C, V]
    tgt_slice  = target  [cont_start_idx - 1 :]               # [C]

    token_lps = lp_slice.gather(1, tgt_slice.unsqueeze(-1)).squeeze(-1)
    sum_lp = token_lps.sum().item()
    avg_lp = sum_lp / max(1, tgt_slice.numel())
    return sum_lp, avg_lp


# === Evaluate on HellaSwag ===
print("?~_~S~Z Loading HellaSwag dataset...")
ds = load_dataset("hellaswag", split="validation")

n = 400 #len(ds)
correct = 0
correct_norm = 0

for a in range(21):
    if a < 10:
        checkpoint_path = '/workspace/modded-nanogpt/logs/85a8d212-b6ff-4007-b702-14aaf40e9183/state_step00'+str(a)+'000.pt' 
    else:
        checkpoint_path = '/workspace/modded-nanogpt/logs/85a8d212-b6ff-4007-b702-14aaf40e9183/state_step0'+str(a)+'000.pt'
    print(checkpoint_path) 
    model = GPT(vocab_size, num_layers, num_heads, model_dim, max_seq_len).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["model"]

    # Handle compiled checkpoints
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("Detected compiled checkpoint — stripping '_orig_mod.' prefixes...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    model.eval()
    print("✅ Model loaded successfully!")
    mean_weight = torch.mean(torch.stack([p.float().mean() for p in model.parameters()]))
    print(f"Mean weight value for {checkpoint_path}: {mean_weight.item():.6f}")
    
    print("?~\~E Model loaded successfully for HellaSwag eval")
    correct = 0
    correct_norm = 0
   
    for i, ex in enumerate(ds):
        ctx = (ex["ctx_a"] + " " + ex["ctx_b"]).strip()
        endings = ex["endings"]
        gold = int(ex["label"])

        scores_sum = []
        scores_avg = []
        for cand in endings:
            s_sum, s_avg = score_continuation(model, ctx, cand)
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
    
    del model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # good for repeated loops
