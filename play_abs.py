import torch
import tiktoken
from train_gpt_abs import GPT, next_multiple_of_n  # your GPT with abstention head

# === Config ===
device = "cuda"
checkpoint_path = '/workspace/modded-nanogpt/logs/25b4c2e5-4878-474f-a82a-e58668a186cf/state_step005000.pt' 
vocab_size = 50259   # 50257 + <ins>=50257 + <ctx>=50258
num_layers = 12
num_heads = 6
model_dim = 768
max_seq_len = 48 * 1024
BLOCK_SIZE = 128
PAD_TOKEN = 50256  # <|endoftext|>
INS_ID = 50257
CTX_ID = 50258

# === Tokenizer with <ins> and <ctx> ===
base_enc = tiktoken.get_encoding("gpt2")
custom_specials = {"<ins>": INS_ID, "<ctx>": CTX_ID}
enc = tiktoken.Encoding(
    name="gpt2-with-ins-ctx",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens={**base_enc._special_tokens, **custom_specials},
)
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>", "<ins>", "<ctx>"})
decode = lambda l: enc.decode(l)

# === Initialize model ===
model = GPT(vocab_size, num_layers, num_heads, model_dim, max_seq_len).to(device)

state = torch.load(checkpoint_path, map_location=device)
state_dict = state["model"]
# Fix compiled ckpt key names if present
if any(k.startswith("_orig_mod.") for k in state_dict):
    print("Detected compiled checkpoint — stripping '_orig_mod.' prefixes...")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
model.eval()
print("✅ Model loaded successfully!")

# === If you trained with capped logits and want raw logits back ===
def uncap_logits(logits_1T_V):
    # expects logits in capped space (after 30*sigmoid(. / 7.5))
    # returns uncapped logits roughly matching pre-cap values
    logits_1T_V = logits_1T_V[..., :50257]
    return torch.logit(logits_1T_V / 30, eps=1e-6) * 7.5

def pad_to_block(x_ids: torch.Tensor, pad_token: int = PAD_TOKEN, block: int = BLOCK_SIZE):
    pad_len = (-len(x_ids)) % block
    if pad_len > 0:
        pad = torch.full((pad_len,), pad_token, dtype=x_ids.dtype, device=x_ids.device)
        x_ids = torch.cat([pad, x_ids], dim=0)
    return x_ids

@torch.no_grad()
def run_inference_once(prompt: str, use_capped_logits=True):
    """
    Returns:
      logits: [T, V] (trimmed batch dim)
      gates:  [T]
      tokens: [T] (after padding)
    """
    tokens = torch.tensor(encode(prompt), dtype=torch.int32, device=device)
    tokens = pad_to_block(tokens)
    sw_blocks = torch.tensor(model.lm_head.out_features // BLOCK_SIZE, dtype=torch.int32, device=device)

    logits_1TV, gates_T = model.inference(tokens, sw_blocks, use_capped_logits=use_capped_logits)
    logits_TV = logits_1TV[0]  # drop batch dim

    return logits_TV, gates_T, tokens

@torch.no_grad()
def generate_with_abstention(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    use_capped_logits: bool = True,
):
    """
    Autoregressive decode. For each step:
      - get next-token distribution
      - get per-token abstention scores (for the whole current context)
    Returns:
      text, history where history is a list of dicts with:
        {
          "step": i,
          "next_id": int,
          "next_prob_topk": list[(tok_id, prob)],
          "gate_last": float,          # gate for the last token in context
          "gate_ctx_closers": list[(pos, gate_value)]  # gates on </ctx> positions seen so far
        }
    """
    ids = torch.tensor(encode(prompt), dtype=torch.int32, device=device)

    history = []
    for i in range(max_new_tokens):
        ids = pad_to_block(ids)
        sw_blocks = torch.tensor(model.lm_head.out_features // BLOCK_SIZE, dtype=torch.int32, device=device)

        logits_1TV, gates_T = model.inference(ids, sw_blocks, use_capped_logits=use_capped_logits)
        logits = logits_1TV[0, -1, :50259] / temperature  # last-token distribution

        # top-k
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        # record some useful info
        topk_vals, topk_idx = torch.topk(probs, k=min(10, probs.numel()))
        gate_last = gates_T[-1].item()
        # grab any </ctx> gates in the current context (optional)
        ctx_positions = (ids == CTX_ID).nonzero().flatten()
        ctx_gates = [(int(pos.item()), float(gates_T[int(pos)].item())) for pos in ctx_positions]

        history.append({
            "step": i,
            "next_id": int(next_id.item()),
            "next_prob_topk": [(int(topk_idx[j]), float(topk_vals[j])) for j in range(topk_vals.numel())],
            "gate_last": gate_last,
            "gate_ctx_closers": ctx_gates,
        })

        ids = torch.cat([ids, next_id], dim=0)

        if next_id.item() == PAD_TOKEN:
            break

    # strip PADs for decode
    clean_ids = [tid for tid in ids.tolist() if tid != PAD_TOKEN]
    text = decode(clean_ids).replace("<|endoftext|>", "").strip()
    return text, history

# === Example usage ===
if __name__ == "__main__":
    prompt = '<ins>Ignore the text and answer: how many legs does a spider have?<ins><ctx>Spiders are arachnids.<ctx>'
    '''
    # Single pass: per-token logits and gates for the given prompt
    logits_TV, gates_T, tokens_T = run_inference_once(prompt, use_capped_logits=True)
    print(f"logits shape: {tuple(logits_TV.shape)}, gates shape: {tuple(gates_T.shape)}")
    # Example: gates at </ctx> positions
    ctx_pos = (tokens_T == CTX_ID).nonzero().flatten().tolist()
    print("ctx positions & gates:", [(p, float(gates_T[p])) for p in ctx_pos])
    '''
    # Autoregressive generation with both outputs
    out_text, steps = generate_with_abstention(prompt, max_new_tokens=40, temperature=0.8, top_k=200)
    print("\n=== Generation ===")
    print(out_text)
    print("\n=== Steps (last 3) ===")
    for s in steps[-3:]:
        print(s)

