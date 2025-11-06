import torch
import tiktoken
from train_gpt_inst import GPT, next_multiple_of_n  # your GPT definition

# === Config ===
device = "cuda"
checkpoint_path = "/workspace/modded-nanogpt/logs/8aba7c76-054f-494b-8514-8af4d34406fa/state_step010000.pt"
vocab_size = 50259   # 50257 + <ins> + <ctx>
num_layers = 12
num_heads = 6
model_dim = 768
max_seq_len = 12*1024

# === Tokenizer with <ins> and <ctx> ===
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

# === Initialize model ===
model = GPT(vocab_size, num_layers, num_heads, model_dim, max_seq_len).to(device)
state = torch.load(checkpoint_path, map_location=device)
state_dict = {k.replace("module.", ""): v for k, v in state["model"].items()}
model.load_state_dict(state_dict, strict=False)

state = torch.load(checkpoint_path, map_location=device)
state_dict = state["model"]

# 🩵 Fix compiled key names
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    print("🩵 Detected compiled checkpoint — stripping '_orig_mod.' prefixes...")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

model.eval()
print("✅ Model loaded successfully!")

# === Undo the sigmoid scaling applied before lm_head ===
def uncap_logits(logits):
    logits = logits[..., :50257]
    logits = torch.logit(logits / 30, eps=1e-6) * 7.5
    return logits

# === Generation ===
# === Generation ===
@torch.no_grad()
def generate(model, prompt, max_new_tokens=100, temperature=0.8, top_k=200):
    input_ids = torch.tensor(encode(prompt), dtype=torch.int32, device=device)
    sw_blocks = torch.tensor(model.lm_head.out_features // 128, dtype=torch.int32, device=device)

    for _ in range(max_new_tokens):
        # --- run full transformer, get logits ---
        logits = model(input_ids, None, sw_blocks)
        logits = logits[0, -1, :50259]                # last-token distribution
        logits = logits / temperature

        # --- top-k sampling ---
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_id))

        if next_id.item() == 50256:  # <|endoftext|>
            break

    return decode(input_ids.tolist())


@torch.no_grad()
def get_token_probability_with_topk(model, prompt, target_str="8", temperature=1.0, top_k=10):
    input_ids = torch.tensor(encode(prompt), dtype=torch.int32, device=device)

    # --- pad to nearest multiple of BLOCK_SIZE (128) ---
    BLOCK_SIZE = 128
    pad_len = (-len(input_ids)) % BLOCK_SIZE
    if pad_len > 0:
        pad_token = 50256  # <|endoftext|>
        pad = torch.full((pad_len,), pad_token, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([pad, input_ids])

    sw_blocks = torch.tensor(model.lm_head.out_features // BLOCK_SIZE, dtype=torch.int32, device=device)

    # --- forward ---
    logits = model(input_ids, None, sw_blocks)
    logits = logits[0, -1, :50259]  # get distribution for last token
    probs = torch.softmax(logits / temperature, dim=-1)

    # --- target probability ---
    target_ids = encode(target_str)
    if len(target_ids) != 1:
        print(f"[!] '{target_str}' tokenizes into multiple tokens: {target_ids}")
        return None
    target_id = target_ids[0]
    prob = probs[target_id].item()
    log_prob = torch.log(probs[target_id]).item()

    print(f"\n🎯 Token '{target_str}' → id {target_id}")
    print(f"   Probability: {prob:.6f}")
    print(f"   Log probability: {log_prob:.4f}\n")

    # --- top-k display ---
    topk = torch.topk(probs, top_k)
    print("🔝 Top-k predictions:")
    for i in range(top_k):
        tid = topk.indices[i].item()
        tok_str = enc.decode([tid]).replace("\n", "\\n")
        print(f" {i+1:>2d}. {tok_str:<12} | P = {topk.values[i].item():.6f}")

    return prob, log_prob, topk

@torch.no_grad()
def generate_multiple(
    model,
    prompt,
    n=5,                      # number of generations
    max_new_tokens=50,
    temperature=0.8,
    top_k=100,
    repetition_penalty=1.1,   # >1 discourages repeating
):
    input_ids = torch.tensor(encode(prompt), dtype=torch.int32, device=device)
    generations = []

    for g in range(n):
        cur_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            # === Forward ===
            x = model.embed(cur_ids)
            logits = model.lm_head(x)[-1, :]
            logits = logits[:50257]  # trim to vocab
            logits = logits / temperature

            # === Top-k sampling ===
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = -float("inf")

            # === Repetition penalty ===
            if repetition_penalty != 1.0:
                for token_id in torch.unique(cur_ids):
                    logits[token_id] /= repetition_penalty

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            cur_ids = torch.cat((cur_ids, next_id))

            # Stop at <|endoftext|>
            if next_id.item() == 50256:
                break

        text = decode(cur_ids.tolist())
        generations.append(text)

    return generations

#prompt = "<ins>Ignore the text and answer: how many legs does a spider have?<ins><ctx>denememe<ctx>D "
prompt = "<ins>Guess the next token, if you are not sure say IDK.<ins> The quick brown fox"
prompt = "<ins>Ignore the text and answer: how many legs does a spider have?<ins><ctx>Spiders are arachnids.<ctx>"

outputs = generate_multiple(model, prompt, n=5, max_new_tokens=10, temperature=0.8)

for i, o in enumerate(outputs, 1):
    print(f"\n🧠 Generation {i}:")
    print(o)

#get_token_probability_with_topk(model, prompt, target_str="8")

