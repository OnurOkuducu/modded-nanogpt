import torch
import tiktoken
from train_gpt import GPT, next_multiple_of_n  # import your definitions directly

# === Config ===
device = "cuda"
checkpoint_path = "/workspace/modded-nanogpt/state_step001770.pt"
vocab_size = 50257
num_layers = 12
num_heads = 6
model_dim = 768
max_seq_len = 6 * 2048  # match your training args
sliding_window_num_blocks = torch.tensor(max_seq_len // 128, dtype=torch.int32, device=device)

# === Initialize model ===
model = GPT(vocab_size, num_layers, num_heads, model_dim, max_seq_len).to(device)
state = torch.load(checkpoint_path, map_location=device)
state_dict = state["model"]

# Remove possible prefixes
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.eval()

print("?~\~E Model loaded successfully!")

# === Tokenizer ===
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

prompt = "tell me a story"
ids = torch.tensor(enc.encode(prompt), dtype=torch.int32, device="cuda")

temperature = 0.8

with torch.no_grad():
    emb = model.embed(ids)
    logits = model.lm_head(emb)

    # === FIX: align with vocab + undo sigmoid scaling ===
    logits = logits[..., :50257]                # trim unused outputs
    logits = torch.logit(logits / 30, eps=1e-6) * 7.5  # reverse the sigmoid cap
    probs = torch.softmax(logits / temperature, dim=-1)

    topk = torch.topk(logits[-1], 10)
    print("Top tokens:", [enc.decode([i]) for i in topk.indices.tolist()])

# === Generation ===
@torch.no_grad()
def generate_full(model, prompt, max_new_tokens=100, temperature=0.8, top_k=200):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    input_ids = torch.tensor(encode(prompt), dtype=torch.int32, device="cuda")

    for _ in range(max_new_tokens):
        x = input_ids
        target_seq = x.clone()  # dummy targets

        # --- pad to block size ---
        BLOCK_SIZE = 128
        pad_len = (-len(x)) % BLOCK_SIZE
        if pad_len > 0:
            x = torch.cat([torch.full((pad_len,), 50256, dtype=x.dtype, device=x.device), x])

        # --- use model's full forward path ---
        sliding_window_num_blocks = torch.tensor(model.lm_head.out_features // 128, dtype=torch.int32, device="cuda")
        long_bm, short_bm = model.create_block_masks(x, sliding_window_num_blocks)
        ve = model.value_embeds(x)
        x0 = torch.nn.functional.rms_norm(model.embed(x)[None], (model.embed.embedding_dim,))
        ve_enc = ve[:model.num_encoder_layers]
        ve_dec = ve[model.num_encoder_layers:]

        skip_connections = []
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm]
        for i in range(model.num_encoder_layers):
            x0 = model.blocks[i](x0, ve_enc[i], x0, block_masks[i])
            skip_connections.append(x0)
        block_masks.reverse()
        for i in range(model.num_decoder_layers):
            x0 = x0 + model.skip_weights[i] * skip_connections.pop()
            x0 = model.blocks[model.num_encoder_layers + i](x0, ve_dec[i], x0, block_masks[i])

        # --- logits through LM head ---
        logits = model.lm_head(torch.nn.functional.rms_norm(x0, (x0.size(-1),)))[0, -1, :]

        # --- fix scaling ---
        logits = logits[:50257]
        logits = torch.logit(logits / 30, eps=1e-6) * 7.5
        logits = logits / temperature

        # --- top-k sampling ---
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids, next_id))
        if next_id.item() == 50256:
            break

    return decode(input_ids.tolist())

@torch.no_grad()
def generate(model, prompt, max_new_tokens=50, temperature=0.8, top_k=200):
    input_ids = torch.tensor(encode(prompt), dtype=torch.int32, device=device)
    for _ in range(max_new_tokens):
        x = model.embed(input_ids)
        logits = model.lm_head(x)[-1, :]

        # === FIX: trim + undo sigmoid + sample ===
        logits = logits[:50257]
        logits = torch.logit(logits / 30, eps=1e-6) * 7.5
        logits = logits / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_id))
        if next_id.item() == 50256:  # <|endoftext|>
            break
    return decode(input_ids.tolist())

# === Run interactive loop ===
while True:
    prompt = input("\n?~_~S~] Enter prompt (or 'quit'): ")
    if prompt.lower() == "quit":
        break
    print("\n?~_?|  Model output:\n")
    print(generate_full(model, prompt, max_new_tokens=100, temperature=0.8, top_k=200))

