import torch
import tiktoken
from train_gpt_inst import GPT, next_multiple_of_n  # your GPT definition

# === Config ===
device = "cuda"
checkpoint_path = "/workspace/modded-nanogpt/state_step001770.pt"
vocab_size = 50259   # 50257 + <ins> + <ctx>
num_layers = 12
num_heads = 6
model_dim = 768
max_seq_len = 6 * 2048

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
model.eval()
print("✅ Model loaded successfully!")

# === Undo the sigmoid scaling applied before lm_head ===
def uncap_logits(logits):
    logits = logits[..., :50257]
    logits = torch.logit(logits / 30, eps=1e-6) * 7.5
    return logits

# === Generation ===
@torch.no_grad()
def generate(model, prompt, max_new_tokens=100, temperature=0.8, top_k=200):
    input_ids = torch.tensor(encode(prompt), dtype=torch.int32, device=device)
    for _ in range(max_new_tokens):
        # Forward pass (use embeddings + lm_head)
        x = model.embed(input_ids)
        logits = model.lm_head(x)[-1, :]
        logits = uncap_logits(logits) / temperature

        # Top-k sampling
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_id))

        if next_id.item() == 50256:  # <|endoftext|>
            break

    return decode(input_ids.tolist())

# === Interactive loop ===
while True:
    prompt = input("\n🟢 Enter prompt (or 'quit'): ")
    if prompt.lower().strip() == "quit":
        break

    print("\n💬 Model output:\n")
    out = generate(model, prompt, max_new_tokens=100, temperature=0.8, top_k=200)
    print(out)
