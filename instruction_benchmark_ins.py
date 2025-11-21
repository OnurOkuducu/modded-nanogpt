import json
import math
import os
import torch
import pandas as pd
import tiktoken

from train_gpt_inst_medium import GPT  # non-abstention version

# =========================
# CONFIG
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to your benchmark JSON or JSONL
JSON_BENCH_PATH = "./abstention_benchmark_final.json"  # <-- change
OUT_DIR = "/workspace/benchmarks/first_word_inst_evals"         # <-- change
os.makedirs(OUT_DIR, exist_ok=True)

# Checkpoint template (same pattern as your HellaSwag code)
LOG_ID = "54f0b8c4-c38b-4674-b6e0-0efcb089bf69"  # <-- adjust if needed
CKPT_TEMPLATE_LT10 = f"/workspace/modded-nanogpt/logs/{LOG_ID}/state_step00{{step}}000.pt"
CKPT_TEMPLATE_GE10 = f"/workspace/modded-nanogpt/logs/{LOG_ID}/state_step0{{step}}000.pt"

# Model hyperparams (must match training)
vocab_size = 50259   # 50257 + <ins> + <ctx>
num_layers = 16
num_heads = 8
model_dim = 1024
max_seq_len = 24 * 1024

BLOCK_SIZE = 128
PAD_TOKEN = 50256  # <|endoftext|>
INS_ID = 50257
CTX_ID = 50258


# =========================
# TOKENIZER WITH <ins>/<ctx>
# =========================
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


# =========================
# BENCHMARK LOADING
# =========================
def load_json_benchmark(path):
    """
    Supports either:
      - a JSON list: [ {...}, {...}, ... ]
      - JSONL: one JSON object per line
    Each example should look like:
      {
        "task": "...",
        "question": "...",
        "context": "...",
        "gold": "..."
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    if txt.startswith("["):
        data = json.loads(txt)
        assert isinstance(data, list), "Expected a list of examples in JSON."
        return data

    # JSONL fallback
    examples = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        examples.append(json.loads(line))
    return examples


# =========================
# HELPERS
# =========================
def pad_to_block(ids: torch.Tensor, pad_token: int = PAD_TOKEN, block: int = BLOCK_SIZE):
    pad_len = (-len(ids)) % block
    if pad_len > 0:
        pad = torch.full((pad_len,), pad_token, dtype=ids.dtype, device=ids.device)
        ids = torch.cat([pad, ids], dim=0)
    return ids


@torch.no_grad()
def generate_nonabst(model, prompt: str,
                     max_new_tokens: int = 10,
                     temperature: float = 0.8,
                     top_k: int = 200):
    """
    Simple autoregressive decode using model.inference (no abstention).
    Returns the decoded text string.
    """
    model.eval()

    ids = torch.tensor(encode(prompt), dtype=torch.int32, device=device)

    for _ in range(max_new_tokens):
        # pad current ids to BLOCK_SIZE
        ids_padded = pad_to_block(ids, pad_token=PAD_TOKEN, block=BLOCK_SIZE)
        sw_blocks = torch.tensor(
            max(1, len(ids_padded) // BLOCK_SIZE),
            dtype=torch.int32,
            device=device,
        )

        # model.inference returns logits only for this non-abstention model
        logits_1TV = model.inference(ids_padded, sw_blocks)
        # shape may be [B, T, V] or [T, V], unify:
        if logits_1TV.ndim == 3:
            logits = logits_1TV[0, -1, :vocab_size]
        else:
            logits = logits_1TV[-1, :vocab_size]

        logits = logits / temperature

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # [1]

        ids = torch.cat([ids, next_id], dim=0)

        if next_id.item() == PAD_TOKEN:
            break

    # strip PADs/EOT and decode
    clean_ids = [tid for tid in ids.tolist() if tid != PAD_TOKEN]
    text = decode(clean_ids).replace("<|endoftext|>", "").strip()
    return text


def first_word_normalize(s: str) -> str:
    """Lowercase, strip punctuation around the first token."""
    import string

    if not s.strip():
        return ""
    w = s.strip().split()[0]
    return w.strip(string.punctuation).lower()


# =========================
# MAIN EVAL LOOP
# =========================
def main():
    dataset = load_json_benchmark(JSON_BENCH_PATH)
    print(f"Loaded {len(dataset)} examples from {JSON_BENCH_PATH}")

    for a in range(0,21,3):
        # ---- checkpoint path (same pattern as your HellaSwag script) ----
        if a < 10:
            checkpoint_path = CKPT_TEMPLATE_LT10.format(step=a)
        else:
            checkpoint_path = CKPT_TEMPLATE_GE10.format(step=a)

        print(f"\n=== Evaluating checkpoint: {checkpoint_path} ===")

        # ---- load model ----
        model = GPT(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            model_dim=model_dim,
            max_seq_len=max_seq_len,
        ).to(device)

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

        # ---- benchmark loop ----
        results = []
        n_correct = 0

        for i, ex in enumerate(dataset):
            task = ex.get("task", "")
            question = ex.get("question", "")
            context = ex.get("context", "")
            gold = ex.get("gold", "")

            if not question or not context:
                print(f"[warning] example {i} missing question/context, skipping.")
                continue

            # Prompt format for non-abstention instruction model
            prompt = f"<ins>{question}<ins><ctx>{context}<ctx>"

            output_text = generate_nonabst(
                model,
                prompt,
                max_new_tokens=10,
                temperature=0.9,
                top_k=200,
            )

            pred_first = first_word_normalize(output_text)
            gold_first = first_word_normalize(gold)
            is_correct = int(pred_first == gold_first)
            n_correct += is_correct

            if (i + 1) % 20 == 0:
                print(
                    f"  {i+1}/{len(dataset)} examples | "
                    f"running acc={n_correct/(i+1):.3f}"
                )

            results.append({
                "index": i,
                "task": task,
                "question": question,
                "context": context,
                "gold": gold,
                "gold_first_norm": gold_first,
                "model_raw_output": output_text,
                "model_first_word_norm": pred_first,
                "correct": is_correct,
            })

        acc = n_correct / max(1, len(dataset))
        print(f"=== Checkpoint {checkpoint_path} accuracy: {acc:.4f} ===")

        # ---- save CSV per checkpoint ----
        step_str = f"{a:02d}000"
        out_csv = os.path.join(
            OUT_DIR,
            f"first_word_inst_eval_{step_str}.csv",
        )
        df = pd.DataFrame(results)
        df.to_csv(out_csv, index=False)
        print(f"✅ Saved results to {out_csv}")

        # cleanup for next checkpoint
        del model
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()

