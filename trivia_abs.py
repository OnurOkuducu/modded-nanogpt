import torch
import pandas as pd
from datasets import load_dataset
import tiktoken
from train_gpt_abs import GPT, next_multiple_of_n

# === Load SimpleQA Dataset ===
dataset = load_dataset("basicv8vc/SimpleQA", split="test")  # or "train" if you prefer
for a in range(1):
    # === Config ===
    device = "cuda"
    if a < 10:
        checkpoint_path =  "/workspace/modded-nanogpt/logs/4ca4ad16-ac74-42d3-b9bd-f1fd1651d20f/state_step005000.pt"   
    else:
        checkpoint_path = '/workspace/modded-nanogpt/logs/85a8d212-b6ff-4007-b702-14aaf40e9183/state_step0'+str(a)+'000.pt'
    vocab_size = 50259   # 50257 + <ins> + <ctx>
    num_layers = 12
    num_heads = 6
    model_dim = 768
    max_seq_len = 24*1024
    BLOCK_SIZE = 128
    PAD_TOKEN = 50256  # <|endoftext|>
    INS_ID = 50257
    CTX_ID = 50258

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
    '''
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    '''
    # === Load Model ===
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

    def pad_to_block(x_ids: torch.Tensor, pad_token: int = PAD_TOKEN, block: int = BLOCK_SIZE):
        pad_len = (-len(x_ids)) % block
        if pad_len > 0:
            pad = torch.full((pad_len,), pad_token, dtype=x_ids.dtype, device=x_ids.device)
            x_ids = torch.cat([pad, x_ids], dim=0)
        return x_ids

    # === Generation Function ===
    @torch.no_grad()
    def generate_multiple(
        model,
        prompt,
        n=5,                      # number of generations
        max_new_tokens=50,
        temperature=0.5,
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
                logits = logits[:50259]  # trim to vocab
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

    # === Evaluation Loop ===
    results = []
    for i, ex in enumerate(dataset):
        question = ex.get("problem", "")
        answer = ex.get("answer", "")

        prompt = f"<ins>Answer the following question:<ins><ctx>{question}<ctx>"
        #output = generate(model, prompt, max_new_tokens=50, temperature=0.8)
        output,_= generate_with_abstention(prompt, max_new_tokens=10, temperature=0.9)
        print(output)
        results.append({
            "question": question,
            "ground_truth": answer,
            "model_output": output,
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(dataset)} examples...")
        if i == 30:
            break
    # === Save to CSV ===
    df = pd.DataFrame(results)
    df.to_csv("simpleqa_eval_"+str(a)+"000.csv", index=False)
    print("✅ Saved results to simpleqa_eval_"+str(a)+"000.csv")


