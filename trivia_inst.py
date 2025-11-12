import torch
import pandas as pd
from datasets import load_dataset
import tiktoken
from train_gpt_inst import GPT, next_multiple_of_n

# === Load SimpleQA Dataset ===
dataset = load_dataset("basicv8vc/SimpleQA", split="test")  # or "train" if you prefer
for a in range(1,21):
    # === Config ===
    device = "cuda"
    if a < 10:
        checkpoint_path = '/workspace/modded-nanogpt/logs/85a8d212-b6ff-4007-b702-14aaf40e9183/state_step00'+str(a)+'000.pt'
    
    else:
        checkpoint_path = '/workspace/modded-nanogpt/logs/85a8d212-b6ff-4007-b702-14aaf40e9183/state_step0'+str(a)+'000.pt'
    vocab_size = 50259   # 50257 + <ins> + <ctx>
    num_layers = 12
    num_heads = 6
    model_dim = 768
    max_seq_len = 24*1024
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
    def generate(model, prompt, max_new_tokens=100, temperature=0.8, top_k=200):
        input_ids = torch.tensor(encode(prompt), dtype=torch.int32, device=device)

        BLOCK_SIZE = 128
        PAD_TOKEN = 50256  # <|endoftext|>

        # --- Initial pad ---
        pad_len = (-len(input_ids)) % BLOCK_SIZE
        if pad_len > 0:
            pad = torch.full((pad_len,), PAD_TOKEN, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([pad, input_ids])

        sw_blocks = torch.tensor(model.lm_head.out_features // BLOCK_SIZE, dtype=torch.int32, device=device)

        for step in range(max_new_tokens):
            # --- Forward pass ---
            logits, = model.inference(input_ids, sw_blocks)
            if logits.ndim == 3:
                logits = logits[0]  # ensure shape [T, V]
            logits = logits[-1, :50259] / temperature

            # --- Top-k sampling ---
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            # Append new token
            input_ids = torch.cat((input_ids, next_id))

            # ✅ Re-pad to multiple of 128 every iteration
            pad_len = (-len(input_ids)) % BLOCK_SIZE
            if pad_len > 0:
                pad = torch.full((pad_len,), PAD_TOKEN, dtype=input_ids.dtype, device=input_ids.device)
                input_ids = torch.cat([pad, input_ids])

            if next_id.item() == PAD_TOKEN:
                break
        # --- Clean decode: remove all <|endoftext|> tokens ---
        output_ids = [tid for tid in input_ids.tolist() if tid != PAD_TOKEN]
        text = decode(output_ids).replace("<|endoftext|>", "").strip()
        return text
        #return decode(input_ids.tolist())


    # === Evaluation Loop ===
    results = []
    for i, ex in enumerate(dataset):
        question = ex.get("problem", "")
        answer = ex.get("answer", "")

        prompt = f"<ins>Answer the following question, do not diverge and make sure your answer is in correct format, if you are not sure you can say IDK.<ins><ctx>{question}<ctx>"
        #output = generate(model, prompt, max_new_tokens=50, temperature=0.8)
        output = generate(model, prompt, max_new_tokens=10, temperature=0.9)
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


