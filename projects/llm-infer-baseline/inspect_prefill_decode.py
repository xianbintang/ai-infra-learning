"""详细观察 Prefill 和 Decode 过程的脚本（含耗时测量）。"""
import time
import torch
from config import MODEL_NAME, DEVICE, describe_environment
from model import load_model
from tokenizer import load_tokenizer

PROMPT = "Q: Explain why batching improves GPU utilization.\nA:"
MAX_NEW_TOKENS = 10  # 增加到 10 个，便于观察

def topk_tokens(tokenizer, logits_1d, k=5):
    """返回 top-k 候选 token。"""
    vals, idx = torch.topk(logits_1d, k)
    items = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        tok = tokenizer.decode([i])
        items.append((i, tok.replace("\n", "\\n"), v))
    return items

def main():
    print("=" * 60)
    print("Prefill vs Decode 详细观察（含耗时测量）")
    print("=" * 60)
    
    # ---------- 环境信息 ----------
    print("\n[ENVIRONMENT]")
    print("Environment:", describe_environment())
    print("Model:", MODEL_NAME)
    print("Device:", DEVICE)
    
    # ---------- 加载模型 ----------
    print("\n[LOADING]")
    load_start = time.perf_counter()
    model = load_model(MODEL_NAME)
    tok = load_tokenizer(MODEL_NAME)
    load_time = time.perf_counter() - load_start
    print(f"Model + Tokenizer 加载耗时: {load_time:.4f} s")
    
    # ---------- Tokenize ----------
    print("\n[INPUT]")
    tokenize_start = time.perf_counter()
    inputs = tok(PROMPT, return_tensors="pt").to(DEVICE)
    tokenize_time = time.perf_counter() - tokenize_start
    
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[-1]
    print(f"Prompt: {repr(PROMPT)}")
    print(f"Prompt token 数: {prompt_len}")
    print(f"input_ids.shape: {tuple(input_ids.shape)}")
    print(f"Tokenize 耗时: {tokenize_time*1000:.4f} ms")

    # ---------- Prefill ----------
    print("\n" + "=" * 60)
    print("[PREFILL PHASE]")
    print("=" * 60)
    
    prefill_start = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    prefill_time = time.perf_counter() - prefill_start

    logits = out.logits
    pkv = out.past_key_values

    print(f"\nPrefill 耗时: {prefill_time*1000:.4f} ms")
    print(f"Prefill 处理 token 数: {prompt_len}")
    print(f"Prefill 每 token 耗时: {prefill_time*1000/prompt_len:.4f} ms/token")
    
    print(f"\nlogits.shape: {tuple(logits.shape)} = [batch, seq_len, vocab]")
    print(f"past_key_values 层数: {len(pkv)}")
    k0, v0 = pkv[0]
    print(f"layer0 K.shape: {tuple(k0.shape)} = [batch, heads, seq_len, head_dim]")
    print(f"layer0 V.shape: {tuple(v0.shape)}")

    # 选择第一个生成的 token
    next_logits = logits[0, -1]
    print(f"\n[Prefill -> 选择下一个 token]")
    print(f"top-5 候选:")
    for i, t, v in topk_tokens(tok, next_logits, k=5):
        print(f"  ID {i:6d}  {t!r:12s}  logit={v:.4f}")

    next_id = torch.argmax(next_logits).view(1, 1)
    print(f"选中: ID {next_id.item()} -> {repr(tok.decode(next_id[0].tolist()))}")

    # ---------- Decode Loop ----------
    print("\n" + "=" * 60)
    print("[DECODE PHASE]")
    print("=" * 60)
    
    generated = [next_id]
    past = pkv
    decode_times = []

    for step in range(1, MAX_NEW_TOKENS):
        decode_start = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        decode_time = time.perf_counter() - decode_start
        decode_times.append(decode_time)

        logits = out.logits
        past = out.past_key_values
        next_logits = logits[0, -1]

        k0, v0 = past[0]
        current_seq_len = k0.shape[2]
        
        print(f"\n[Decode Step {step}]")
        print(f"  输入 token: ID {next_id.item()} -> {repr(tok.decode([next_id.item()]))}")
        print(f"  耗时: {decode_time*1000:.4f} ms")
        print(f"  KV Cache seq_len: {current_seq_len} (增长了 1)")
        print(f"  top-3 候选:")
        for i, t, v in topk_tokens(tok, next_logits, k=3):
            print(f"    ID {i:6d}  {t!r:12s}  logit={v:.4f}")

        next_id = torch.argmax(next_logits).view(1, 1)
        generated.append(next_id)
        print(f"  选中: ID {next_id.item()} -> {repr(tok.decode(next_id[0].tolist()))}")

    # ---------- 生成结果 ----------
    gen_ids = torch.cat(generated, dim=-1)[0].tolist()
    generated_text = tok.decode(gen_ids, skip_special_tokens=True)
    
    print("\n" + "=" * 60)
    print("[GENERATED RESULT]")
    print("=" * 60)
    print(f"生成的 token IDs: {gen_ids}")
    print(f"生成的文本: {repr(generated_text)}")

    # ---------- 性能统计 ----------
    total_decode_time = sum(decode_times)
    avg_decode_time = total_decode_time / len(decode_times) if decode_times else 0
    total_new_tokens = len(generated)
    total_generate_time = prefill_time + total_decode_time
    tokens_per_second = total_new_tokens / total_generate_time if total_generate_time > 0 else 0
    
    print("\n" + "=" * 60)
    print("[PERFORMANCE SUMMARY]")
    print("=" * 60)
    print(f"\n📊 耗时分解:")
    print(f"  Model 加载:     {load_time:.4f} s")
    print(f"  Tokenize:       {tokenize_time*1000:.4f} ms")
    print(f"  Prefill:        {prefill_time*1000:.4f} ms ({prompt_len} tokens)")
    print(f"  Decode 总计:    {total_decode_time*1000:.4f} ms ({len(decode_times)} steps)")
    print(f"  生成总耗时:     {total_generate_time*1000:.4f} ms")
    
    print(f"\n📈 性能指标:")
    print(f"  Prefill 每 token:  {prefill_time*1000/prompt_len:.4f} ms/token")
    print(f"  Decode 平均:       {avg_decode_time*1000:.4f} ms/token")
    print(f"  生成速度:          {tokens_per_second:.2f} tokens/s")
    
    print(f"\n📝 Decode 每步耗时:")
    for i, t in enumerate(decode_times, 1):
        print(f"  Step {i}: {t*1000:.4f} ms")
    
    print(f"\n💡 关键发现:")
    if avg_decode_time > 0 and prefill_time/prompt_len > 0:
        ratio = avg_decode_time / (prefill_time/prompt_len)
        print(f"  Decode 每步耗时 vs Prefill 每 token: {ratio:.2f}x")
    print(f"  KV Cache 最终大小: {k0.shape} (seq_len={k0.shape[2]})")
    
    # ---------- 回答关键问题 ----------
    print("\n" + "=" * 60)
    print("[回答: Prefill vs Decode 谁更慢?]")
    print("=" * 60)
    print(f"""
Prefill 阶段:
  - 处理 {prompt_len} 个 token，耗时 {prefill_time*1000:.2f} ms
  - 平均每 token: {prefill_time*1000/prompt_len:.2f} ms
  - 特点: 一次性计算，可并行，计算密集型

Decode 阶段:
  - 生成 {len(decode_times)} 个 token，耗时 {total_decode_time*1000:.2f} ms
  - 平均每 token: {avg_decode_time*1000:.2f} ms
  - 特点: 串行执行，依赖 KV Cache，内存带宽密集型

结论: 
  - 单个 token 来看，Decode 通常比 Prefill 慢（需要读取 KV Cache）
  - 但 Prefill 处理多个 token，总耗时可能更长
  - 长 prompt 场景：Prefill 是瓶颈
  - 长生成场景：Decode 是瓶颈
""")

if __name__ == "__main__":
    main()
