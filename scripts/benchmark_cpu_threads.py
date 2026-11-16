"""
CPU Thread Benchmarking Script for AfriqueQwen.
Evaluates n_threads in [4, 6, 8] across multiple trials.
Measures:
- Model initialization time
- Prompt evaluation time & speed (tokens/sec)
- Generation time & speed (tokens/sec)
- Total latency
- CPU utilization and RAM usage
"""

import sys
import time
from pathlib import Path
import psutil

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.afrique_qwen import AfriqueQwenLLM
from src.llm.prompts import SYSTEM_PROMPT


def get_mem_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)


def benchmark_config(n_threads: int, trials: int = 2):
    print(f"\n=======================================================")
    print(f"BENCHMARKING CONFIGURATION: n_threads = {n_threads} ({trials} trials)")
    print(f"=======================================================")

    results = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Muraho neza! Amakuru ki?"},
    ]

    for trial in range(1, trials + 1):
        print(f"\n--- Trial {trial}/{trials} (threads={n_threads}) ---")
        mem_before = get_mem_mb()

        # 1. Model Initialization Time
        t_init_start = time.time()
        llm = AfriqueQwenLLM(n_ctx=4096, n_threads=n_threads, max_tokens=64)
        model = llm.load_model()
        t_init = time.time() - t_init_start
        mem_after_load = get_mem_mb()

        prompt_tokens = llm.count_tokens(messages)

        # 2. Prompt Evaluation & Token Generation Timing via Streaming
        cpu_samples = []
        psutil.cpu_percent(interval=None)  # prime cpu monitor

        t_start = time.time()
        response_stream = model.create_chat_completion(
            messages=messages,
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            stream=True,
        )

        first_token_time = None
        tokens_generated = 0
        chunks_text = []

        for chunk in response_stream:
            cpu_samples.append(psutil.cpu_percent(interval=None))
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    chunks_text.append(content)
                    tokens_generated += 1

        t_end = time.time()
        mem_peak = get_mem_mb()

        prompt_eval_time = (first_token_time - t_start) if first_token_time else 0.0
        generation_time = (t_end - first_token_time) if first_token_time else 0.0
        total_time = t_end - t_start
        prompt_tps = prompt_tokens / prompt_eval_time if prompt_eval_time > 0 else 0.0
        gen_tps = tokens_generated / generation_time if generation_time > 0 else 0.0
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0

        print(f"Init Time:         {t_init:.2f}s")
        print(f"Prompt Tokens:     {prompt_tokens}")
        print(f"Prompt Eval Time:  {prompt_eval_time:.2f}s ({prompt_tps:.2f} t/s)")
        print(f"Generated Tokens:  {tokens_generated}")
        print(f"Generation Time:   {generation_time:.2f}s ({gen_tps:.2f} t/s)")
        print(f"Total Infer Time:  {total_time:.2f}s")
        print(f"Avg CPU %:         {avg_cpu:.1f}%")
        print(f"RAM Loaded / Peak: {mem_after_load:.1f} MB / {mem_peak:.1f} MB")

        results.append({
            "t_init": t_init,
            "prompt_tokens": prompt_tokens,
            "prompt_eval_time": prompt_eval_time,
            "prompt_tps": prompt_tps,
            "gen_tokens": tokens_generated,
            "gen_time": generation_time,
            "gen_tps": gen_tps,
            "total_time": total_time,
            "avg_cpu": avg_cpu,
            "mem_load": mem_after_load,
            "mem_peak": mem_peak,
        })

        # Explicitly unload model to reset for next trial
        del model
        del llm._llm

    # Averages
    avg = {
        "n_threads": n_threads,
        "t_init": sum(r["t_init"] for r in results) / len(results),
        "prompt_eval_time": sum(r["prompt_eval_time"] for r in results) / len(results),
        "prompt_tps": sum(r["prompt_tps"] for r in results) / len(results),
        "gen_tokens": sum(r["gen_tokens"] for r in results) / len(results),
        "gen_time": sum(r["gen_time"] for r in results) / len(results),
        "gen_tps": sum(r["gen_tps"] for r in results) / len(results),
        "total_time": sum(r["total_time"] for r in results) / len(results),
        "avg_cpu": sum(r["avg_cpu"] for r in results) / len(results),
        "mem_load": sum(r["mem_load"] for r in results) / len(results),
        "mem_peak": sum(r["mem_peak"] for r in results) / len(results),
    }

    print(f"\n>>> SUMMARY FOR n_threads = {n_threads}:")
    print(f"Avg Init Time:        {avg['t_init']:.2f}s")
    print(f"Avg Prompt Eval:      {avg['prompt_eval_time']:.2f}s ({avg['prompt_tps']:.2f} t/s)")
    print(f"Avg Gen Speed:        {avg['gen_tps']:.2f} t/s")
    print(f"Avg Total Infer Time: {avg['total_time']:.2f}s")
    print(f"Avg CPU:              {avg['avg_cpu']:.1f}%")
    print(f"Avg RAM Peak:         {avg['mem_peak']:.1f} MB")
    return avg


def main():
    print("Starting CPU Thread Benchmarking (4, 6, 8 threads)...")
    all_summaries = []
    for th in [4, 6, 8]:
        summary = benchmark_config(n_threads=th, trials=2)
        all_summaries.append(summary)

    print("\n" + "=" * 75)
    print("FINAL COMPARISON TABLE")
    print("=" * 75)
    print(f"{'Threads':<8}{'Init(s)':<10}{'Prompt(t/s)':<14}{'Gen(t/s)':<12}{'Total(s)':<12}{'CPU(%)':<10}{'RAM(MB)':<10}")
    print("-" * 75)
    for s in all_summaries:
        print(f"{s['n_threads']:<8}{s['t_init']:<10.2f}{s['prompt_tps']:<14.2f}{s['gen_tps']:<12.2f}{s['total_time']:<12.2f}{s['avg_cpu']:<10.1f}{s['mem_peak']:<10.1f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
