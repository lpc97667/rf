# End-to-end smoke test: LayerCast vs RF inside vLLM.

import argparse
import os
import time

import torch

from patch_vllm import patch_qwen2_vllm, patch_llama_vllm, patch_qwen3_vllm
patch_qwen2_vllm()
patch_llama_vllm()
patch_qwen3_vllm()

import vllm
from vllm import SamplingParams

PROMPTS = [
    "Solve step by step: what is 17 * 23?",
    "Prove that the square root of 2 is irrational.",
    "Write a Python function that returns the nth Fibonacci number.",
    "A train travels 60 km in 45 minutes. What is its average speed in km/h?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    fused = os.environ.get("RF_FUSED", "1")
    print(f"# RF_FUSED={fused} model={args.model}")

    model = vllm.LLM(model=args.model, tensor_parallel_size=1,
                     max_model_len=4096, dtype="float32", enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=42)

    prompts = (PROMPTS * ((args.batch + len(PROMPTS) - 1) // len(PROMPTS)))[:args.batch]

    model.generate(prompts[:1], SamplingParams(temperature=0.0, max_tokens=8))

    t0 = time.perf_counter()
    outs = model.generate(prompts, sampling_params=sp)
    dt = time.perf_counter() - t0

    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"# generated {ntok} tokens in {dt:.2f}s -> {ntok / dt:.1f} tok/s "
          f"(batch={args.batch})")
    for i, o in enumerate(outs):
        ids = list(o.outputs[0].token_ids)
        print(f"prompt{i}: n={len(ids)} cumlogprob={o.outputs[0].cumulative_logprob}")
        print(f"prompt{i}_ids: {ids}")
        print(f"prompt{i}_text: {o.outputs[0].text[:120]!r}")


if __name__ == "__main__":
    main()
