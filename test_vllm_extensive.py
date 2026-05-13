import os
import sys
import time
import traceback

import torch
from vllm import LLM, SamplingParams


MODEL = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    started = time.time()

    section("1. Environment")

    print("Python:", sys.version)
    print("Torch:", torch.__version__)
    print("Torch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    assert_true(torch.cuda.is_available(), "torch.cuda.is_available() is False")

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print("GPU:", device_name)
    print("Compute capability:", capability)
    print(f"GPU memory: {total_mem_gb:.2f} GiB")

    section("2. vLLM native extension")

    import vllm
    import vllm._C

    print("vLLM package:", vllm.__file__)
    print("vLLM version:", getattr(vllm, "__version__", "unknown"))
    print("vllm._C import: OK")

    section("3. Load model")

    print("Model:", MODEL)

    llm = LLM(
        model=MODEL,
        dtype="float16",
        gpu_memory_utilization=0.50,
        max_model_len=2048,
        trust_remote_code=False,
    )

    print("Model loaded.")

    section("4. Single prompt generation")

    sampling = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=64,
    )

    prompt = "Write one short sentence about why GPUs are useful for AI."
    outputs = llm.generate([prompt], sampling)

    text = outputs[0].outputs[0].text
    print("Prompt:", prompt)
    print("Output:", repr(text))

    assert_true(len(text.strip()) > 0, "Single prompt produced empty output")

    section("5. Batched generation")

    prompts = [
        "Give exactly three colors:",
        "Give exactly three animals:",
        "Give exactly three programming languages:",
        "Give exactly three planets:",
        "Give exactly three fruits:",
        "Give exactly three countries:",
        "Give exactly three file extensions:",
        "Give exactly three GPU vendors:",
    ]

    batch_outputs = llm.generate(prompts, sampling)

    assert_true(len(batch_outputs) == len(prompts), "Batch output count mismatch")

    for i, out in enumerate(batch_outputs):
        generated = out.outputs[0].text.strip()
        print(f"[{i}] Prompt: {prompts[i]}")
        print(f"[{i}] Output: {generated!r}")
        assert_true(len(generated) > 0, f"Batch item {i} produced empty output")

    section("6. Deterministic-ish low-temperature generation")

    deterministic_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=32,
    )

    deterministic_prompt = "The capital of France is"
    det_outputs = llm.generate([deterministic_prompt], deterministic_sampling)
    det_text = det_outputs[0].outputs[0].text.strip()

    print("Prompt:", deterministic_prompt)
    print("Output:", repr(det_text))

    assert_true(len(det_text) > 0, "Deterministic prompt produced empty output")

    section("7. Longer context prompt")

    long_context = "\n".join(
        f"Line {i}: The test value is {i * 7}."
        for i in range(1, 101)
    )

    long_prompt = (
        long_context
        + "\n\nQuestion: What is the test value on line 42? Answer briefly."
    )

    long_outputs = llm.generate(
        [long_prompt],
        SamplingParams(temperature=0.0, max_tokens=64),
    )

    long_text = long_outputs[0].outputs[0].text.strip()
    print("Output:", repr(long_text))

    assert_true(len(long_text) > 0, "Long context prompt produced empty output")

    section("8. Multiple sampling params")

    creative_outputs = llm.generate(
        ["Give a tiny haiku about CUDA."],
        SamplingParams(
            temperature=0.8,
            top_p=0.9,
            max_tokens=64,
            n=2,
        ),
    )

    candidates = creative_outputs[0].outputs
    print("Number of candidates:", len(candidates))

    assert_true(len(candidates) == 2, "Expected n=2 candidates")

    for i, candidate in enumerate(candidates):
        print(f"Candidate {i}:", repr(candidate.text.strip()))
        assert_true(len(candidate.text.strip()) > 0, f"Candidate {i} is empty")

    section("9. GPU memory summary")

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3

    print(f"Allocated: {allocated:.2f} GiB")
    print(f"Reserved:  {reserved:.2f} GiB")

    elapsed = time.time() - started

    section("PASS")

    print(f"All tests passed in {elapsed:.1f} seconds.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("\nFAILED")
        traceback.print_exc()
        raise SystemExit(1)