# vLLM Windows Wheel for RTX 5090 / CUDA 13.2

This repository provides a prebuilt Windows wheel for `vllm` built for an NVIDIA RTX 5090 Laptop GPU on Windows.

The goal is to make setup easier for Windows users who cannot directly use the standard upstream `vllm` Linux/CUDA wheels.

## Build summary

This wheel was built with:

- Windows x64
- Python 3.12
- CUDA Toolkit 13.2
- PyTorch 2.11.0+cu130
- vLLM `0.20.1.dev7+g377ffb268`
- NVIDIA RTX 5090 Laptop GPU
- Compute capability `sm_120`

Wheel filename:

```text
vllm-0.20.1.dev7+g377ffb268.cu132-cp312-cp312-win_amd64.whl
````

## Runtime requirements

You need:

* Windows x64
* Python 3.12
* NVIDIA GPU compatible with this build
* NVIDIA driver new enough to support CUDA 13.2
* `uv` or `pip`

Check your driver with:

```powershell
nvidia-smi
```

Your output should show:

```text
CUDA Version: 13.2
```

or newer.

The CUDA Toolkit is **not required** to install and run this wheel. The CUDA Toolkit is only required if you want to rebuild the wheel from source.

## Important compatibility note

This wheel was built using CUDA Toolkit 13.2.

If your NVIDIA driver only supports CUDA 13.1 or older, vLLM may install successfully but fail at runtime with an error similar to:

```text
CUDA error: the provided PTX was compiled with an unsupported toolchain
```

Fix: update your NVIDIA driver until `nvidia-smi` reports CUDA Version 13.2 or newer.

## Quick install with uv

Create a clean project folder:

```powershell
mkdir C:\vllm-test
cd C:\vllm-test
uv init
uv python pin 3.12
```

Install PyTorch CUDA 13.0 wheels and this vLLM wheel:

```powershell
uv add torch==2.11.0+cu130 torchaudio==2.11.0+cu130 torchvision==0.26.0+cu130 --index https://download.pytorch.org/whl/cu130
```

Then install the vLLM wheel from the GitHub release:

```powershell
uv pip install "https://github.com/leeyy0/vllm-rtx5090-release/releases/download/vllm-0.20.1.dev7-cu132-py312-win-amd64-rtx5090/vllm-0.20.1.dev7+g377ffb268.cu132-cp312-cp312-win_amd64.whl" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
```

## Quick install with pip

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip setuptools wheel
```

Install PyTorch:

```powershell
python -m pip install torch==2.11.0+cu130 torchaudio==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Install the vLLM wheel:

```powershell
python -m pip install "https://github.com/leeyy0/vllm-rtx5090-release/releases/download/vllm-0.20.1.dev7-cu132-py312-win-amd64-rtx5090/vllm-0.20.1.dev7+g377ffb268.cu132-cp312-cp312-win_amd64.whl" --extra-index-url https://download.pytorch.org/whl/cu130
```

## Basic validation (delete `uv run` if you had installed with pip)

Run:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import vllm; print(vllm.__version__); print(vllm.__file__)"
python -c "import vllm._C; print('vllm._C OK')"
```

Expected result:

```text
vllm._C OK
```

## Inference test

Create `test_vllm.py`:

```python
from vllm import LLM, SamplingParams

model = "Qwen/Qwen2.5-0.5B-Instruct"

llm = LLM(
    model=model,
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.5,
    disable_log_stats=True,
)

prompts = [
    "Write one short sentence about why GPUs are useful for AI.",
    "Give exactly three colors:",
    "The capital of France is",
]

sampling_params = SamplingParams(
    temperature=0.2,
    top_p=0.95,
    max_tokens=64,
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print("=" * 80)
    print("Prompt:", output.prompt)
    print("Output:", repr(output.outputs[0].text))
```

Run:

```powershell
uv run python .\test_vllm.py
```

or, with a normal venv:

```powershell
python .\test_vllm.py
```

## Using this wheel in another uv project

In your application project, install the wheel directly:

```powershell
uv add "https://github.com/leeyy0/vllm-rtx5090-release/releases/download/vllm-0.20.1.dev7-cu132-py312-win-amd64-rtx5090/vllm-0.20.1.dev7+g377ffb268.cu132-cp312-cp312-win_amd64.whl"
```

You may also need to make sure PyTorch is installed from the CUDA 13.0 PyTorch index:

```powershell
uv add torch==2.11.0+cu130 torchaudio==2.11.0+cu130 torchvision==0.26.0+cu130 --index https://download.pytorch.org/whl/cu130
```

Then lock your app environment:

```powershell
uv lock --python 3.12 --index-strategy unsafe-best-match
```

Commit both:

```text
pyproject.toml
uv.lock
```

## Generate a lock file for this release repo

From the root of this repository:

```powershell
uv lock --python 3.12 --index-strategy unsafe-best-match
```

This generates:

```text
uv.lock
```

To test a frozen install:

```powershell
uv sync --frozen
uv run python .\test_vllm.py
```

## Generate requirements-lock.txt

After `uv sync --frozen`, export a pip-compatible lock file:

```powershell
uv pip freeze > requirements-lock.txt
```

This can be useful for users who prefer `pip`, but `uv.lock` is the preferred reproducible lock file for uv users.

## Known warnings

You may see warnings such as:

```text
NIXL is not available
```

or Hugging Face cache symlink warnings on Windows. These are not necessarily fatal.

If you see:

```text
CUDA error: the provided PTX was compiled with an unsupported toolchain
```

update your NVIDIA driver until `nvidia-smi` reports CUDA Version 13.2 or newer.

## Rebuilding from source

Most users do not need to rebuild.

If rebuilding, you need:

* Visual Studio C++ build tools
* CUDA Toolkit 13.2
* CMake
* Ninja
* Python 3.12
* PyTorch 2.11.0+cu130

Before building, make sure the Visual Studio compiler environment is active and that `cl`, `nvcc`, `cmake`, and `ninja` are available:

```powershell
where.exe cl
where.exe nvcc
where.exe cmake
where.exe ninja
```

Required build environment variables:

```powershell
$env:DISTUTILS_USE_SDK = "1"
$env:VLLM_TARGET_DEVICE = "cuda"
$env:MAX_JOBS = "10"

$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:CUDA_HOME = $env:CUDA_PATH
$env:CUDA_ROOT = $env:CUDA_PATH
$env:Path = "$env:CUDA_PATH\bin;$env:Path"
```

Build wheel:

```powershell
python -m pip install build
python -m build --wheel --no-isolation
```

The wheel will be written to:

```text
dist/
```

## Disclaimer

This is an unofficial community build. It is not an official vLLM release.
