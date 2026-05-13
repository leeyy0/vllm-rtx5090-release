# vLLM Windows build for RTX 5090 / Blackwell

This repo provides a Windows wheel for `vllm-windows`, built and tested locally on an NVIDIA RTX 5090 / Blackwell GPU.

This is a community Windows build. It is not an official vLLM release.

## Tested environment

- OS: Windows x64
- GPU: NVIDIA GeForce RTX 5090 / Blackwell
- NVIDIA driver: 591.94
- Driver-reported CUDA support: 13.1, shown by `nvidia-smi`
- CUDA Toolkit used for build: 13.2, shown by `nvcc --version`
- Python: 3.12
- PyTorch: 2.11.0+cu130
- vLLM: 0.20.1.dev7+g377ffb268.cu132
- Compiler: Visual Studio 2026 Community, Desktop development with C++

Note: `nvidia-smi` reports the CUDA version supported by the installed NVIDIA driver. It is not the same thing as the installed CUDA Toolkit version. This wheel was built using CUDA Toolkit 13.2.

## Files

The main artifact is:

```text
wheels/vllm-0.20.1.dev7+g377ffb268.cu132-cp312-cp312-win_amd64.whl
```

This wheel is for:

```text
Python 3.12
Windows x64
CUDA/PyTorch cu130 runtime stack
```

## Prerequisites

Install:

1. NVIDIA driver new enough for RTX 5090 / Blackwell
2. Python 3.12
3. Visual C++ Redistributable, or Visual Studio Build Tools / Visual Studio Community
4. `uv`, recommended

Check GPU driver:

```powershell
nvidia-smi
```

Check Python:

```powershell
python --version
```

Expected:

```text
Python 3.12.x
```

## Install with uv

Clone this repo:

```powershell
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

Create the environment:

```powershell
uv sync
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the local vLLM wheel if it was not installed by `uv sync`:

```powershell
uv pip install .\wheels\vllm-0.20.1.dev7+g377ffb268.cu132-cp312-cp312-win_amd64.whl --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
```

## Verify install

Check PyTorch CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Expected output should look like:

```text
2.11.0+cu130
13.0
True
```

Check vLLM native extension:

```powershell
python -c "import vllm._C; print('vllm._C OK')"
```

Expected:

```text
vllm._C OK
```

## Minimal inference test

```powershell
python -c "from vllm import LLM; llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct', dtype='float16', gpu_memory_utilization=0.5); print(llm.generate(['hi'])[0].outputs[0].text)"
```

The first run may take time because the model needs to download.

## Known limitations

- This is a community Windows build.
- This wheel is tested only on the environment listed above.
- Other GPUs, drivers, Python versions, CUDA versions, or Torch versions may not work.
- If `import vllm._C` fails, the native extension or DLL runtime is not loading correctly.
- If `torch.cuda.is_available()` is `False`, PyTorch CUDA is not installed correctly.

## Build notes

This wheel was built from source because prebuilt wheels did not load correctly on the RTX 5090 / Blackwell Windows setup.

The original failure was:

```text
ImportError: DLL load failed while importing _C
```

Building locally aligned the vLLM native CUDA/C++ extension with:

```text
local Python 3.12
local PyTorch cu130
local CUDA Toolkit 13.2
local MSVC compiler
local RTX 5090 / Blackwell GPU
```

## Rebuilding from source

Enable Git long paths:

```powershell
git config --global core.longpaths true
```

Enable Windows long paths in Administrator PowerShell:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Open a fresh PowerShell and verify:

```powershell
(Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem").LongPathsEnabled
git config --global --get core.longpaths
```

Expected:

```text
1
true
```

Load the Visual Studio compiler environment:

```powershell
cmd /c "`"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`" && set" | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
    }
}
```

Set build variables:

```powershell
$env:DISTUTILS_USE_SDK = "1"
$env:VLLM_TARGET_DEVICE = "cuda"
$env:MAX_JOBS = "12"

$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:CUDA_HOME = $env:CUDA_PATH
$env:CUDA_ROOT = $env:CUDA_PATH
$env:Path = "$env:CUDA_PATH\bin;$env:Path"
```

Verify tools:

```powershell
where.exe cl
where.exe nvcc
where.exe cmake
where.exe ninja
```

Build wheel:

```powershell
python -m pip install build
python -m build --wheel --no-isolation
```

The wheel will be created in:

```text
dist/
```
