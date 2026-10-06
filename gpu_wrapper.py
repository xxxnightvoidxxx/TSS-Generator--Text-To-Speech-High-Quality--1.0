"""
gpu_wrapper.py
Ensures main.py runs with GPU inference via onnxruntime-gpu.

Fixes the classic "8 CPU threads even though I installed onnxruntime-gpu" problem:
  1. Detects and removes the CPU-only `onnxruntime` package (it shadows the GPU one)
  2. Verifies CUDAExecutionProvider is available
  3. Sets environment variables that force CUDA provider selection
  4. Launches main.py in-process with the correct provider order

Place this file next to main.py and run it instead.
"""

import os
import sys
import subprocess
from pathlib import Path


# ============================================================
# 1. Ensure onnxruntime-gpu is the ONLY onnxruntime installed
# ============================================================
def fix_onnxruntime_conflict():
    """The CPU onnxruntime package shadows onnxruntime-gpu. Remove it."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[!] onnxruntime is not installed at all. Installing onnxruntime-gpu...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime-gpu"])
        return

    # Check which package is actually installed by looking at the dist-info
    import importlib.metadata as md
    installed = {d.metadata["Name"].lower(): d.version
                 for d in md.distributions()}

    has_cpu = "onnxruntime" in installed
    has_gpu = "onnxruntime-gpu" in installed

    print(f"[*] Installed onnxruntime packages:")
    if has_cpu:
        print(f"    - onnxruntime (CPU):      {installed['onnxruntime']}")
    if has_gpu:
        print(f"    - onnxruntime-gpu (GPU):  {installed['onnxruntime-gpu']}")

    if has_cpu:
        print("[!] CPU onnxruntime is installed. It shadows the GPU build.")
        print("[*] Uninstalling onnxruntime (CPU) and onnxruntime-gpu, then reinstalling GPU only...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"])
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime-gpu"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "onnxruntime-gpu"])
        print("[✓] Reinstalled. Please re-run this wrapper.")
        sys.exit(0)


# ============================================================
# 2. Set environment variables BEFORE importing onnxruntime
# ============================================================
def configure_cuda_env():
    """Provider priority and CUDA tuning. Must run before ort import."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Force CUDA provider to be tried first
    os.environ.setdefault("ORT_CUDA_USE_TF32", "1")
    # Prevent fallback silence
    os.environ.setdefault("ORT_LOGGING_LEVEL", "2")
    # For some ACE-Step builds using cudnn
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ============================================================
# 3. Verify providers and report
# ============================================================
def check_providers():
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"[*] onnxruntime version: {ort.__version__}")
    print(f"[*] Available providers: {providers}")

    if "CUDAExecutionProvider" not in providers:
        print()
        print("[✗] CUDAExecutionProvider is NOT available.")
        print("    Common causes:")
        print("      - Missing or mismatched CUDA Toolkit (onnxruntime-gpu 1.17+ needs CUDA 12.x)")
        print("      - Missing cuDNN 8.x DLLs on PATH")
        print("      - NVIDIA driver too old")
        print("    Install CUDA 12.x + cuDNN 8.x, then re-run.")
        sys.exit(1)

    print("[✓] CUDAExecutionProvider is available.")
    return providers


# ============================================================
# 4. Monkey-patch InferenceSession to default to CUDA
# ============================================================
def force_cuda_provider():
    """
    Some libraries (like ACE-Step) construct InferenceSession without
    specifying providers, which defaults to CPU. This patches the default.
    """
    import onnxruntime as ort
    _orig_init = ort.InferenceSession.__init__

    def patched_init(self, path_or_bytes, sess_options=None,
                     providers=None, provider_options=None, **kwargs):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _orig_init(self, path_or_bytes, sess_options=sess_options,
                   providers=providers, provider_options=provider_options, **kwargs)

    ort.InferenceSession.__init__ = patched_init
    print("[✓] Patched InferenceSession to default to CUDAExecutionProvider.")


# ============================================================
# 5. Launch main.py in-process
# ============================================================
def launch_main():
    main_path = Path(__file__).parent / "main.py"
    if not main_path.exists():
        print(f"[✗] main.py not found next to this wrapper at {main_path}")
        sys.exit(1)

    print(f"[*] Launching {main_path}...\n")
    # Make main.py runnable as if it were __main__
    sys.argv = [str(main_path), *sys.argv[1:]]
    code = main_path.read_text(encoding="utf-8")
    # exec in the wrapper's namespace with __name__ == "__main__"
    exec(compile(code, str(main_path), "exec"),
         {"__name__": "__main__", "__file__": str(main_path)})


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  ACE-Step GPU Wrapper — forcing CUDAExecutionProvider")
    print("=" * 60)

    fix_onnxruntime_conflict()
    configure_cuda_env()
    check_providers()
    force_cuda_provider()

    print("=" * 60)
    print()

    launch_main()