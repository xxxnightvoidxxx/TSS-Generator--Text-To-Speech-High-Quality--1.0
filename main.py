#!/usr/bin/env python3
"""
TTS Generator - Text to Speech using Kokoro-82M
Best open-weight TTS model for quality vs. hardware requirements.

Supports:
  --text        Direct text string on the command line
  --text-file   Path to a UTF-8 text file (used for very long input
                so the Windows command-line length limit is not hit)
  --output      Output WAV file path
  --voice       Voice name
  --speed       Speed multiplier
  --setup       Detect hardware and prepare the model
"""

import os
import sys
import json
import argparse
import subprocess
import platform
from pathlib import Path


# ------------------------------------------------------------------
# 0. Self-bootstrap into virtual environment
# ------------------------------------------------------------------
def in_venv():
    return sys.prefix != sys.base_prefix


def restart_in_venv():
    venv_dir = Path(__file__).resolve().parent / ".venv"
    if not venv_dir.exists():
        print("[*] Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    if platform.system() == "Windows":
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        python_exe = venv_dir / "bin" / "python"

    os.execv(str(python_exe),
             [str(python_exe), str(Path(__file__).resolve())] + sys.argv[1:])


if not in_venv():
    restart_in_venv()


# ------------------------------------------------------------------
# 1. Dependency check
# ------------------------------------------------------------------
IMPORT_MAP = {
    "kokoro": "kokoro",
    "soundfile": "soundfile",
    "numpy": "numpy",
    "torch": "torch",
    "psutil": "psutil",
    "GPUtil": "GPUtil",
}


def ensure_dependencies():
    import importlib
    missing = []
    for name, pkg in IMPORT_MAP.items():
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[*] Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        for name in list(sys.modules.keys()):
            if name in IMPORT_MAP:
                del sys.modules[name]
        print("[OK] Dependencies installed.")


# ------------------------------------------------------------------
# 2. Hardware detection
# ------------------------------------------------------------------
def detect_hardware():
    import psutil

    info = {
        "cpu": platform.processor() or "Unknown",
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "has_cuda": False,
        "gpu": None,
        "vram_gb": 0.0,
    }

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
            info["has_cuda"] = True
    except Exception:
        pass

    if not info["has_cuda"]:
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                info["gpu"] = gpus[0].name
                info["vram_gb"] = round(gpus[0].memoryTotal / 1024.0, 1)
                info["has_cuda"] = True
        except Exception:
            pass

    return info


def print_hardware(info):
    print()
    print("=" * 50)
    print("  Hardware Detection")
    print("=" * 50)
    print(f"  CPU: {info['cpu']}")
    print(f"  RAM: {info['ram_gb']} GB")
    if info["gpu"]:
        print(f"  GPU: {info['gpu']} ({info['vram_gb']} GB VRAM)")
    else:
        print(f"  GPU: None (CPU mode)")
    print("=" * 50)
    print()


# ------------------------------------------------------------------
# 3. Model setup
# ------------------------------------------------------------------
def prepare_model():
    print("[*] Preparing Kokoro-82M TTS model...")
    print("[*] Downloading model files (first run only)...")

    try:
        from kokoro import KPipeline
        KPipeline(lang_code="a")
        print("[OK] Kokoro model ready.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False


# ------------------------------------------------------------------
# 4. Text-to-speech generation
# ------------------------------------------------------------------
def generate_speech(text, output_path, voice="af_heart", speed=1.0):
    from kokoro import KPipeline
    import soundfile as sf
    import numpy as np

    print(f"[*] Generating speech...")
    print(f"    Voice: {voice}")
    print(f"    Speed: {speed}x")
    print(f"    Length: {len(text):,} characters")

    pipeline = KPipeline(lang_code="a")

    audio_chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        audio_chunks.append(audio)

    if not audio_chunks:
        print("[ERROR] No audio generated.")
        return None

    full_audio = np.concatenate(audio_chunks)
    sf.write(output_path, full_audio, 24000)
    print(f"[OK] Speech saved: {output_path}")
    return output_path


# ------------------------------------------------------------------
# 5. Workflows
# ------------------------------------------------------------------
def run_setup():
    ensure_dependencies()
    hw = detect_hardware()
    print_hardware(hw)

    with open("hardware.json", "w", encoding="utf-8") as f:
        json.dump(hw, f, indent=2)

    prepare_model()
    print("[OK] Setup finished.")


def run_generate(args):
    ensure_dependencies()

    # ---- Read text from file if --text-file was given ----
    if getattr(args, "text_file", None):
        try:
            with open(args.text_file, "r", encoding="utf-8") as f:
                args.text = f.read()
        except Exception as e:
            print(f"[ERROR] Could not read text file: {e}")
            sys.exit(1)

    if not args.text:
        print("[ERROR] Provide text with --text or --text-file")
        sys.exit(1)

    output = args.output or "speech.wav"
    generate_speech(
        text=args.text,
        output_path=output,
        voice=args.voice,
        speed=args.speed,
    )


# ------------------------------------------------------------------
# 6. CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TTS Generator (Kokoro-82M)")
    parser.add_argument("--setup", action="store_true",
                        help="Download model and check hardware")
    parser.add_argument("--text", "-t", type=str,
                        help="Text to synthesize")
    parser.add_argument("--text-file", type=str,
                        help="Path to a file containing the text to synthesize")
    parser.add_argument("--output", "-o", type=str, default="speech.wav",
                        help="Output file")
    parser.add_argument("--voice", "-v", type=str, default="af_heart",
                        help="Voice name")
    parser.add_argument("--speed", "-s", type=float, default=1.0,
                        help="Speech speed (0.5-2.0)")

    args = parser.parse_args()

    if args.setup:
        run_setup()
    elif args.text or args.text_file:
        run_generate(args)
    else:
        parser.print_help()
        print()
        print("Examples:")
        print("  python main.py --setup")
        print("  python main.py --text \"Hello world.\"")
        print("  python main.py --text-file script.txt --output speech.wav")


if __name__ == "__main__":
    main()