#!/usr/bin/env python3
"""
TTS Studio GUI — a front end for the existing main.py.

- Grey read-only placeholder in the text box.
- LOAD .TXT and LOAD .PDF buttons.
- Drag and drop .txt and .pdf onto the window.
- Copy / paste works normally.
- PROCESS runs main.py (same venv as run.bat) via a temp text file,
  so no Windows command-line length limit is ever hit.
- Output filenames are timestamped, so playback never locks the next run.
- Full subprocess output is written to last_error.log.
- PLAY stays disabled until processing finishes, then plays inside the GUI.
- STOP halts playback.
- Auto-detects GPU (CUDA) and CPU cores; auto-applies best settings.
- Settings dialog under Options.
- main.py is never modified by the GUI.
"""

import os
import sys
import json
import time
import tempfile
import threading
import subprocess
import platform
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog, messagebox, Menu

try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False

try:
    from pypdf import PdfReader
    PYPDF_OK = True
except ImportError:
    PYPDF_OK = False

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_OK = True
except ImportError:
    DND_OK = False


# ------------------------------------------------------------------
# Theme
# ------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

COLORS = {
    "bg_deep":       "#0a0e1a",
    "bg_panel":      "#111827",
    "bg_card":       "#1a2233",
    "bg_input":      "#0f172a",
    "accent":        "#7c3aed",
    "accent_hover":  "#8b5cf6",
    "accent_glow":   "#a78bfa",
    "success":       "#10b981",
    "success_hover": "#34d399",
    "warning":       "#f59e0b",
    "danger":        "#ef4444",
    "text_primary":  "#f1f5f9",
    "text_muted":    "#94a3b8",
    "border":        "#1e293b",
    "disabled":      "#334155",
}


# ------------------------------------------------------------------
# Base window with optional drag-and-drop
# ------------------------------------------------------------------
if DND_OK:
    class _Base(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self):
            super().__init__()
            self.TkdndVersion = TkinterDnD._require(self)
else:
    class _Base(ctk.CTk):
        def __init__(self):
            super().__init__()


# ------------------------------------------------------------------
# Hardware / settings detection
# ------------------------------------------------------------------
def detect_best_settings(project_dir):
    settings = {
        "use_cuda": False,
        "num_threads": 4,
        "gpu_name": None,
        "cpu_cores": 1,
    }

    try:
        import psutil
        cores = psutil.cpu_count(logical=False) or 1
    except Exception:
        cores = os.cpu_count() or 1
    settings["cpu_cores"] = cores
    settings["num_threads"] = max(1, min(16, cores))

    try:
        import torch
        if torch.cuda.is_available():
            settings["use_cuda"] = True
            settings["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    if settings["use_cuda"]:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if not any("CUDA" in p for p in providers):
                settings["use_cuda"] = False
        except Exception:
            settings["use_cuda"] = False

    return settings


def load_settings(project_dir):
    path = project_dir / "tts_settings.json"
    auto = detect_best_settings(project_dir)

    user = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                user = json.load(f)
        except Exception:
            user = {}

    merged = dict(auto)
    for k in ("use_cuda", "num_threads"):
        if k in user:
            merged[k] = user[k]
    merged["_auto"] = auto
    merged["_path"] = str(path)
    return merged


def save_settings(project_dir, settings):
    path = project_dir / "tts_settings.json"
    payload = {
        "use_cuda": bool(settings.get("use_cuda", False)),
        "num_threads": int(settings.get("num_threads", 4)),
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


# ------------------------------------------------------------------
# Main window
# ------------------------------------------------------------------
class TTSStudio(_Base):
    def __init__(self):
        super().__init__()

        self.title("TTS Studio")
        self.geometry("1120x820")
        self.minsize(920, 680)
        self.configure(fg_color=COLORS["bg_deep"])

        self.project_dir = Path(__file__).resolve().parent
        self.python_exe = self._find_venv_python()
        self.main_py = self.project_dir / "main.py"

        self.last_output = None
        self.is_processing = False
        self.is_playing = False
        self._placeholder_active = False

        self.settings = load_settings(self.project_dir)

        self._init_audio()
        self._build_menu()
        self._build_layout()
        self._register_drag_drop()

        self.after(100, self._refresh_engine_indicator)

    # --------------------------------------------------------------
    # Python interpreter
    # --------------------------------------------------------------
    def _find_venv_python(self):
        if platform.system() == "Windows":
            candidate = self.project_dir / ".venv" / "Scripts" / "python.exe"
        else:
            candidate = self.project_dir / ".venv" / "bin" / "python"
        return str(candidate) if candidate.exists() else sys.executable

    def _init_audio(self):
        if PYGAME_OK:
            try:
                pygame.mixer.init()
            except Exception:
                pass

    # --------------------------------------------------------------
    # Menu
    # --------------------------------------------------------------
    def _build_menu(self):
        menubar = Menu(self, bg=COLORS["bg_panel"], fg=COLORS["text_primary"],
                       activebackground=COLORS["accent"],
                       activeforeground="#ffffff", borderwidth=0)

        file_menu = Menu(menubar, tearoff=0, bg=COLORS["bg_panel"],
                         fg=COLORS["text_primary"],
                         activebackground=COLORS["accent"],
                         activeforeground="#ffffff")
        file_menu.add_command(label="New Text", accelerator="Ctrl+N",
                              command=self._new_text)
        file_menu.add_command(label="Open Text File...", accelerator="Ctrl+O",
                              command=self._open_text)
        file_menu.add_command(label="Open PDF...", accelerator="Ctrl+P",
                              command=self._open_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="Save Audio As...", accelerator="Ctrl+S",
                              command=self._save_audio)
        file_menu.add_command(label="Open Output Folder",
                              command=self._open_output_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._quit)
        menubar.add_cascade(label="File", menu=file_menu)

        options_menu = Menu(menubar, tearoff=0, bg=COLORS["bg_panel"],
                            fg=COLORS["text_primary"],
                            activebackground=COLORS["accent"],
                            activeforeground="#ffffff")
        options_menu.add_command(label="Performance Settings...",
                                 command=self._show_performance_settings)
        options_menu.add_separator()
        options_menu.add_command(label="Show main.py path",
                                 command=self._show_main_path)
        options_menu.add_command(label="Show venv python",
                                 command=self._show_python_path)
        menubar.add_cascade(label="Options", menu=options_menu)

        about_menu = Menu(menubar, tearoff=0, bg=COLORS["bg_panel"],
                          fg=COLORS["text_primary"],
                          activebackground=COLORS["accent"],
                          activeforeground="#ffffff")
        about_menu.add_command(label="About", command=self._show_about)
        about_menu.add_command(label="Hardware Info", command=self._show_hardware)
        menubar.add_cascade(label="About", menu=about_menu)

        self.config(menu=menubar)

        self.bind("<Control-n>", lambda e: self._new_text())
        self.bind("<Control-o>", lambda e: self._open_text())
        self.bind("<Control-p>", lambda e: self._open_pdf())
        self.bind("<Control-s>", lambda e: self._save_audio())
        self.bind("<Control-Return>", lambda e: self._start_processing())

    # --------------------------------------------------------------
    # Layout
    # --------------------------------------------------------------
    def _build_layout(self):
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"], height=70,
                              corner_radius=0)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="  TTS STUDIO",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=COLORS["accent_glow"],
        ).pack(side="left", padx=20)

        ctk.CTkLabel(
            header, text="Wrapper for main.py",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLORS["text_muted"],
        ).pack(side="left", padx=4)

        self.engine_badge = ctk.CTkLabel(
            header, text="Engine: ...",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color=COLORS["accent_glow"],
        )
        self.engine_badge.pack(side="left", padx=(16, 0))

        self.status_dot = ctk.CTkLabel(
            header, text="●", font=ctk.CTkFont(size=16),
            text_color=COLORS["success"],
        )
        self.status_dot.pack(side="right", padx=(0, 10))

        self.status_label = ctk.CTkLabel(
            header, text="Ready",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLORS["text_muted"],
        )
        self.status_label.pack(side="right", padx=(0, 6))

        main = ctk.CTkFrame(self, fg_color=COLORS["bg_deep"], corner_radius=0)
        main.pack(fill="both", expand=True)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        self._build_editor(main)
        self._build_controls(main)
        self._build_statusbar()

    def _build_editor(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"], corner_radius=12)
        frame.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)

        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(16, 8))

        ctk.CTkLabel(
            row, text="TEXT INPUT",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=COLORS["text_muted"],
        ).pack(side="left")

        self.char_count = ctk.CTkLabel(
            row, text="0 characters",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLORS["text_muted"],
        )
        self.char_count.pack(side="right")

        load_row = ctk.CTkFrame(frame, fg_color="transparent")
        load_row.pack(fill="x", padx=20, pady=(0, 8))

        ctk.CTkButton(
            load_row, text="📄  LOAD .TXT",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["accent"],
            height=32,
            command=self._open_text,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        ctk.CTkButton(
            load_row, text="📕  LOAD .PDF",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["accent"],
            height=32,
            command=self._open_pdf,
        ).pack(side="left", fill="x", expand=True, padx=(4, 4))

        ctk.CTkButton(
            load_row, text="🗑  CLEAR",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["danger"],
            height=32,
            command=self._new_text,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

        self.text_box = ctk.CTkTextbox(
            frame,
            font=ctk.CTkFont(family="Consolas", size=14),
            fg_color=COLORS["bg_input"],
            text_color=COLORS["text_primary"],
            border_color=COLORS["border"],
            border_width=1,
            corner_radius=8,
            wrap="word",
            undo=True,
        )
        self.text_box.pack(fill="both", expand=True, padx=20, pady=(0, 8))

        drop_hint_text = (
            "Drag and drop a .txt or .pdf here, or use the buttons above."
            if DND_OK else
            "Drag and drop unavailable. Run: pip install tkinterdnd2"
        )
        ctk.CTkLabel(
            frame, text=drop_hint_text,
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=COLORS["text_muted"],
        ).pack(padx=20, pady=(0, 12))

        self.placeholder_text = (
            "Type or paste your text here. There is no limit.\n\n"
            "You can also drag a .txt or .pdf file onto this window,\n"
            "or use the LOAD buttons above.\n\n"
            "Press Ctrl+Enter or click PROCESS to generate audio."
        )
        inner = self.text_box._textbox
        inner.tag_configure("placeholder", foreground=COLORS["text_muted"])

        inner.insert("1.0", self.placeholder_text)
        inner.tag_add("placeholder", "1.0", "end-1c")
        inner.mark_set("insert", "1.0")
        inner.configure(insertwidth=0)
        inner.bind("<Key>", self._on_key, add="+")
        inner.bind("<Button-1>", self._on_click, add="+")
        inner.bind("<<Paste>>", self._on_paste, add="+")
        inner.bind("<KeyRelease>", self._on_key_release, add="+")
        self._placeholder_active = True

    def _build_controls(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=COLORS["bg_panel"], corner_radius=12)
        panel.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)

        ctk.CTkLabel(
            panel, text="SPEED",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w", padx=20, pady=(20, 6))

        speed_row = ctk.CTkFrame(panel, fg_color="transparent")
        speed_row.pack(fill="x", padx=20)

        self.speed_var = ctk.DoubleVar(value=1.0)
        self.speed_slider = ctk.CTkSlider(
            speed_row, from_=0.5, to=2.0, number_of_steps=15,
            variable=self.speed_var,
            progress_color=COLORS["accent"],
            button_color=COLORS["accent_glow"],
            button_hover_color=COLORS["accent_hover"],
            command=self._update_speed_label,
        )
        self.speed_slider.pack(side="left", fill="x", expand=True)

        self.speed_label = ctk.CTkLabel(
            speed_row, text="1.00x",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLORS["text_primary"], width=50,
        )
        self.speed_label.pack(side="right", padx=(8, 0))

        ctk.CTkLabel(
            panel, text="OUTPUT PREFIX",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w", padx=20, pady=(20, 6))

        self.output_var = ctk.StringVar(value="speech")
        ctk.CTkEntry(
            panel, textvariable=self.output_var,
            fg_color=COLORS["bg_input"],
            border_color=COLORS["border"],
            font=ctk.CTkFont(family="Segoe UI", size=12),
        ).pack(fill="x", padx=20)

        ctk.CTkLabel(
            panel, text="Each run is saved as prefix_YYYYMMDD_HHMMSS.wav",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w", padx=20)

        ctk.CTkLabel(
            panel, text="PROGRESS",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w", padx=20, pady=(20, 6))

        self.progress = ctk.CTkProgressBar(
            panel, progress_color=COLORS["accent"],
            fg_color=COLORS["bg_input"], height=8,
        )
        self.progress.set(0)
        self.progress.pack(fill="x", padx=20)

        self.progress_label = ctk.CTkLabel(
            panel, text="Idle",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLORS["text_muted"],
        )
        self.progress_label.pack(anchor="w", padx=20, pady=(4, 0))

        self.process_button = ctk.CTkButton(
            panel, text="⚙  PROCESS",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            height=52, corner_radius=10,
            command=self._start_processing,
        )
        self.process_button.pack(fill="x", padx=20, pady=(24, 8))

        self.play_button = ctk.CTkButton(
            panel, text="▶  PLAY",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            fg_color=COLORS["disabled"],
            hover_color=COLORS["success_hover"],
            height=52, corner_radius=10,
            state="disabled",
            command=self._play_audio,
        )
        self.play_button.pack(fill="x", padx=20, pady=(0, 8))

        self.stop_button = ctk.CTkButton(
            panel, text="■  STOP",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["danger"],
            height=32, corner_radius=8,
            state="disabled",
            command=self._stop_audio,
        )
        self.stop_button.pack(fill="x", padx=20, pady=(0, 12))

        btn_row = ctk.CTkFrame(panel, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkButton(
            btn_row, text="Save As",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["accent"],
            height=34,
            command=self._save_audio,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        ctk.CTkButton(
            btn_row, text="Open Folder",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["accent"],
            height=34,
            command=self._open_output_folder,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

    def _build_statusbar(self):
        bar = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"], height=30,
                           corner_radius=0)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self.status_text = ctk.CTkLabel(
            bar, text="  Ready. Calls main.py using the same venv as run.bat.",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=COLORS["text_muted"],
        )
        self.status_text.pack(side="left", padx=10)

        self.engine_label = ctk.CTkLabel(
            bar, text="Engine: --  ",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=COLORS["text_muted"],
        )
        self.engine_label.pack(side="right", padx=10)

    # --------------------------------------------------------------
    # Engine indicator / settings
    # --------------------------------------------------------------
    def _refresh_engine_indicator(self):
        s = self.settings
        if s.get("use_cuda"):
            name = s.get("gpu_name") or "CUDA"
            short = name if len(name) <= 24 else name[:22] + "…"
            badge = f"Engine: CUDA  ({short})"
            bar = f"Engine: CUDA × {s.get('num_threads', 4)} threads  "
        else:
            badge = f"Engine: CPU × {s.get('num_threads', 4)} threads"
            bar = f"Engine: CPU × {s.get('num_threads', 4)} threads  "

        try:
            self.engine_badge.configure(text=badge)
            self.engine_label.configure(text=bar)
        except Exception:
            pass

    def _settings_env(self):
        env = os.environ.copy()
        env["TTS_USE_CUDA"] = "1" if self.settings.get("use_cuda") else "0"
        env["TTS_NUM_THREADS"] = str(int(self.settings.get("num_threads", 4)))
        env["PYTHONUNBUFFERED"] = "1"
        return env

    # --------------------------------------------------------------
    # Performance settings dialog
    # --------------------------------------------------------------
    def _show_performance_settings(self):
        win = ctk.CTkToplevel(self)
        win.title("Performance Settings")
        win.geometry("540x520")
        win.configure(fg_color=COLORS["bg_deep"])
        win.transient(self)
        win.grab_set()

        auto = self.settings.get("_auto", {})
        current = self.settings

        ctk.CTkLabel(
            win, text="PERFORMANCE",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=COLORS["accent_glow"],
        ).pack(pady=(24, 4))

        ctk.CTkLabel(
            win, text="Auto-detected at startup. Override if you wish.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLORS["text_muted"],
        ).pack(pady=(0, 16))

        auto_card = ctk.CTkFrame(win, fg_color=COLORS["bg_panel"], corner_radius=10)
        auto_card.pack(fill="x", padx=24, pady=(0, 12))

        auto_lines = [
            f"CPU cores: {auto.get('cpu_cores', '?')}",
            f"CUDA available: {'yes' if auto.get('use_cuda') else 'no'}",
            f"Detected GPU: {auto.get('gpu_name') or 'none'}",
        ]
        for line in auto_lines:
            ctk.CTkLabel(
                auto_card, text="  " + line,
                font=ctk.CTkFont(family="Consolas", size=11),
                text_color=COLORS["text_primary"], anchor="w",
            ).pack(fill="x", padx=12, pady=2)

        ctk.CTkLabel(
            win, text="Use GPU (CUDA) if available",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=COLORS["text_primary"],
        ).pack(anchor="w", padx=24, pady=(12, 4))

        use_cuda_var = ctk.BooleanVar(value=bool(current.get("use_cuda")))
        ctk.CTkCheckBox(
            win, text="Enable CUDA acceleration",
            variable=use_cuda_var,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=ctk.CTkFont(family="Segoe UI", size=11),
        ).pack(anchor="w", padx=24)

        ctk.CTkLabel(
            win, text="CPU thread count",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=COLORS["text_primary"],
        ).pack(anchor="w", padx=24, pady=(16, 4))

        thread_row = ctk.CTkFrame(win, fg_color="transparent")
        thread_row.pack(fill="x", padx=24)

        threads_var = ctk.IntVar(value=int(current.get("num_threads", 4)))
        threads_label = ctk.CTkLabel(
            thread_row, text=f"{threads_var.get()}",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLORS["text_primary"], width=40,
        )

        def on_thread_change(value):
            threads_var.set(int(value))
            threads_label.configure(text=f"{int(value)}")

        ctk.CTkSlider(
            thread_row, from_=1, to=16, number_of_steps=15,
            variable=threads_var,
            progress_color=COLORS["accent"],
            button_color=COLORS["accent_glow"],
            button_hover_color=COLORS["accent_hover"],
            command=on_thread_change,
        ).pack(side="left", fill="x", expand=True)
        threads_label.pack(side="right", padx=(8, 0))

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=24, pady=(24, 24))

        def apply_and_close():
            self.settings["use_cuda"] = bool(use_cuda_var.get())
            self.settings["num_threads"] = int(threads_var.get())
            save_settings(self.project_dir, self.settings)
            self._refresh_engine_indicator()
            self._set_status("Performance settings saved.", "success")
            win.destroy()

        def restore_auto():
            use_cuda_var.set(bool(auto.get("use_cuda")))
            threads_var.set(int(auto.get("num_threads", 4)))
            threads_label.configure(text=f"{int(auto.get('num_threads', 4))}")

        ctk.CTkButton(
            btn_row, text="Restore Auto",
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["accent"],
            command=restore_auto,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        ctk.CTkButton(
            btn_row, text="Cancel",
            fg_color=COLORS["bg_card"],
            hover_color=COLORS["danger"],
            command=win.destroy,
        ).pack(side="left", fill="x", expand=True, padx=(4, 4))

        ctk.CTkButton(
            btn_row, text="Apply",
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            command=apply_and_close,
        ).pack(side="left", fill="x", expand=True, padx=(4, 0))

    # --------------------------------------------------------------
    # Drag and drop
    # --------------------------------------------------------------
    def _register_drag_drop(self):
        if not DND_OK:
            return
        try:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass

    def _parse_drop_paths(self, data):
        paths = []
        if not data:
            return paths
        buf = ""
        in_brace = False
        for ch in data:
            if ch == "{":
                in_brace = True
                buf = ""
            elif ch == "}":
                in_brace = False
                if buf:
                    paths.append(buf)
                buf = ""
            elif ch == " " and not in_brace:
                if buf:
                    paths.append(buf)
                    buf = ""
            else:
                buf += ch
        if buf:
            paths.append(buf)
        return paths

    def _on_drop(self, event):
        paths = self._parse_drop_paths(event.data)
        if not paths:
            return
        loaded = 0
        for p in paths:
            path = Path(p)
            if not path.exists():
                continue
            suffix = path.suffix.lower()
            if suffix == ".txt":
                if self._load_txt_file(path):
                    loaded += 1
            elif suffix == ".pdf":
                if self._load_pdf_file(path):
                    loaded += 1
        if loaded == 0:
            self._set_status(
                "Nothing loaded. Only .txt and .pdf are supported.",
                "warning",
            )

    # --------------------------------------------------------------
    # File loading helpers
    # --------------------------------------------------------------
    def _set_text(self, content):
        inner = self.text_box._textbox
        inner.delete("1.0", "end")
        inner.tag_remove("placeholder", "1.0", "end")
        self._placeholder_active = False
        if content.strip():
            inner.insert("1.0", content)
        else:
            self._restore_placeholder()
        self._update_char_count()

    def _load_txt_file(self, path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            self._set_text(content)
            self._set_status(f"Loaded: {path.name}  ({len(content):,} chars)",
                             "success")
            return True
        except Exception as e:
            self._set_status(f"TXT load failed: {e}", "danger")
            return False

    def _load_pdf_file(self, path):
        if not PYPDF_OK:
            messagebox.showerror(
                "PDF support unavailable",
                "pypdf is not installed.\n\n"
                "Run this in your venv:\n"
                "    pip install pypdf"
            )
            return False
        try:
            self._set_status(f"Reading: {path.name}...", "warning")
            self.update_idletasks()
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages:
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                if text.strip():
                    pages.append(text.strip())
            content = "\n\n".join(pages)
            if not content.strip():
                self._set_status(
                    f"{path.name}: no extractable text (may be a scanned PDF)",
                    "warning",
                )
                return False
            self._set_text(content)
            self._set_status(
                f"Loaded: {path.name}  ({len(reader.pages)} pages, "
                f"{len(content):,} chars)",
                "success",
            )
            return True
        except Exception as e:
            self._set_status(f"PDF load failed: {e}", "danger")
            return False

    # --------------------------------------------------------------
    # Placeholder behavior
    # --------------------------------------------------------------
    def _clear_placeholder(self):
        if not self._placeholder_active:
            return
        inner = self.text_box._textbox
        inner.delete("1.0", "end")
        inner.tag_remove("placeholder", "1.0", "end")
        inner.configure(insertwidth=2)
        self._placeholder_active = False

    def _restore_placeholder(self):
        inner = self.text_box._textbox
        if inner.get("1.0", "end-1c").strip() != "":
            return
        if self._placeholder_active:
            return
        inner.delete("1.0", "end")
        inner.insert("1.0", self.placeholder_text)
        inner.tag_add("placeholder", "1.0", "end-1c")
        inner.mark_set("insert", "1.0")
        inner.configure(insertwidth=0)
        self._placeholder_active = True

    def _on_key(self, event):
        nav_keys = {
            "Left", "Right", "Up", "Down", "Home", "End",
            "Prior", "Next", "Shift_L", "Shift_R",
            "Control_L", "Control_R", "Alt_L", "Alt_R",
            "Caps_Lock", "Escape", "Tab",
        }
        if self._placeholder_active:
            if event.keysym in nav_keys:
                return
            self._clear_placeholder()

    def _on_click(self, event):
        if self._placeholder_active:
            self._clear_placeholder()

    def _on_paste(self, event):
        if self._placeholder_active:
            self._clear_placeholder()

    def _on_key_release(self, event=None):
        self._update_char_count()
        if not self._placeholder_active:
            inner = self.text_box._textbox
            if inner.get("1.0", "end-1c").strip() == "":
                self._restore_placeholder()
                self._update_char_count()

    # --------------------------------------------------------------
    # Text helpers
    # --------------------------------------------------------------
    def _update_char_count(self, event=None):
        if self._placeholder_active:
            self.char_count.configure(text="0 characters")
            return
        text = self.text_box.get("1.0", "end-1c")
        self.char_count.configure(text=f"{len(text):,} characters")

    def _update_speed_label(self, value):
        self.speed_label.configure(text=f"{float(value):.2f}x")

    def _new_text(self):
        inner = self.text_box._textbox
        inner.delete("1.0", "end")
        inner.tag_remove("placeholder", "1.0", "end")
        self._placeholder_active = False
        self._restore_placeholder()
        self._update_char_count()
        self._set_status("New document.", "success")

    def _open_text(self):
        path = filedialog.askopenfilename(
            title="Open text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self._load_txt_file(Path(path))

    def _open_pdf(self):
        path = filedialog.askopenfilename(
            title="Open PDF file",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self._load_pdf_file(Path(path))

    def _save_audio(self):
        if not self.last_output or not Path(self.last_output).exists():
            messagebox.showinfo("No audio", "Process first.")
            return
        dest = filedialog.asksaveasfilename(
            title="Save audio as",
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if dest:
            import shutil
            shutil.copy(self.last_output, dest)
            self._set_status(f"Saved: {Path(dest).name}", "success")

    def _open_output_folder(self):
        folder = self.project_dir
        if platform.system() == "Windows":
            os.startfile(folder)
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(folder)])
        else:
            subprocess.run(["xdg-open", str(folder)])

    # --------------------------------------------------------------
    # Processing — unique output filename per run
    # --------------------------------------------------------------
    def _start_processing(self):
        if self.is_processing:
            return
        if self._placeholder_active:
            messagebox.showwarning("Empty", "Enter some text first.")
            return
        text = self.text_box.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Empty", "Enter some text first.")
            return
        if not self.main_py.exists():
            messagebox.showerror("Missing file",
                                 f"main.py not found at:\n{self.main_py}")
            return

        self._stop_audio()
        self.is_processing = True
        self.last_output = None

        self.process_button.configure(state="disabled", text="⏳  PROCESSING...")
        self.play_button.configure(state="disabled",
                                   fg_color=COLORS["disabled"])
        self.stop_button.configure(state="disabled")

        self.progress.set(0.1)
        self.progress_label.configure(text="Running main.py...")
        self._set_status("Processing...", "warning")

        threading.Thread(target=self._run_main, args=(text,),
                         daemon=True).start()

    def _run_main(self, text):
        prefix = self.output_var.get().strip() or "speech"
        # Strip any accidental extension the user typed
        if prefix.lower().endswith(".wav"):
            prefix = prefix[:-4]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{prefix}_{timestamp}.wav"

        tmp_file = None
        log_path = self.project_dir / "last_error.log"
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".txt", prefix="tts_text_", dir=str(self.project_dir)
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            tmp_file = tmp_path

            cmd = [
                self.python_exe,
                str(self.main_py),
                "--text-file", tmp_file,
                "--output", output_name,
            ]

            proc = subprocess.Popen(
                cmd,
                cwd=str(self.project_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self._settings_env(),
            )

            log_lines = []
            for line in proc.stdout:
                stripped = line.rstrip()
                log_lines.append(stripped)
                self._append_log(stripped)

            proc.wait()

            # Save full output for inspection if anything went wrong
            try:
                with open(log_path, "w", encoding="utf-8") as lf:
                    lf.write("\n".join(log_lines))
            except Exception:
                pass

            if proc.returncode == 0:
                out_path = self.project_dir / output_name
                if out_path.exists():
                    self.after(0, self._on_success, str(out_path))
                else:
                    self.after(0, self._on_error,
                               f"main.py finished but no file at:\n{out_path}")
            else:
                self.after(0, self._on_error,
                           f"main.py exited with code {proc.returncode}.\n"
                           f"See last_error.log for the full message.")

        except Exception as e:
            self.after(0, self._on_error, f"{type(e).__name__}: {e}")
        finally:
            if tmp_file and Path(tmp_file).exists():
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

    def _append_log(self, line):
        short = line[:70]
        self.after(0, lambda: self.progress_label.configure(text=short))
        self.after(0, lambda: self.progress.set(min(0.95, self.progress.get() + 0.05)))

    def _on_success(self, output_path):
        self.is_processing = False
        self.last_output = output_path
        self.process_button.configure(state="normal", text="⚙  PROCESS")
        self.play_button.configure(state="normal", fg_color=COLORS["success"])
        self.stop_button.configure(state="normal")
        self.progress.set(1.0)
        self.progress_label.configure(text="Ready to play")
        self._set_status(f"Done: {Path(output_path).name}", "success")

    def _on_error(self, message):
        self.is_processing = False
        self.process_button.configure(state="normal", text="⚙  PROCESS")
        self.play_button.configure(state="disabled", fg_color=COLORS["disabled"])
        self.stop_button.configure(state="disabled")
        self.progress.set(0)
        self.progress_label.configure(text="Error")
        self._set_status("Error", "danger")
        messagebox.showerror("Error", message)

    # --------------------------------------------------------------
    # Playback
    # --------------------------------------------------------------
    def _play_audio(self):
        if not self.last_output or not Path(self.last_output).exists():
            messagebox.showinfo("No audio", "Process first.")
            return
        if not PYGAME_OK:
            messagebox.showerror(
                "Playback unavailable",
                "pygame is not installed.\n\n"
                "Run this in your venv:\n"
                "    pip install pygame"
            )
            return
        try:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception:
                pass
            pygame.mixer.music.load(str(self.last_output))
            pygame.mixer.music.play()
            self.is_playing = True
            self._set_status(f"Playing: {Path(self.last_output).name}", "success")
            threading.Thread(target=self._watch_playback, daemon=True).start()
        except Exception as e:
            self._set_status(f"Playback failed: {e}", "danger")

    def _watch_playback(self):
        while PYGAME_OK and pygame.mixer.music.get_busy():
            time.sleep(0.2)
        self.is_playing = False
        self.after(0, lambda: self._set_status("Ready", "muted"))

    def _stop_audio(self):
        if not PYGAME_OK:
            return
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass
        self.is_playing = False

    # --------------------------------------------------------------
    # Status
    # --------------------------------------------------------------
    def _set_status(self, text, level="muted"):
        color_map = {
            "muted":   COLORS["text_muted"],
            "success": COLORS["success"],
            "warning": COLORS["warning"],
            "danger":  COLORS["danger"],
        }
        color = color_map.get(level, COLORS["text_muted"])
        self.status_label.configure(text=text, text_color=color)
        self.status_dot.configure(text_color=color)
        self.status_text.configure(text=f"  {text}")

    # --------------------------------------------------------------
    # Dialogs
    # --------------------------------------------------------------
    def _show_main_path(self):
        messagebox.showinfo("main.py", str(self.main_py))

    def _show_python_path(self):
        messagebox.showinfo("Python interpreter", self.python_exe)

    def _show_about(self):
        win = ctk.CTkToplevel(self)
        win.title("About TTS Studio")
        win.geometry("480x420")
        win.configure(fg_color=COLORS["bg_deep"])
        win.transient(self)

        ctk.CTkLabel(
            win, text="TTS STUDIO",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=COLORS["accent_glow"],
        ).pack(pady=(30, 6))

        ctk.CTkLabel(
            win,
            text=(
                "GUI front end for main.py.\n\n"
                "PROCESS runs main.py the same way run.bat does,\n"
                "using a temporary text file so very long text works.\n"
                "Each run saves a unique .wav so playback never locks\n"
                "the next generation.\n"
                "Full subprocess output is saved to last_error.log.\n\n"
                "PLAY plays the resulting .wav inside this window.\n"
                "STOP halts playback.\n\n"
                "LOAD .PDF extracts text from a PDF into the box.\n"
                "Drag and drop .txt or .pdf onto the window.\n"
                "Copy and paste work as usual.\n\n"
                "Performance settings auto-detect GPU and CPU."
            ),
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLORS["text_primary"],
            justify="center",
        ).pack(pady=10)

        ctk.CTkButton(
            win, text="Close",
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            command=win.destroy,
        ).pack(pady=20)

    def _show_hardware(self):
        info_lines = []
        try:
            import psutil
            info_lines.append(f"CPU cores (physical): {psutil.cpu_count(logical=False)}")
            info_lines.append(f"CPU threads (logical): {psutil.cpu_count(logical=True)}")
            info_lines.append(f"RAM: {round(psutil.virtual_memory().total / 1024**3, 1)} GB")
        except Exception:
            info_lines.append("psutil not installed")

        try:
            import torch
            info_lines.append(f"Torch CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                info_lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            info_lines.append("torch not installed")

        try:
            import onnxruntime as ort
            info_lines.append("ONNX providers: " + ", ".join(ort.get_available_providers()))
        except Exception:
            info_lines.append("onnxruntime not installed")

        info_lines.append("")
        info_lines.append(f"Active: CUDA={self.settings.get('use_cuda')}  "
                          f"threads={self.settings.get('num_threads')}")

        messagebox.showinfo("Hardware", "\n".join(info_lines))

    def _quit(self):
        if self.is_processing:
            if not messagebox.askyesno("Quit", "Processing is running. Quit anyway?"):
                return
        self._stop_audio()
        self.destroy()


def main():
    app = TTSStudio()
    app.mainloop()


if __name__ == "__main__":
    main()