import os
import json
import threading
import time
import math
import copy
from pathlib import Path
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import scrolledtext
from tkinter import ttk

IMAGES_FOLDER = "images"
CUSTOM_DATA_DIR = "custom_classes"      # stores example images per class
CUSTOM_MODEL_PATH = "custom_head.pth"   # saved custom classifier weights
CUSTOM_META_PATH = "custom_classes.json" # class name index mapping
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# ImageNet normalization
INPUT_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Training defaults
DEFAULT_EPOCHS = 10
DEFAULT_LR = 0.001
MIN_IMAGES_PER_CLASS = 3

class ClassifierApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("AI Image Classifier — with Custom Training")
        self.geometry("1080x800")
        self.minsize(960, 640)

        # --- Style ---
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Palette
        self.C = {
            "bg":      "#0f1724",
            "card":    "#0b1220",
            "primary": "#0EA5A4",
            "accent":  "#7C3AED",
            "text":    "#E6EEF8",
            "muted":   "#9AA6B2",
            "danger":  "#EF4444",
            "success": "#22C55E",
        }
        self.F = {
            "heading": ("Segoe UI", 15, "bold"),
            "normal":  ("Segoe UI", 11),
            "mono":    ("Consolas", 10),
            "small":   ("Segoe UI", 9),
        }

        self._configure_styles()

        # --- State ---
        self.classifier = None
        self.current_photo = None
        self.selected_folder = IMAGES_FOLDER

        # --- Build UI ---
        self._build_ui()
        # --- Load model ---
        #self._set_status("Loading model...")
        #threading.Thread(target=self._load_model, daemon=True).start()


    def _configure_styles(self):
        C, F = self.C, self.F
        s = self.style
        s.configure("TFrame",        background=C["bg"])
        s.configure("Card.TFrame",   background=C["card"], relief="flat")
        s.configure("TLabel",        background=C["bg"],   foreground=C["text"], font=F["normal"])
        s.configure("Card.TLabel",   background=C["card"], foreground=C["text"], font=F["normal"])
        s.configure("Heading.TLabel",background=C["bg"],   foreground=C["text"], font=F["heading"])
        s.configure("Muted.TLabel",  background=C["bg"],   foreground=C["muted"],font=F["normal"])
        s.configure("CardMuted.TLabel", background=C["card"], foreground=C["muted"], font=F["small"])
        s.configure("Mono.TLabel",   background=C["card"], foreground=C["text"], font=F["mono"])
        s.configure("TButton",       font=F["normal"], padding=6)
        s.configure("Accent.TButton",foreground="#fff", background=C["accent"])
        s.map("Accent.TButton",      background=[("active", C["primary"])])
        s.configure("Primary.TButton",foreground="#fff", background=C["primary"])
        s.configure("Danger.TButton", foreground="#fff", background=C["danger"])
        s.configure("Success.TButton",foreground="#fff", background=C["success"])
        s.configure("TProgressbar",  troughcolor=C["card"], background=C["primary"])

    def _build_ui(self):
        C = self.C

        # --- Top bar ---
        top = ttk.Frame(self, padding=(18, 10))
        top.pack(fill=tk.X)
        ttk.Label(top, text="AI Image Classifier", style="Heading.TLabel").pack(side=tk.LEFT)
        ttk.Label(top, text="ImageNet 1000 + Custom Classes", style="Muted.TLabel").pack(side=tk.LEFT, padx=(12, 0))

        main = ttk.Frame(self, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Preview card
        preview = ttk.Frame(left, style="Card.TFrame", padding=10)
        preview.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(preview, text="Preview", style="CardMuted.TLabel").pack(anchor=tk.W)
        self.preview_label = ttk.Label(preview, text="No image loaded", anchor="center", style="Card.TLabel")
        self.preview_label.pack(fill=tk.BOTH, expand=True, pady=(8, 4))

        # Info card
        info_card = ttk.Frame(preview, style="Card.TFrame")
        info_card.pack(fill=tk.X, pady=(4, 4))
        self.info_text = tk.StringVar(value="Tensor: —    Prediction: —")
        ttk.Label(info_card, textvariable=self.info_text, style="Mono.TLabel").pack(anchor=tk.W, padx=4, pady=6)

        self.info_var = tk.StringVar(value="Prediction: —")
        ttk.Label(preview, textvariable=self.info_var, style="Mono.TLabel").pack(anchor=tk.W, padx=4, pady=(4, 2))

        # Log card
        log_card = ttk.Frame(left, style="Card.TFrame", padding=8)
        log_card.pack(fill=tk.BOTH, expand=True)
        ttk.Label(log_card, text="Activity Log", style="CardMuted.TLabel").pack(anchor=tk.W)
        self.log_box = scrolledtext.ScrolledText(log_card, height=12, bg=C["card"], fg=C["text"], insertbackground=C["text"], wrap=tk.WORD, font=self.F["mono"], relief=tk.FLAT)
        self.log_box.pack(fill=tk.BOTH, expand=True, pady=(6, 2))
        self.log_box.insert(tk.END, "Ready. Load model in progress...\n")
        self.log_box.config(state=tk.DISABLED)

        # Right column
        right = ttk.Frame(main, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y, anchor=tk.N)

        controls = ttk.Frame(right, style="Card.TFrame", padding=12)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Classify Single Image", style="Mono.TLabel").pack(anchor=tk.W, pady=(0, 6))

        row1 = ttk.Frame(controls, style="Card.TFrame")
        row1.pack(fill=tk.X, pady=(0, 6))
        self.path_var = tk.StringVar()
        self.path_entry = ttk.Entry(row1, textvariable=self.path_var, width=28)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # self.path_entry.bind("<Return>", lambda e: self._classify_single())
        ttk.Button(row1, text="Browse", command=self._browse_file).pack(side=tk.LEFT, padx=(6, 0))
        #ttk.Button(row1, text="Classify", style="Accent.TButton", command=self._classify_single).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Separator(controls, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(controls, text="Classify Folder", style="Mono.TLabel").pack(anchor=tk.W, pady=(0, 6))
        row2 = ttk.Frame(controls, style="Card.TFrame")
        row2.pack(fill=tk.X, pady=(0, 4))
        self.folder_var = tk.StringVar(value=f"(default) {IMAGES_FOLDER}")
        ttk.Label(row2, textvariable=self.folder_var, style="Card.TLabel", wraplength=200).pack(side=tk.LEFT)
        ttk.Button(row2, text="Choose...", command=self._choose_folder).pack(side=tk.RIGHT, padx=(6, 0))

    def _browse_file(self):
        f = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.gif *.webp *.svg"), ("All", "*.*")])
        if f:
            self.path_var.set(f)

    def _choose_folder(self):
        d = filedialog.askdirectory(title="Select images folder")
        if d:
            self.selected_folder = d
            self.folder_var.set(d)
            self._log(f"Selected folder: {d}")



def main():
    app = ClassifierApp()
    app.mainloop()



if __name__ == "__main__":
    main()