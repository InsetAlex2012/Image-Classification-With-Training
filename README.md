# AI Image Classifier (Tkinter + PyTorch)

## Overview
A desktop GUI application for image classification using PyTorch and a modern Tkinter interface.  
It supports ImageNet-based classification and is structured for future extension into custom training and folder-based datasets.

---

## Features

### Core System
- Built with PyTorch + torchvision
- Uses pretrained model backbone (ImageNet-ready)
- Designed for both single-image and folder-based classification
- Modular structure for adding custom training features

### Image Handling
- Supports common image formats:
  - JPG / JPEG
  - PNG
  - BMP
  - GIF
  - WEBP / SVG (file selection support)
- Automatic loading via file browser or folder selection

### GUI
- Tkinter-based modern interface
- ttk themed layout system
- Image preview panel
- Activity log window
- Clean two-column layout (preview + controls)

### Architecture
- Separated UI and logic structure
- Designed for threading support (future-safe for model inference/training)
- Extensible classifier pipeline placeholder

---

## Requirements

Install dependencies:
pip install torch torchvision pillow

---

## How to Run

Run the application with:
python main.py

---

## How to Use

### 1. Load an Image
- Click "Browse"
- Select an image file

### 2. (Planned Feature)
- Single-image classification button (currently prepared in UI)

### 3. Folder Mode (UI Ready)
- Choose a folder for batch processing (logic not yet enabled in this version)

---

## Project Structure

main.py            Main application file  
images/            Default image folder  
(custom future)    Model weights / classifier extensions  

---

## Notes
- This version is a UI + pipeline foundation
- Classification logic is prepared for extension but not fully activated in this build
- Designed to be extended into:
  - Custom training system
  - Batch inference
  - Live prediction updates

---

## Author

Your Name Here
