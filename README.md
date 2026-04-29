# 🧠 Custom Image Classifier (PyTorch + Tkinter)

![Image Classification with Custom Training](image_2026-04-29_204613802.png)

A graphical **AI Image Classifier** built with Python, PyTorch, and Tkinter.  
It allows you to create your own image classes, train a model inside the app, and run predictions on single images or entire folders.

---

## ✨ Features

### 🧠 AI Functionality
- Uses pretrained **ResNet-18** backbone (ImageNet feature extractor)
- Custom classification head trained on user data
- Confidence-based predictions (% output)
- Supports multiple user-defined classes
- Feature extraction training (fast and lightweight)

### 🏋️ Training System
- Add custom classes through the GUI
- Select folders of images per class
- Automatic dataset copying into structured directories
- Train model directly inside the application
- Saves trained model locally (`custom_head.pth`)
- Saves class mapping (`custom_classes.json`)

### 🖼️ Image Support
- JPG / JPEG
- PNG
- BMP
- GIF
- Automatic preprocessing:
  - Resize to 224×224
  - Normalization (ImageNet standard)
- Single image + folder batch classification

### 🧩 Graphical User Interface (GUI)
- Built with **Tkinter + ttk**
- Dark modern UI theme
- Image preview panel with live updates
- Activity log console (real-time training + prediction logs)
- Progress bar with ETA estimation
- Single-image and folder classification modes
- Class management (add / remove classes)

### ⚙️ System Design
- Threaded training and inference (non-blocking UI)
- Modular classifier architecture
- Frozen feature extractor for fast training
- Scalable design for future model upgrades

---

## 🛠 Technologies Used

- Python 3
- PyTorch
- torchvision
- PIL (Pillow)
- Tkinter / ttk
- threading
- json / os / pathlib

---

## ▶️ How to Run

1. Install Python 3.9+
2. Install dependencies:
   pip install torch torchvision pillow
3. Run the application:
   python main.py

---

## 📂 Project Structure

- main.py — main application file  
- images/ — default image folder (optional)  
- custom_classes/ — training data per class (auto-created)  
- custom_classes.json — class mapping (auto-generated)  
- custom_head.pth — trained model weights (auto-generated)  

---

## ⚠️ Notes
- Requires at least 2 classes before training
- Model improves with more varied training images
- First run loads pretrained ResNet-18 backbone automatically
- Training happens inside the app (no external scripts needed)

---

## 👤 Author

Your Name Here
