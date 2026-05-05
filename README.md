# Face_Rec_Siddu: Advanced Face Recognition System

A high-performance Python face recognition and detection system built for real-time monitoring, video analysis, and identity verification. This project extends standard implementations with optimized processing, intelligent storage, and quality filtering.

---

## 🚀 Key Features

- **Real-Time Webcam Recognition**
  - Optimized processing using frame skipping for higher FPS.

- **Intelligent Tiered Storage**
  - Automatically classifies detected faces into:
    - `known` → High confidence matches  
    - `verification` → Medium confidence (needs review)  
    - `unknown` → Low confidence or new faces  

- **Blur Filtering**
  - Uses Laplacian variance to discard low-quality (blurry) frames.

- **Image & Video Analysis**
  - Supports both static image and video processing with bounding boxes and confidence scores.

- **Precomputed Encodings**
  - Uses `encodings.pkl` for faster startup and recognition.

---

## 📋 Table of Contents

- Installation
- Quick Start
- Webcam Monitoring
- Image & Video Processing
- Project Structure
- License

---

## 💻 Installation

```bash
# Clone repository
git clone https://github.com/Siddeshmoghavera/Face_Rec_Siddu.git
cd Face_Rec_Siddu

# Create virtual environment (recommended)
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


## ⚠️ Requirements

- Requires `opencv-python`, `dlib`, and `face_recognition`
- On Windows, install Visual C++ Build Tools if `dlib` installation fails

---
## ⚡ Quick Start

```bash
python examples/facerec_from_webcam_faster.py