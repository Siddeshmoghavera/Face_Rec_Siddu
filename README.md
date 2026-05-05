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

-Starts webcam detection
- Displays bounding boxes
- Saves detected faces automatically

-Press ESC to exit.


## 🎥 Usage: Advanced Webcam Monitoring

The `facerec_from_webcam_faster.py` script is designed for production-like environments where you need to track guests or staff.

- **Face Tracking**: Assigns a unique `face_id` to each person in the frame and tracks them across frames.
- **Best-Frame Selection**: Only saves the highest confidence, least blurry image of a person within a configurable cooldown period.
- **Verification Queue**: Outputs a structured queue of faces that need manual review (useful for integrating with a Manager Dashboard).

*Captured images are saved with timestamps in the `examples/captured_faces/` directory.*
```