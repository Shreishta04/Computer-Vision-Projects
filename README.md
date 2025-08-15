Computer Vision Projects – Real-Time Detection and Analysis

This repository contains three Python-based real-time computer vision applications using OpenCV, MediaPipe, DeepFace, and NumPy.
Each project uses webcam input for live detection, processing, and visualization.

Projects Overview

1. Face Mesh Detection
   Technologies: OpenCV, MediaPipe
   Detects and visualizes 468 facial landmarks in real-time using MediaPipe's Face Mesh model.
   Landmarks are drawn as small green circles for clear visualization.

   Features:
   - Real-time face mesh detection.
   - Displays 3D face landmark points projected on the camera feed.
   - Lightweight and fast.

2. Real-Time Emotion Recognition
   Technologies: OpenCV, DeepFace
   Uses DeepFace to analyze and detect the dominant emotion of the user in real-time.

   Features:
   - Detects emotions like happy, sad, angry, surprised, etc.
   - Displays the dominant emotion as text overlay.
   - Works with enforce_detection=False for robustness.

3. Hand Distance Measurement with Proximity Alert
   Technologies: OpenCV, MediaPipe, NumPy, winsound (Windows)
   Detects hand position and calculates its distance from the camera using a known palm width and calibrated focal length.

   Features:
   - Real-time hand detection and tracking.
   - Distance calculation based on perceived palm size.
   - Alerts with a beep sound if hand is closer than a set threshold.

Installation & Setup

1. Clone this repository:
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

2. Install dependencies:
   pip install opencv-python mediapipe deepface numpy

Running the Projects
- Face Mesh Detection: python face_mesh.py
- Emotion Recognition: python emotion_recognition.py
- Hand Distance Measurement: python hand_distance.py

Requirements
- Python 3.8+
- Webcam
- Packages: opencv-python, mediapipe, deepface, numpy, winsound (Windows only)

Notes
- Press 'q' to exit any project.
- Adjust KNOWN_WIDTH and FOCAL_LENGTH in the hand distance script for accurate results.
