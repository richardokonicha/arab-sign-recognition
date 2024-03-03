````
# Gesture Recognition System

## Aim
The Gesture Recognition System is designed to analyze video data to recognize human gestures or activities using advanced computer vision and machine learning techniques. Utilizing the power of MediaPipe for holistic body landmark detection and TensorFlow for deep learning, this project aims to bridge the gap between human-computer interaction by enabling machines to understand human gestures in real-time.

## What It Does
This system processes video streams to detect and analyze human body, hand, and facial landmarks. By extracting these keypoints and feeding them into a trained LSTM (Long Short-Term Memory) neural network model, it classifies the gestures into predefined categories. This technology has a wide range of applications, from sign language interpretation to controlling devices through gestures.

## Installation

### Prerequisites
- Python 3.6+
- Pip
- Virtualenv (optional but recommended)

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Y4sserDev/ai-saudisignlang.git
   cd gesture-recognition-system
````

2. **Create and Activate Virtual Environment** (optional)

   - For macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - For Windows:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

To run the project, execute the main script after installing all dependencies. Ensure your webcam is accessible or modify the script to use a pre-recorded video

or run in jupyter

`jupyter-lab`
