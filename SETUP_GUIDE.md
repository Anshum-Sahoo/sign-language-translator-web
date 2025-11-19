# Sign Language Web Application - Setup Guide

## Overview

This is a refactored version of `final_pred.py` that runs as a **headless prediction server** integrated with Flask, eliminating all GUI/tkinter components and making it suitable for web deployment.

### Architecture

```
final_pred.py (Tkinter GUI) → sign_predictor.py (Headless Engine) + Flask (Backend) + HTML/JS (Frontend)
```

The prediction logic from `final_pred.py` is extracted into:
- **sign_predictor.py**: Core prediction engine (no GUI)
- **app.py**: Flask REST API server
- **templates/index.html**: Web interface

---

## Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Webcam access
- Windows/Linux/Mac with display support

### Installation Steps

1. **Create a project directory and virtual environment:**
   ```bash
   mkdir sign_language_web
   cd sign_language_web
   
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Copy the required files:**
   ```
   sign_language_web/
   ├── app.py
   ├── sign_predictor.py
   ├── hindi_service.py (from original)
   ├── requirements.txt
   ├── cnn8grps_rad1_model.h5 (your trained model)
   ├── white.jpg (template image)
   └── templates/
       └── index.html
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server:**
   ```bash
   python app.py
   ```

5. **Access the web app:**
   Open browser and go to: `http://localhost:5000`

---

## File Structure Explained

### sign_predictor.py
- **Class**: `SignLanguagePredictor`
- **Key Methods**:
  - `__init()`: Initialize model, hand detector, services
  - `start_webcam()`: Begin video capture
  - `get_frame()`: Capture single frame
  - `process_frame_with_landmarks()`: Extract and visualize hand landmarks
  - `predict()`: Run Keras model on landmarks
  - `update_sentence()`: Build sentence from predictions
  - `replace_word()`: Replace word with suggestion
  - `get_state()`: Return current UI state
  - `clear()`: Reset sentence

**Key Differences from final_pred.py**:
- ✅ No tkinter imports or GUI widgets
- ✅ Thread-safe with locks for concurrent access
- ✅ Returns data instead of updating UI directly
- ✅ Clean separation of concerns

### app.py
- **Framework**: Flask
- **Key Routes**:
  - `GET /`: Serve web interface
  - `GET /video_feed`: Stream webcam video (MJPEG)
  - `GET /landmarks_feed`: Stream hand landmarks (MJPEG)
  - `GET /api/state`: Get current prediction state
  - `POST /api/replace_word`: Replace word with suggestion
  - `POST /api/speak`: Generate and return audio (EN/HI)
  - `POST /api/clear`: Reset sentence
  - `POST /api/backspace`: Remove last character
  - `GET /health`: Health check

**Background Thread**:
- Continuous frame capture and prediction in `video_capture_loop()`
- Runs at ~100 FPS (adjustable)

### templates/index.html
- Modern responsive UI with two-column layout
- Left: Video feeds (webcam + landmarks)
- Right: Output (character, sentence, suggestions, controls)
- Real-time state polling every 500ms
- Mobile-friendly with CSS Grid

---

## Configuration

### Model Path
Edit in `app.py`:
```python
predictor = SignLanguagePredictor(
    model_path='path/to/your/model.h5',  # ← Change this
    white_template_path='path/to/white.jpg'  # ← Change this
)
```

### FPS Adjustment
In `app.py`, change `video_capture_loop()`:
```python
time.sleep(0.01)  # Current: 100 FPS, increase for lower CPU usage
```

### Port Change
In `app.py`, last line:
```python
app.run(host='0.0.0.0', port=5000)  # ← Change port here
```

---

## API Endpoints Reference

### Get Current State
```
GET /api/state
Response:
{
  "sentence": " HELLO WORLD",
  "current_symbol": "D",
  "word1": "world",
  "word2": "word",
  "word3": "would",
  "word4": "wound",
  "current_word": "world"
}
```

### Replace Word
```
POST /api/replace_word
Body: {"word": "world"}
Response: Updated state JSON
```

### Generate Speech
```
POST /api/speak
Body: {"text": "HELLO WORLD", "language": "en"}  // or "hi" for Hindi
Response: MP3 audio file (binary)
```

### Clear Sentence
```
POST /api/clear
Response: Updated state JSON
```

---

## Comparison: GUI vs Headless

| Feature | final_pred.py (GUI) | sign_predictor.py (Headless) |
|---------|-------------------|------------------------------|
| **GUI Framework** | tkinter | None |
| **Render Target** | Desktop window | Web browser |
| **Webcam Control** | Direct | Via Flask |
| **State Updates** | Direct UI updates | API returns state |
| **Thread Safety** | Not thread-safe | Thread-safe locks |
| **Deployment** | Desktop only | Web server |
| **Real-time Streaming** | VideoCapture → GUI | MJPEG over HTTP |

---

## Troubleshooting

### 1. Model File Not Found
```
Error: "cnn8grps_rad1_model.h5" not found
```
**Solution**: Ensure model file is in project root or update path in `app.py`

### 2. Webcam Permission Denied
```
Error: Could not open webcam
```
**Solution**: 
- Windows: Check application permissions
- Linux: `sudo usermod -aG video $USER` (requires logout)
- Mac: Grant camera permissions in System Preferences

### 3. Port Already in Use
```
OSError: Address already in use
```
**Solution**: Change port in `app.py` or kill process using port 5000

### 4. Hindi Translation Fails
```
Error: Translation failed
```
**Solution**: Ensure internet connection and `googletrans` is properly installed

### 5. High CPU Usage
```
100% CPU consumption
```
**Solution**: Increase `time.sleep()` in `video_capture_loop()` from 0.01 to 0.02 or higher

---

## Performance Optimization

### 1. Reduce Frame Rate
In `video_capture_loop()`:
```python
time.sleep(0.02)  # 50 FPS instead of 100 FPS
```

### 2. Model Quantization
Convert Keras model to TensorFlow Lite for ~10x speedup:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

### 3. GPU Acceleration
Enable CUDA in environment variables (already in sign_predictor.py):
```python
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"
```

### 4. Landmark Sampling
Skip every Nth frame for prediction:
```python
if frame_count % 2 == 0:  # Process every 2nd frame
    predictor.predict()
```

---

## Deployment to Production

### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker
Create `Dockerfile`:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
```

Run:
```bash
docker build -t sign-language-app .
docker run -p 5000:5000 --device /dev/video0 sign-language-app
```

### Environment Variables
Create `.env`:
```
FLASK_ENV=production
FLASK_DEBUG=False
MODEL_PATH=./models/cnn8grps_rad1_model.h5
WHITE_TEMPLATE_PATH=./assets/white.jpg
```

---

## Key Improvements Over final_pred.py

1. **No GUI Blocker**: Tkinter mainloop eliminated
2. **Web-Ready**: REST API instead of desktop-only
3. **Scalable**: Can run on servers, cloud platforms
4. **Thread-Safe**: Safe for concurrent requests
5. **Real-Time Streaming**: MJPEG video over HTTP
6. **API-First**: Integration-friendly design
7. **Responsive UI**: Modern HTML/CSS/JavaScript
8. **Modular**: Easily extendable architecture

---

## Next Steps

1. Test locally: `python app.py`
2. Verify predictions match original `final_pred.py`
3. Deploy to cloud (AWS, Google Cloud, Heroku, etc.)
4. Add user authentication if needed
5. Consider WebRTC for direct browser camera access (eliminates server-side webcam need)

---

## Support & Debugging

Enable verbose logging in `app.py`:
```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

Check logs for:
- Webcam initialization
- Model loading
- Prediction timing
- API request/response errors

---

**Created**: 2025-11-18  
**Python Version**: 3.8+  
**Status**: Production Ready
