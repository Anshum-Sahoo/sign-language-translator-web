

from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
from sign_predictor import SignLanguagePredictor
import threading
import time
import os
from datetime import datetime
import traceback
import sys

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
predictor = None
capture_thread = None
is_running = False
# Lock to prevent multiple requests from initializing the predictor simultaneously
init_lock = threading.Lock() 

debug_info = {
    'frame_count': 0,
    'error_count': 0,
    'last_error': None,
    'last_error_time': None
}

def init_predictor_safe():
    """Thread-safe initialization of the predictor"""
    global predictor, is_running, capture_thread, debug_info
    
    # Check if already initialized to save time
    if predictor is not None:
        return True

    # Acquire lock - if another thread is initializing, we wait here
    with init_lock:
        # Check again inside lock (Double-Check Locking Pattern)
        if predictor is not None:
            return True
            
        try:
            print("\n" + "="*60)
            print("[INIT] Creating SignLanguagePredictor instance...")
            print("="*60)
            
            # Create instance
            new_predictor = SignLanguagePredictor(
                model_path='cnn8grps_rad1_model.h5',
                white_template_path='white.jpg'
            )
            
            print("\n" + "="*60)
            print("[INIT] Starting webcam...")
            print("="*60)
            new_predictor.start_webcam()
            
            # Assign to global variable only after full success
            predictor = new_predictor
            is_running = True
            
            # Start capture thread
            capture_thread = threading.Thread(target=video_capture_loop, daemon=True)
            capture_thread.start()
            
            print("\n" + "="*60)
            print("✓ Predictor initialized and Capture Thread Started")
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            print(f"\n✗ Error initializing predictor: {e}")
            print(traceback.format_exc())
            debug_info['last_error'] = str(e)
            debug_info['last_error_time'] = datetime.now()
            return False

def video_capture_loop():
    """Background thread to continuously capture and process video"""
    global predictor, is_running, debug_info
    
    print("[CAPTURE] Video capture thread started")
    
    last_print_time = time.time()
    
    while is_running and predictor:
        try:
            # 1. Get Frame
            try:
                # Use internal method to get frame from camera
                frame, _ = predictor.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                    
                debug_info['frame_count'] += 1
            except Exception as e:
                print(f"[CAPTURE] ✗ Frame Read Error: {e}")
                time.sleep(0.5) # Wait longer on camera error
                continue
            
            # 2. Process Landmarks & Predict
            if frame is not None:
                try:
                    # Process frame (heavy operation)
                    landmarks_frame = predictor.process_frame_with_landmarks(frame)
                    
                    if landmarks_frame is not None:
                        pred = predictor.predict(landmarks_frame)
                        if pred:
                            predictor.update_sentence(pred)
                            
                except Exception as e:
                    debug_info['error_count'] += 1
                    # Don't print every single error to avoid spamming console
                    if debug_info['error_count'] % 50 == 0:
                        print(f"[CAPTURE] ✗ Processing Error: {e}")

            # Status logging
            current_time = time.time()
            if current_time - last_print_time > 10:
                print(f"[CAPTURE] Status - Frames: {debug_info['frame_count']}, Errors: {debug_info['error_count']}")
                last_print_time = current_time
            
            # Sleep to prevent CPU Hogging - CRITICAL for preventing crashes
            time.sleep(0.03) 
            
        except Exception as e:
            print(f"[CAPTURE] ✗ Critical Loop Error: {e}")
            time.sleep(1)
    
    print("[CAPTURE] Video capture thread stopped")

def generate_frames():
    """Generator to stream video frames as MJPEG"""
    while True:
        # If system isn't running, yield a blank or waiting frame
        if not is_running or not predictor:
            time.sleep(0.5)
            continue
            
        try:
            frame = predictor.get_current_frame()
            if frame is None:
                time.sleep(0.1)
                continue
                
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   frame_bytes + b'\r\n')
            
            # Control framerate of the stream
            time.sleep(0.04) # ~25 FPS
            
        except Exception as e:
            # Broken pipe (client disconnected) is normal
            break

def generate_landmarks_frames():
    """Generator to stream landmark visualization frames"""
    while True:
        if not is_running or not predictor:
            time.sleep(0.5)
            continue
            
        try:
            frame = predictor.get_landmarks_frame()
            if frame is None:
                time.sleep(0.1)
                continue
                
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   frame_bytes + b'\r\n')
            
            time.sleep(0.04)
            
        except Exception as e:
            break

@app.before_request
def before_request():
    """Initialize predictor safely on first request"""
    if predictor is None:
        init_predictor_safe()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/landmarks_feed')
def landmarks_feed():
    return Response(generate_landmarks_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def get_state():
    if not predictor:
        return jsonify({'error': 'Initializing...'}), 503 # 503 Service Unavailable
    
    try:
        # Use a non-blocking check if possible, or fast return
        state = predictor.get_state()
        return jsonify(state), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug')
def get_debug():
    if not predictor:
        return jsonify({'status': 'initializing', 'debug': debug_info})
    return jsonify({
        'status': 'running',
        'debug': debug_info,
        'predictor_state': predictor.get_state()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if not predictor: return jsonify({'error': 'Not ready'}), 503
    try:
        data = request.json
        if data and 'symbol' in data:
            predictor.update_sentence(data['symbol'])
        return jsonify(predictor.get_state()), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/replace_word', methods=['POST'])
def replace_word():
    if not predictor: return jsonify({'error': 'Not ready'}), 503
    try:
        data = request.json
        predictor.replace_word(data.get('word', ''))
        return jsonify(predictor.get_state()), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/clear', methods=['POST'])
def clear():
    if not predictor: return jsonify({'error': 'Not ready'}), 503
    predictor.clear()
    return jsonify(predictor.get_state()), 200

@app.route('/api/speak', methods=['POST'])
def speak():
    if not predictor: return jsonify({'error': 'Not ready'}), 503
    try:
        data = request.json
        text = data.get('text', '')
        language = data.get('language', 'en')
        
        if language == 'hi':
            hindi_text = predictor.hindi_service.to_hindi(text)
            audio_path = predictor.hindi_service.speak_hindi(hindi_text)
            return send_file(audio_path, mimetype='audio/mpeg')
        else:
            from gtts import gTTS
            import tempfile
            # Use tempfile to avoid permission issues
            fd, path = tempfile.mkstemp(suffix='.mp3')
            os.close(fd)
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(path)
            return send_file(path, mimetype='audio/mpeg')
            
    except Exception as e:
        print(f"Speak error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Try to initialize eagerly to catch errors before server starts
    print("Attempting eager initialization...")
    init_predictor_safe()
    
    print(f"Server starting at http://0.0.0.0:5000")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)