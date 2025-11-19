# sign_predictor.py - FIXED VISUALIZATION
# Fixes: Landmarks stream hanging when no hand is detected
# Fixes: cvzone version compatibility

import numpy as np
import math
import cv2
from hindi_service import HindiService
import os
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import threading
import time
import traceback

# Try to import enchant for word suggestions
try:
    import enchant
    ENCHANT_AVAILABLE = True
except ImportError:
    ENCHANT_AVAILABLE = False
    print("Warning: pyenchant not found. Word suggestions will be disabled.")

class SignLanguagePredictor:
    def __init__(self, model_path='cnn8grps_rad1_model.h5', white_template_path='white.jpg'):
        print("="*60)
        print("[INIT] Initializing SignLanguagePredictor...")
        
        # 1. Load Model
        try:
            self.model = load_model(model_path)
            print(f"✓ Model loaded: {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

        # 2. Initialize Hand Detectors
        self.hd = HandDetector(maxHands=1)
        self.hd2 = HandDetector(maxHands=1)
        
        # 3. Initialize Services
        self.hindi_service = HindiService()
        
        if ENCHANT_AVAILABLE:
            try:
                self.ddd = enchant.Dict("en-US")
                print("✓ Enchant dictionary loaded")
            except Exception as e:
                print(f"⚠ Enchant error: {e}")
                self.ddd = None
        else:
            self.ddd = None

        # 4. Load White Template & Initialize Default Frames
        self.white_path = white_template_path
        if os.path.exists(white_template_path):
            self.white_template = cv2.imread(white_template_path)
        else:
            print("⚠ white.jpg not found, creating blank white image")
            self.white_template = np.ones((400, 400, 3), np.uint8) * 255
            
        # Initialize current frames with blank data immediately so stream works
        self.current_frame = np.zeros((480, 640, 3), np.uint8)
        self.current_landmarks_frame = self.white_template.copy()

        # 5. Initialize Variables
        self.offset = 29
        self.str = " "
        self.word = " "
        self.current_symbol = "C"
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "
        
        self.ct = {'blank': 0}
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" " for _ in range(10)]
        self.pts = None
        
        self.vs = None
        self.lock = threading.Lock()
        
        print("✓ Predictor initialized")
        print("="*60)

    def start_webcam(self):
        if self.vs is None:
            self.vs = cv2.VideoCapture(0)
            time.sleep(1) # Warmup

    def stop_webcam(self):
        if self.vs:
            self.vs.release()
            self.vs = None

    def get_frame(self):
        if not self.vs: return None, None
        ret, frame = self.vs.read()
        if ret:
            frame = cv2.flip(frame, 1)
            with self.lock:
                self.current_frame = frame.copy()
            return frame, None
        return None, None

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def process_frame_with_landmarks(self, frame):
        """
        Processes frame and UPDATES self.current_landmarks_frame always.
        Returns the white image (blank or with skeleton).
        """
        if frame is None: return None
        
        # Create a fresh white canvas for this frame
        white = self.white_template.copy()
        if white.shape != (400, 400, 3):
            white = cv2.resize(white, (400, 400))
            
        try:
            # 1. Find hands in main frame
            # cvzone compatibility check: some versions return (hands, img), others just hands
            hands_data = self.hd.findHands(frame, draw=False, flipType=True)
            
            # Unpack if tuple (newer cvzone versions)
            if isinstance(hands_data, tuple):
                hands = hands_data[0]
            else:
                hands = hands_data
            
            if hands and len(hands) > 0:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                # Safe crop with bounds checking
                h_img, w_img, _ = frame.shape
                y1, y2 = max(0, y - self.offset), min(h_img, y + h + self.offset)
                x1, x2 = max(0, x - self.offset), min(w_img, x + w + self.offset)
                
                image_crop = frame[y1:y2, x1:x2]
                
                if image_crop.size > 0:
                    # 2. Find hands in cropped image
                    hands_data_crop = self.hd2.findHands(image_crop, draw=False, flipType=True)
                    
                    # Unpack if tuple
                    if isinstance(hands_data_crop, tuple):
                        handz = hands_data_crop[0]
                    else:
                        handz = hands_data_crop
                    
                    if handz and len(handz) > 0:
                        hand_data = handz[0]
                        self.pts = hand_data['lmList'] # SAVE POINTS
                        
                        # Centering logic
                        os_val = ((400 - w) // 2) - 15
                        os1_val = ((400 - h) // 2) - 15
                        
                        # DRAW SKELETON
                        ranges = [(0,4), (5,8), (9,12), (13,16), (17,20)]
                        for r in ranges:
                            for t in range(r[0], r[1]):
                                cv2.line(white, (self.pts[t][0] + os_val, self.pts[t][1] + os1_val), 
                                        (self.pts[t + 1][0] + os_val, self.pts[t + 1][1] + os1_val), (0, 255, 0), 3)
                        
                        # Palm connections
                        palm_connections = [(5,9), (9,13), (13,17), (0,5), (0,17)]
                        for p in palm_connections:
                             cv2.line(white, (self.pts[p[0]][0] + os_val, self.pts[p[0]][1] + os1_val), 
                                             (self.pts[p[1]][0] + os_val, self.pts[p[1]][1] + os1_val), (0, 255, 0), 3)

                        # Joints
                        for i in range(21):
                            cv2.circle(white, (self.pts[i][0] + os_val, self.pts[i][1] + os1_val), 2, (0, 0, 255), 1)

            # ALWAYS update the frame, even if blank
            with self.lock:
                self.current_landmarks_frame = white.copy()
            
            return white
            
        except Exception as e:
            print(f"Process error: {e}")
            # On error, still update frame to prevent freeze
            with self.lock:
                self.current_landmarks_frame = white.copy()
            return white

    def predict(self, test_image):
        """Exact port of 'predict' function from final_pred.py"""
        if test_image is None or self.pts is None: return None
        
        try:
            white = test_image
            white = white.reshape(1, 400, 400, 3)
            
            prob = np.array(self.model.predict(white, verbose=0)[0], dtype='float32')
            ch1 = np.argmax(prob, axis=0)
            prob[ch1] = 0
            ch2 = np.argmax(prob, axis=0)
            prob[ch2] = 0
            ch3 = np.argmax(prob, axis=0)
            prob[ch3] = 0

            pl = [ch1, ch2]

            # --- LOGIC PORT ---
            l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
                 [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
                 [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
            if pl in l:
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 0

            l = [[2, 2], [2, 1]]
            if pl in l:
                if (self.pts[5][0] < self.pts[4][0]):
                    ch1 = 0

            l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                    ch1 = 2

            l = [[6, 0], [6, 6], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[8], self.pts[16]) < 52:
                    ch1 = 2

            l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                    ch1 = 3

            l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[4][0] > self.pts[0][0]:
                    ch1 = 3

            l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[2][1] + 15 < self.pts[16][1]:
                    ch1 = 3

            l = [[6, 4], [6, 1], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[4], self.pts[11]) > 55:
                    ch1 = 4

            l = [[1, 4], [1, 6], [1, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.distance(self.pts[4], self.pts[11]) > 50) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 4

            l = [[3, 6], [3, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[4][0] < self.pts[0][0]):
                    ch1 = 4

            l = [[2, 2], [2, 5], [2, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[1][0] < self.pts[12][0]):
                    ch1 = 4

            l = [[3, 6], [3, 5], [3, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] > self.pts[10][1]:
                    ch1 = 5

            l = [[3, 2], [3, 1], [3, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][1] + 17 > self.pts[20][1]:
                    ch1 = 5

            l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[4][0] > self.pts[0][0]:
                    ch1 = 5

            l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                    ch1 = 5

            l = [[5, 7], [5, 2], [5, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[3][0] < self.pts[0][0]:
                    ch1 = 7

            l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[6][1] < self.pts[8][1]:
                    ch1 = 7

            l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[18][1] > self.pts[20][1]:
                    ch1 = 7

            l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[5][0] > self.pts[16][0]:
                    ch1 = 6

            l = [[7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                    ch1 = 6

            l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[8], self.pts[16]) > 50:
                    ch1 = 6

            l = [[4, 6], [4, 2], [4, 1], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[4], self.pts[11]) < 60:
                    ch1 = 6

            l = [[1, 4], [1, 6], [1, 0], [1, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                    ch1 = 6

            l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0], [6, 3], [6, 4], [7, 5], [7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 1

            l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 1

            l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 1

            l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                    ch1 = 1

            l = [[4, 1], [4, 2], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.distance(self.pts[4], self.pts[11]) < 50) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 1

            l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                    ch1 = 1

            l = [[6, 6], [6, 4], [6, 1], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                    ch1 = 1

            l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                    ch1 = 1

            l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[4][0] < self.pts[5][0] + 15) and ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                    ch1 = 7

            l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                    ch1 = 1

            fg = 13
            l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and self.pts[0][0] + fg < self.pts[20][0]) and not (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                    ch1 = 1

            l = [[5, 0], [5, 5], [0, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                    ch1 = 1

            if ch1 == 0:
                ch1 = 'S'
                if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                    ch1 = 'A'
                if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                    ch1 = 'T'
                if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                    ch1 = 'E'
                if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                    ch1 = 'M'
                if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                    ch1 = 'N'

            if ch1 == 2:
                if self.distance(self.pts[12], self.pts[4]) > 42:
                    ch1 = 'C'
                else:
                    ch1 = 'O'

            if ch1 == 3:
                if (self.distance(self.pts[8], self.pts[12])) > 72:
                    ch1 = 'G'
                else:
                    ch1 = 'H'

            if ch1 == 7:
                if self.distance(self.pts[8], self.pts[4]) > 42:
                    ch1 = 'Y'
                else:
                    ch1 = 'J'

            if ch1 == 4:
                ch1 = 'L'

            if ch1 == 6:
                ch1 = 'X'

            if ch1 == 5:
                if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                    if self.pts[8][1] < self.pts[5][1]:
                        ch1 = 'Z'
                    else:
                        ch1 = 'Q'
                else:
                    ch1 = 'P'

            if ch1 == 1:
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 'B'
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'D'
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 'F'
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 'I'
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'W'
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]:
                    ch1 = 'K'
                if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'U'
                if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                    ch1 = 'V'
                if (self.pts[8][0] > self.pts[12][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'R'

            if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = " "

            if ch1 == 'E' or ch1=='Y' or ch1=='B':
                if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = "next"

            if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
                if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                    ch1 = 'Backspace'

            return ch1
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def update_sentence(self, ch1):
        """Logic from final_pred.py video_loop for sentence building"""
        try:
            if ch1 is None: return

            # Check for "next" (completion of character)
            if ch1 == "next" and self.prev_char != "next":
                if self.ten_prev_char[(self.count - 2) % 10] != "next":
                    # Backspace logic
                    if self.ten_prev_char[(self.count - 2) % 10] == "Backspace":
                        self.str = self.str[0:-1]
                    # Append character logic
                    elif self.ten_prev_char[(self.count - 2) % 10] != "Backspace":
                        self.str = self.str + self.ten_prev_char[(self.count - 2) % 10]
                else:
                    if self.ten_prev_char[(self.count - 0) % 10] != "Backspace":
                        self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]

            # Space logic
            if ch1 == " " and self.prev_char != " ":
                self.str = self.str + " "

            # State updates
            self.prev_char = ch1
            self.current_symbol = ch1
            self.count += 1
            self.ten_prev_char[self.count % 10] = ch1

            # Suggestion Logic
            self._update_suggestions()
            
        except Exception as e:
            print(f"Update error: {e}")

    def _update_suggestions(self):
        if len(self.str.strip()) != 0 and self.ddd:
            st = self.str.rfind(" ")
            word = self.str[st + 1:]
            self.word = word
            if len(word.strip()) != 0:
                if self.ddd.check(word):
                    # Word exists, get suggestions
                    sug = self.ddd.suggest(word)
                    self.word1 = sug[0] if len(sug) >= 1 else " "
                    self.word2 = sug[1] if len(sug) >= 2 else " "
                    self.word3 = sug[2] if len(sug) >= 3 else " "
                    self.word4 = sug[3] if len(sug) >= 4 else " "
                else:
                    # Word is wrong, suggest corrections
                    sug = self.ddd.suggest(word)
                    self.word1 = sug[0] if len(sug) >= 1 else " "
                    self.word2 = sug[1] if len(sug) >= 2 else " "
                    self.word3 = sug[2] if len(sug) >= 3 else " "
                    self.word4 = sug[3] if len(sug) >= 4 else " "
            else:
                self.word1 = self.word2 = self.word3 = self.word4 = " "

    def replace_word(self, new_word):
        try:
            with self.lock:
                if len(self.str.strip()) == 0: return
                st = self.str.rfind(" ")
                self.str = self.str[:st + 1] + new_word.upper()
                self._update_suggestions()
        except: pass

    def clear(self):
        with self.lock:
            self.str = " "
            self.word1 = self.word2 = self.word3 = self.word4 = " "
            self.current_symbol = "C"

    def get_state(self):
        with self.lock:
            return {
                'sentence': self.str,
                'current_symbol': self.current_symbol,
                'word1': self.word1,
                'word2': self.word2,
                'word3': self.word3,
                'word4': self.word4,
                'current_word': self.word
            }
    
    def get_current_frame(self):
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None
            
    def get_landmarks_frame(self):
        with self.lock:
            return self.current_landmarks_frame.copy() if self.current_landmarks_frame is not None else None