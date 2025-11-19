# hindi_service.py
from googletrans import Translator
from gtts import gTTS
import os
import platform
import subprocess

class HindiService:
    def __init__(self, audio_file="temp_hindi.mp3"):
        print("=== HindiService initialized ===")
        self.translator = Translator()
        self.audio_file = os.path.abspath(audio_file)
        print(f"Audio file will be saved at: {self.audio_file}")
    
    def to_hindi(self, english_text: str) -> str:
        print(f"Translating: '{english_text}'")
        try:
            result = self.translator.translate(english_text, src='en', dest='hi')
            print(f"Translation successful: '{result.text}'")
            return result.text
        except Exception as e:
            print(f"Translation failed: {e}")
            return english_text
    
    def speak_hindi(self, hindi_text: str):
        print(f"Generating Hindi speech for: '{hindi_text}'")
        try:
            if os.path.exists(self.audio_file):
                print(f"Removing old audio file: {self.audio_file}")
                os.remove(self.audio_file)
            
            print("Creating gTTS object...")
            tts = gTTS(text=hindi_text, lang='hi', slow=False)
            
            print(f"Saving to: {self.audio_file}")
            tts.save(self.audio_file)
            
            if not os.path.exists(self.audio_file):
                print("ERROR: Audio file was not created!")
                return False
            
            file_size = os.path.getsize(self.audio_file)
            print(f"Audio file created, size: {file_size} bytes")
            
            system = platform.system()
            print(f"Playing audio on {system} system...")
            
            if system == "Windows":
                cmd = f"start {self.audio_file}"
                print(f"Running command: {cmd}")
                os.system(cmd)
            elif system == "Darwin":
                subprocess.run(["afplay", self.audio_file], check=False)
            else:
                subprocess.run(["mpg321", self.audio_file], check=False)
            
            print("Audio playback command executed")
            return True
        except Exception as e:
            print(f"ERROR in speak_hindi: {e}")
            import traceback
            traceback.print_exc()
            return False
