# Gestura.ai 🖐️
> **Real-Time AI Sign Language Interpreter**  
> *Bridging communication gaps with Computer Vision and Deep Learning.*

![Gestura Dashboard](https://via.placeholder.com/1200x600.png?text=Project+Gestura+Preview)
*(Add your own screenshot here later!)*

## 💡 Inspiration
Communication should be universal. **Gestura** was built to translate sign language gestures into spoken text in real-time, making digital interaction accessible for everyone. 

Built during a 48-hour hackathon, this project challenges the limits of browser-based Computer Vision by integrating **MediaPipe** hand tracking with a custom **Flask** backend that runs safely on standard hardware.

## 🚀 Features
- **Real-Time Detection:** Instantly recognizes hand gestures via webcam using a custom CNN model.
- **Smart Transcription:** Converts recognized symbols into full sentences with an intuitive chat interface.
- **Multilingual Speech:** Text-to-Speech (TTS) support for both **English** and **Hindi**.
- **Safe Threading:** Engineered with robust threading locks to prevent webcam race conditions in Flask.
- **Modern UI:** A clean, accessible dashboard designed for ease of use.

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Backend** | Python, Flask |
| **Computer Vision** | OpenCV, MediaPipe |
| **AI / ML** | TensorFlow, Keras (CNN Model) |
| **Frontend** | HTML5, CSS3 (Glassmorphism UI) |
| **Audio** | gTTS (Google Text-to-Speech) |

## ⚙️ Installation & Run

1. **Clone the Repository**
git clone https://github.com/Anshum-Sahoo/sign-language-translator-web.git
cd sign-language-translator-web

2. **Create a Virtual Environment**
python -m venv venv

Windows
.\venv\Scripts\activate

Mac/Linux
source venv/bin/activate

3. **Install Dependencies**
pip install -r requirements.txt

4. **Run the Application**
python app.py
*The app will start at `http://localhost:5000`*

## 🧩 How It Works (The "Vibe Code" Logic)
1. **Capture:** The app grabs video frames using a thread-safe loop in `app.py`.
2. **Process:** MediaPipe extracts 21 hand landmarks from each frame.
3. **Predict:** These landmarks are fed into a trained CNN model (`cnn8grps_rad1_model.h5`).
4. **Output:** The prediction is sent to the frontend via a JSON API, updating the chat bubble instantly.

## 🤝 Credits
- **Lead Developer:** [Anshum Sahoo](https://github.com/Anshum-Sahoo)
- **Backend Logic:** Special thanks to **[Devansh Git](https://github.com/devanshgit)** for the collaboration on threading architecture.

---
*Built with 💻 and ☕. Open Source for everyone.*
