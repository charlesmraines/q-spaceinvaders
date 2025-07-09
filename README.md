
# Space Invaders Q-Learning Agent

This project demonstrates a simple Q-learning agent trained to play Atari's **Space Invaders** using [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) and the Arcade Learning Environment (ALE). The agent uses visual preprocessing and a simplified 2-state space for learning.

## 🕹️ Features

- Classic **Space Invaders** via ALE
- Minimal Q-learning algorithm with epsilon-greedy exploration
- Basic screen preprocessing to detect invaders and shooter alignment
- Clean CLI visualization of the Q-table during training
- Live game rendering after training using OpenCV

---

## 🚀 Setup & Installation

Make sure you have Python 3.9+ installed. Then follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/space-invaders-qlearning.git
cd space-invaders-qlearning
```

### 2. Create a virtual environment (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Atari ROMs (once)
```bash
AutoROM --accept-license
```

---

## ▶️ Running the Code

```bash
python3 space_invaders.py
```

The script will:

1. **Train the Q-learning agent** over 1000 episodes (takes ~5–10 minutes depending on your system).
2. Display a **live-updating Q-table** every 100 episodes in the terminal.
3. After training, render the game in a window using OpenCV so you can **watch the agent play**.

Press `q` during the test phase to exit the game window.

---

## 🧠 How It Works

- The screen is preprocessed to extract a simplified **2-state representation**:
  - `State 0`: No invader aligned with the shooter
  - `State 1`: An invader is aligned with the shooter
- The agent selects among 6 discrete actions using a Q-table:
  - `LEFT`, `RIGHT`, `FIRE`, etc.
- Rewards are used to iteratively update the Q-values during training.

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`:

- gymnasium
- ale-py
- autorom
- numpy
- opencv-python
- tqdm
- tabulate
- matplotlib (optional, only if using the plot mode)

Install them using:

```bash
pip install -r requirements.txt
```

---

## 📝 Notes

- If you're running on a headless server or WSL, you may not see OpenCV windows unless you configure a virtual display.
- You can tune the number of episodes, learning rate, or state extraction logic to improve performance.

---

## 📷 Preview

![Q-table Terminal Output](assets/qtable_example.png)
*(Optional: add a screenshot of the Q-table or gameplay)*

---

## ✍️ Author

Charles Raines  
PhD Student in AI & Robotics  
[LinkedIn](https://www.linkedin.com/in/charles-m-raines)

---

## 📜 License

MIT License – free to use and modify.
