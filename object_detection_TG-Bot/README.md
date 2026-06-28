# 🤖 YOLO Custom Object Detection Telegram Bot

A responsive, asynchronous Telegram bot that brings the custom-trained YOLOv8 object detection model straight to your chat client.

---

## 💡 Why This Bot Was Created

This bot was designed with a dual purpose in mind:
1. **Demonstrate Real-Life Model Integration**: Show how a deep learning model can be moved out of Jupyter notebooks and training scripts and into a practical, real-world utility that anyone can interact with.
2. **Educational Template for Custom Bots**: Provide a clear, step-by-step example of how to build and configure your own Telegram bots using Python and the `python-telegram-bot` framework.
3. **Foundation for Personal Projects**: Serve as a customizable base that you can easily adapt for your own target use cases and datasets once you understand the integration mechanics.

---

## 🚀 How to Run the Bot

### 1. Configure the Environment
Ensure you have created a `.env` file inside the `object_detection_TG-Bot` directory with your Telegram token:
```env
BOT_TOKEN=your_telegram_bot_token_here
```

### 2. Install Dependencies
Run the installation in your Python environment:
```bash
pip install -r requirements.txt
```

### 3. Run the Bot
Start the listener service:
```bash
python bot.py
```

---

## 🙋‍♂️ General Questions & Answers (FAQ)

### 📌 Q: How does the bot detect objects in my images?
**A:** When you send an image, the bot interceptor captures the file, downloads it locally, and processes it through the custom-trained YOLOv8 model (`my_model.pt`). The model detects target objects, predicts their bounding boxes, labels them, and runs an annotated overlay back to your chat.

### 📌 Q: Which specific objects is this bot trained to recognize?
**A:** The bot is specialized to detect **7 custom object classes**:
1. 📚 **Book**
2. ♟️ **Chess**
3. 🎨 **Colour** (Colors/drawing tools)
4. 🧴 **Favicol** (adhesive brand)
5. 🧪 **Glue**
6. 📱 **Mobile** (smartphones)
7. 🖊️ **Pen**

### 📌 Q: Why does the bot use compatibility monkey-patches at startup?
**A:** To ensure the codebase remains robust across changing library updates:
* **Python 3.13 Timezone Shim**: Resolves a compatibility issue between standard python 3.13 timezone representations and the `apscheduler` framework used internally by `python-telegram-bot`.
* **YOLO Layer Injection**: Resolves `AttributeError: Can't get attribute 'C3k2'` error by dynamically registering model classes when loading custom weights compiled under newer layers.

### 📌 Q: Can I run this bot inside a docker container or cloud server?
**A:** Yes! The bot is designed to run asynchronously and does not require a graphical user interface (unlike standard openCV window display calls). You can safely host it headless on a remote VPS, AWS instance, or Docker container.

### 📌 Q: How do I change the model weights being used?
**A:** You can change the target model path directly in `bot.py` or place your updated weights file as `/home/zoro0x1/hdd/MyPrograms/python/Yolo-project-09-07-25/my_model/my_model.pt`.

---

## 👨‍💻 Author

**Shivam Prasad**
* 🌐 **Portfolio**: [shivamprasad1001.in](https://shivamprasad1001.in)
* 🔗 **LinkedIn**: [shivamprasad1001](https://www.linkedin.com/in/shivamprasad1001)

