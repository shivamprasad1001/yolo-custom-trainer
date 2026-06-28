import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

# Patch timezones for apscheduler and tzlocal before importing telegram.ext
try:
    import tzlocal
    import pytz
    import datetime as dtm
    import apscheduler.util
    tzlocal.get_localzone = lambda: pytz.timezone('UTC')
    original_astimezone = apscheduler.util.astimezone
    def patched_astimezone(obj):
        if obj is dtm.timezone.utc:
            return pytz.utc
        if isinstance(obj, dtm.tzinfo) and not (hasattr(obj, 'localize') and hasattr(obj, 'normalize')):
            try:
                tz_name = str(obj)
                if hasattr(obj, 'key'):
                    tz_name = obj.key
                return pytz.timezone(tz_name)
            except Exception:
                return pytz.utc
        return original_astimezone(obj)
    apscheduler.util.astimezone = patched_astimezone
except Exception:
    pass

import cv2
from dotenv import load_dotenv
from telegram import KeyboardButton, ReplyKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import C2f, C3, Bottleneck
from ultralytics.nn.modules.conv import Conv
import ultralytics.nn.modules.block
import ultralytics.nn.modules

# Monkey patch YOLO classes for YOLOv11 compatibility with older ultralytics packages
class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class PSABlock(nn.Module):
    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C3k(C3):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2(C2f):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

class C2PSA(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class SCDown(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))

for name, cls in [
    ("Attention", Attention),
    ("PSABlock", PSABlock),
    ("C3k", C3k),
    ("C3k2", C3k2),
    ("C2PSA", C2PSA),
    ("SCDown", SCDown),
]:
    setattr(ultralytics.nn.modules.block, name, cls)
    setattr(ultralytics.nn.modules, name, cls)

# Set up logging configuration
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Resolve the paths relative to the current file
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR.parent / "my_model" / "train" / "weights" / "best.pt"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load the YOLO model
try:
    model = YOLO(str(MODEL_PATH))
    all_classes = model.names  # Get class index to name mapping
except Exception as e:
    logger.critical(f"Failed to load YOLO model from {MODEL_PATH}: {e}")
    sys.exit(1)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command."""
    msg = (
        "👋 Welcome! I’m your smart detection bot.\n\n"
        "📷 Send me a photo and I'll analyze it.\n"
        "📋 Use /menu to explore what I can do.\n\n"
        "💡 *I'm still under development — more powerful features are coming soon!*"
    )
    if update.message:
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /menu command, presenting questions as keyboard buttons."""
    buttons = [
        [KeyboardButton("📦 What objects can you detect?")],
        [KeyboardButton("🤖 Who created you?")],
        [KeyboardButton("🧠 What can you do?")],
        [KeyboardButton("🖼 Show my past results")]
    ]
    reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)
    if update.message:
        await update.message.reply_text(
            "Choose a question or send me a photo 📷",
            reply_markup=reply_markup
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text messages and keyboard interactions."""
    if not update.message or not update.message.text:
        return

    text = update.message.text.lower()
    user = update.effective_user
    if not user:
        return

    username = user.username if user.username else f"user_{user.id}"
    user_dir = DATA_DIR / username

    if "what objects" in text:
        class_list = ", ".join(all_classes.values())
        await update.message.reply_text(f"I can detect these objects:\n🧾 {class_list}")
    elif "who created" in text:
        await update.message.reply_text(
            "👨‍💻 I was created by *Shivam Prasad*, an AI developer and tech explorer.",
            parse_mode=ParseMode.MARKDOWN,
        )
    elif "what can you do" in text:
        await update.message.reply_text(
            "🧠 I can detect objects in images, label them, and return results. More features coming soon!"
        )
    elif "show my past" in text:
        if user_dir.exists():
            files = [f for f in os.listdir(user_dir) if f.startswith("output")]
            if files:
                latest_file = sorted(files)[-1]
                latest_file_path = user_dir / latest_file
                try:
                    with open(latest_file_path, "rb") as img:
                        await update.message.reply_photo(photo=img, caption="🖼 Your most recent result:")
                except Exception as e:
                    logger.error(f"Error reading historical result file {latest_file_path}: {e}")
                    await update.message.reply_text("🚫 Error loading your latest result image.")
            else:
                await update.message.reply_text("No results yet. Send me a photo to get started!")
        else:
            await update.message.reply_text(
                "I don't have any past results for you. Try sending an image first."
            )
    else:
        await update.message.reply_text("I didn't understand that. Use /menu to see options or send a photo.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle received photo messages by running them through YOLO and returning detections."""
    if not update.message or not update.message.photo:
        return

    try:
        user = update.effective_user
        if not user:
            return

        username = user.username if user.username else f"user_{user.id}"
        user_dir = DATA_DIR / username
        user_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = user_dir / f"input_{timestamp}.jpg"
        output_path = user_dir / f"output_{timestamp}.jpg"

        photo = update.message.photo[-1]
        
        # Download the file
        telegram_file = await photo.get_file()
        await telegram_file.download_to_drive(custom_path=input_path)

        # Run inference
        results = model(str(input_path))
        annotated = results[0].plot()
        cv2.imwrite(str(output_path), annotated)

        # Send back prediction image
        with open(output_path, "rb") as img:
            await update.message.reply_photo(photo=img)

        # Send list of detected classes
        classes = set(results[0].names[int(cls)] for cls in results[0].boxes.cls)
        class_list = ", ".join(classes) if classes else "No objects detected."
        await update.message.reply_text(f"✅ Detected: {class_list}")

    except Exception as e:
        logger.error(f"Error handling photo prediction: {e}", exc_info=True)
        if update.message:
            await update.message.reply_text(
                "🚫 *Server is currently offline or facing issues.*\n"
                "Please contact Shivam (@shivamprasad1001) for support.",
                parse_mode=ParseMode.MARKDOWN,
            )


def graceful_shutdown(signum: int, frame) -> None:
    """Handle termination signals cleanly."""
    logger.info(f"Received signal {signum}. Shutting down...")
    sys.exit(0)


def main() -> None:
    """Start the bot and listen for updates."""
    # Load environment variables
    load_dotenv(dotenv_path=BASE_DIR / ".env")
    
    token = os.getenv("BOT_TOKEN")
    if not token:
        logger.critical("BOT_TOKEN environment variable not set. Please check your .env file.")
        sys.exit(1)

    # Initialize application
    application = Application.builder().token(token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", menu))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Register shutdown signal handlers
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    logger.info("🤖 Bot started. Listening...")
    application.run_polling()


if __name__ == "__main__":
    main()