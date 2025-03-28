import os
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from supabase import create_client
from dotenv import load_dotenv
import requests

# --- Setup ---
load_dotenv()
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


# --- Bot Functions ---
def get_response(confidence: float) -> str:
    """Simple response generator"""
    if confidence >= 85:
        return "🔥 Absolutely sarcastic!"
    elif confidence >= 75:
        return "💯 Pretty sure this is sarcasm"
    elif confidence >= 65:
        return "👍 Likely sarcastic"
    elif confidence >= 50:
        return "🤔 Might be sarcastic..."
    return "❌ Doesn't seem sarcastic"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Send me text and I'll detect sarcasm!")


async def analyze_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text

        # Call your API
        response = requests.post(
            f"{os.getenv('API_URL')}/predict",
            json={"text": text},
            timeout=5
        )
        data = response.json()

        sarcasm_prob = data['prediction'][0][1] * 100
        await update.message.reply_text(
            f"{get_response(sarcasm_prob)}\nConfidence: {sarcasm_prob:.1f}%"
        )

        # Save to Supabase
        supabase.table("interactions").insert({
            "user_id": update.effective_user.id,
            "chat_id": update.effective_chat.id,
            "raw_text": text,
            "confidence": sarcasm_prob,
            "is_sarcasm": sarcasm_prob >= 50
        }).execute()

    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("⚠️ Something went wrong")


# --- Main ---
def main():
    # Create bot
    bot = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()

    # Add handlers
    bot.add_handler(CommandHandler("start", start))
    bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_text))

    # Start polling (that's it!)
    logger.info("Bot started polling...")
    bot.run_polling()


if __name__ == "__main__":
    main()