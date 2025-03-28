import os
import logging
import asyncio
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from flask import Flask, request, jsonify
from supabase import create_client
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Initialize Flask with async support
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Initialize Telegram bot
application = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()


# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message"""
    await update.message.reply_text(
        "🤖 Hi! I'm a sarcasm detection bot.\n\n"
        "Use /predict followed by text to analyze:\n"
        "Example: /predict Oh great, another meeting..."
    )


async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle prediction requests"""
    try:
        text = " ".join(context.args) if context.args else ""

        if not text:
            await update.message.reply_text("Please provide text after /predict")
            return

        if len(text.split()) > 18:
            await update.message.reply_text("Please limit to 18 words")
            return

        # Call Flask API
        response = requests.post(
            f"{os.getenv('API_URL')}/predict",
            json={"text": text},
            timeout=5
        )
        data = response.json()

        # Process response
        sarcasm_prob = data['prediction'][0][1] * 100
        response_msg = _get_response(sarcasm_prob)

        # Save to Supabase
        supabase.table("interactions").insert({
            "user_id": update.effective_user.id,
            "chat_id": update.effective_chat.id,
            "raw_text": text,
            "confidence": sarcasm_prob,
            "is_sarcasm": sarcasm_prob >= 50
        }).execute()

        await update.message.reply_text(
            f"{response_msg}\n\nConfidence: {sarcasm_prob:.1f}%"
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        await update.message.reply_text("⚠️ Error processing your request")


def _get_response(confidence: float) -> str:
    """Generate response based on confidence level"""
    if confidence >= 85:
        return "🔥 Absolutely sarcastic!"
    elif confidence >= 75:
        return "💯 Pretty sure this is sarcasm"
    elif confidence >= 65:
        return "👍 Likely sarcastic"
    elif confidence >= 50:
        return "🤔 Might be sarcastic..."
    else:
        return "❌ Doesn't seem sarcastic"


# --- Webhook Endpoints ---
@app.route('/webhook', methods=['POST'])
async def webhook():
    """Async webhook handler"""
    try:
        json_data = await request.get_json()
        update = Update.de_json(json_data, application.bot)
        await application.process_update(update)
        return 'OK'
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"status": "error"}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return 'OK', 200


# --- Startup ---
async def register_webhook():
    """Register webhook with Telegram"""
    if os.getenv("WEBHOOK_URL"):
        await application.bot.set_webhook(
            url=os.getenv("WEBHOOK_URL"),
            allowed_updates=["message", "callback_query"]
        )
        logger.info(f"Webhook registered at {os.getenv('WEBHOOK_URL')}")


def main():
    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("predict", predict))

    # Start based on environment
    if os.getenv("WEBHOOK_MODE"):
        port = int(os.getenv("PORT", 8000))

        # Run async tasks
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(register_webhook())

        app.run(host='0.0.0.0', port=port)
    else:
        application.run_polling()


if __name__ == '__main__':
    main()