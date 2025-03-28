import os
import logging
from telegram import Update
from telegram.ext import (
    Updater,  # Using synchronous Updater instead of async Application
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext,
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

# Initialize Supabase (synchronous)
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Initialize Flask
app = Flask(__name__)

# Initialize Telegram bot (synchronous)
updater = Updater(token=os.getenv("TELEGRAM_TOKEN"), use_context=True)
dispatcher = updater.dispatcher


# --- Synchronous Command Handlers ---
def start(update: Update, context: CallbackContext):
    """Send welcome message"""
    update.message.reply_text(
        "🤖 Hi! I'm a sarcasm detection bot.\n\n"
        "Use /predict followed by text to analyze:\n"
        "Example: /predict Oh great, another meeting..."
    )


def predict(update: Update, context: CallbackContext):
    """Handle prediction requests"""
    try:
        text = " ".join(context.args) if context.args else ""

        if not text:
            update.message.reply_text("Please provide text after /predict")
            return

        if len(text.split()) > 18:
            update.message.reply_text("Please limit to 18 words")
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
        response_msg = get_response(sarcasm_prob)

        # Save to Supabase
        supabase.table("interactions").insert({
            "user_id": update.effective_user.id,
            "chat_id": update.effective_chat.id,
            "raw_text": text,
            "confidence": sarcasm_prob,
            "is_sarcasm": sarcasm_prob >= 50
        }).execute()

        update.message.reply_text(
            f"{response_msg}\n\nConfidence: {sarcasm_prob:.1f}%"
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        update.message.reply_text("⚠️ Error processing your request")


def get_response(confidence: float) -> str:
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


# --- Webhook Endpoint (synchronous) ---
@app.route('/webhook', methods=['POST'])
def webhook():
    """Synchronous webhook handler"""
    try:
        json_data = request.get_json()
        update = Update.de_json(json_data, updater.bot)
        dispatcher.process_update(update)  # Synchronous processing
        return 'OK'
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"status": "error"}), 500


@app.route('/health')
def health():
    return 'OK', 200


# --- Startup ---
def register_webhook():
    """Register webhook synchronously"""
    if os.getenv("WEBHOOK_URL"):
        updater.bot.set_webhook(
            url=os.getenv("WEBHOOK_URL"),
            allowed_updates=["message", "callback_query"]
        )
        logger.info(f"Webhook registered at {os.getenv('WEBHOOK_URL')}")


def main():
    # Register handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("predict", predict))

    # Start based on environment
    if os.getenv("WEBHOOK_MODE"):
        port = int(os.getenv("PORT", 8000))
        register_webhook()

        # Start Flask in main thread
        app.run(host='0.0.0.0', port=port, threaded=True)
    else:
        updater.start_polling()
        updater.idle()  # Block until stopped


if __name__ == '__main__':
    main()