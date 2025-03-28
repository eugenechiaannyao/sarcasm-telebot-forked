import os
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)
from supabase import create_client
from dotenv import load_dotenv
import requests
from aiohttp import web  # Async web framework
import asyncio

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

# --- Bot Setup ---
application = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()


# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Hi! I'm a sarcasm detection bot.\n\n"
        "Use /predict followed by text to analyze:\n"
        "Example: /predict Oh great, another meeting..."
    )


async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = " ".join(context.args) if context.args else ""

        if not text:
            await update.message.reply_text("Please provide text after /predict")
            return

        # Call your Flask API (sync-to-async bridge)
        response = requests.post(
            f"{os.getenv('API_URL')}/predict",
            json={"text": text},
            timeout=5
        )
        data = response.json()

        sarcasm_prob = data['prediction'][0][1] * 100
        response_msg = get_response(sarcasm_prob)

        # Async Supabase insert
        await supabase.table("interactions").insert({
            "user_id": update.effective_user.id,
            "chat_id": update.effective_chat.id,
            "raw_text": text,
            "confidence": sarcasm_prob,
            "is_sarcasm": sarcasm_prob >= 50
        }).execute()

        await update.message.reply_text(f"{response_msg}\n\nConfidence: {sarcasm_prob:.1f}%")

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        await update.message.reply_text("⚠️ Error processing your request")

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
    return "❌ Doesn't seem sarcastic"

# --- Webhook Server ---
async def handle_webhook(request):
    """Async webhook handler"""
    try:
        json_data = await request.json()
        update = Update.de_json(json_data, application.bot)
        await application.process_update(update)
        return web.Response(text="OK")
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return web.Response(status=500)


async def setup_webhook():
    """Configure webhook on startup"""
    webhook_url = f"{os.getenv('WEBHOOK_URL')}/webhook"
    await application.bot.set_webhook(
        url=webhook_url,
        allowed_updates=["message", "callback_query"]
    )
    logger.info(f"Webhook configured at {webhook_url}")


# --- Startup ---
async def main():
    # Your existing setup code
    application = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()

    # Webhook mode
    if os.getenv("WEBHOOK_MODE"):
        await setup_webhook()

        # Create server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 8000)))
        await site.start()

        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down...")
            await site.stop()
            await runner.cleanup()
            await application.shutdown()
    else:
        # Polling mode
        await application.run_polling()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped")