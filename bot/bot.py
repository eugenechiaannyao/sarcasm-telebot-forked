import os
import logging
from typing import Dict

from flask import Flask, request
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackContext
)
from supabase import create_client
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class SarcasmBot:
    def __init__(self):
        # Initialize Supabase client
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )

        # Dictionary to store pending interactions
        self.pending_interactions: Dict[int, dict] = {}  # {user_id: interaction_data}

        # Initialize Telegram application
        self.application = Application.builder() \
            .token(os.getenv("TELEGRAM_TOKEN")) \
            .build()

        # Register all handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("predict", self.predict))
        self.application.add_handler(CommandHandler("help", self.help))  # New
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_feedback)
        )

    async def set_commands(self):
        """Set bot command suggestions"""
        commands = [
            ("start", "Start the bot"),
            ("predict", "Analyze text for sarcasm"),
            ("help", "Show help guide")
        ]
        await self.application.bot.set_my_commands(commands)

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send help instructions"""
        help_text = """
        🤖 <b>Sarcasm Detection Bot</b>

        <b>Commands:</b>
            /predict <i>text</i> - Analyze text for sarcasm (max 18 words)
            /help - Show this message

        <b>Examples:</b>
            <code>/predict Oh great, another meeting</code>
            <code>/predict This is just what I needed</code>

        After each prediction, I'll ask if I was right. Your feedback helps me learn!
        """
        await update.message.reply_html(help_text)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send welcome message"""
        await update.message.reply_text(
            "🤖 Hi! I'm a sarcasm detection bot.\n\n"
            "Use /predict followed by text to analyze:\n"
            "Example: /predict Oh great, another meeting..."
        )

    async def predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle prediction requests"""
        user_id = update.effective_user.id
        try:
            text = ' '.join(context.args) if context.args else ""

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
            response.raise_for_status()
            data = response.json()

            # Process prediction
            print(data)
            sarcasm_prob = data['prediction'][0][1] * 100
            response_msg = self._get_response(sarcasm_prob)

            # Store interaction data temporarily
            self.pending_interactions[user_id] = {
                "raw_text": text,
                "processed_text": data.get('processed_text', ''),
                "confidence": sarcasm_prob,
                "is_sarcasm": sarcasm_prob >= 50,
                "chat_id": update.effective_chat.id
            }

            # Send response
            await update.message.reply_text(
                f"{response_msg}\n\nConfidence: {sarcasm_prob:.1f}%"
            )

            # Request feedback
            await update.message.reply_text(
                "Was this actually sarcasm?",
                reply_markup=ReplyKeyboardMarkup(
                    [["Yes", "No"]],
                    one_time_keyboard=True,
                    resize_keyboard=True
                )
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            await update.message.reply_text("⚠️ Error processing your request")
            self.pending_interactions.pop(user_id, None)

    async def handle_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Process user feedback and save complete interaction"""
        user_id = update.effective_user.id
        if user_id not in self.pending_interactions:
            await update.message.reply_text("❌ No pending prediction to give feedback on")
            return

        try:
            feedback = update.message.text.lower() == "yes"
            interaction = self.pending_interactions[user_id]

            # Insert complete record
            self.supabase.table("interactions").insert({
                "user_id": user_id,
                "chat_id": interaction["chat_id"],
                "raw_text": interaction["raw_text"],
                "processed_text": interaction["processed_text"],
                "confidence": interaction["confidence"],
                "is_sarcasm": interaction["is_sarcasm"],
                "user_feedback": feedback
            }).execute()

            await update.message.reply_text(
                "✅ Thanks for your feedback!",
                reply_markup=None  # This removes the keyboard
            )

        except Exception as e:
            logger.error(f"Feedback error: {e}")
            await update.message.reply_text("❌ Failed to save your feedback")
        finally:
            self.pending_interactions.pop(user_id, None)

    def _get_response(self, confidence: float) -> str:
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

    def run(self):
        """Run the bot"""
        app = Flask(__name__)

        if os.getenv("WEBHOOK_MODE"):
            app.run(host='0.0.0.0', port=int(os.getenv("PORT", 8000)))
        else:
            self.application.run_polling(
                close_loop=False,  # Critical for Railway
                stop_signals=[]
            )  # Fallback for local dev


if __name__ == "__main__":
    bot = SarcasmBot()
    bot.run()