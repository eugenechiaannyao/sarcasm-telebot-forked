# Stable version v2
import asyncio
import os
import logging
from typing import Dict
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
        """Handle prediction requests with retry logic"""
        user_id = update.effective_user.id
        text = ' '.join(context.args) if context.args else ""

        # Input validation
        if not text:
            await update.message.reply_text("Please provide text after /predict")
            return
        if len(text.split()) > 18:
            await update.message.reply_text("Please limit to 18 words")
            return

        # Retry configuration
        max_retries = 3
        initial_timeout = 5
        backoff_factor = 2  # Exponential backoff (5s, 10s, 20s)

        for attempt in range(max_retries):
            try:
                # Progressive timeout with backoff
                timeout = initial_timeout * (backoff_factor ** attempt)

                # Call Flask API with current timeout
                response = requests.post(
                    f"{os.getenv('API_URL')}/predict",
                    json={"text": text},
                    timeout=timeout
                )

                # Handle HTTP errors
                if response.status_code == 502:
                    raise requests.exceptions.HTTPError("502 Bad Gateway (service may be waking up)")
                response.raise_for_status()

                data = response.json()

                # Validate API response structure
                if 'prediction' not in data or len(data['prediction'][0]) < 2:
                    raise ValueError("Invalid API response format")

                sarcasm_prob = data['prediction'][0][1]

                confidence_score, is_sarcasm = self.calculate_normalized_confidence(sarcasm_prob)

                response_msg = self._get_response(confidence_score, is_sarcasm)

                # Store interaction
                self.pending_interactions[user_id] = {
                    "raw_text": text,
                    "processed_text": data.get('processed_text', ''),
                    "confidence": sarcasm_prob,
                    "is_sarcasm": is_sarcasm,
                    "chat_id": update.effective_chat.id
                }

                # Send response
                await update.message.reply_text(
                    f"{response_msg}\n\nConfidence: {confidence_score:.1f}%"
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
                return  # Success - exit retry loop

            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                if attempt == max_retries - 1:  # Final attempt failed
                    logger.error(f"All retries exhausted for user {user_id}")
                    await update.message.reply_text(
                        "🔌 My brain is still booting up...\n"
                        "Please try again in 10-15 seconds!"
                    )
                else:
                    await asyncio.sleep(timeout)  # Backoff before retry

            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                await update.message.reply_text("⚠️ An unexpected error occurred")
                break  # Don't retry for non-network errors

        # Cleanup if all retries failed
        self.pending_interactions.pop(user_id, None)

    def calculate_normalized_confidence(self, raw_prob: float) -> tuple[float, bool]:
        """
        Calculate normalized confidence score and sarcasm determination
        Args:
            raw_prob: Raw probability from model (0.0-1.0)
        Returns:
            tuple[normalized_confidence (0-100), is_sarcasm (bool)]
        """
        sarcasm_prob = raw_prob * 100  # Convert to percentage

        # Determine if sarcastic
        is_sarcasm = sarcasm_prob > 50

        # Base confidence (distance from neutral)
        base_confidence = abs(sarcasm_prob - 50) * 2  # Converts 50-100 → 0-100

        # Linear scaling from neutral (50%) to certain (100%)
        if is_sarcasm:
            # Scale 55% → 65%, 100% → 95%
            confidence = 65 + (sarcasm_prob - 55) * 0.75
        else:
            # Scale 45% → 65%, 0% → 95% (inverse)
            confidence = 65 + (45 - sarcasm_prob) * 0.75

        # Cap at 95% to avoid absolute certainty
        return confidence, is_sarcasm

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

    def _get_response(self, confidence: float, is_sarcasm: bool) -> str:
        """Gentler response tiers for flattened confidence"""
        if is_sarcasm:
            if confidence >= 85:
                return "💯 That's definitely sarcasm"
            elif confidence >= 75:
                return "👍 Yeah, that's probably sarcastic"
            elif confidence >= 65:
                return "🤔 Might be sarcastic..."
            else:
                return "❓ Not entirely sure"
        else:
            if confidence >= 80:
                return "✅ Seems genuine"
            elif confidence >= 70:
                return "💁‍♂️ Probably not sarcastic"
            else:
                return "⚖️ Could go either way"

    def run(self):
        """Run the bot"""
        self.application.run_polling()


if __name__ == "__main__":
    bot = SarcasmBot()
    bot.run()