import asyncio
import os
import logging
import time
from typing import Dict, Optional
from collections import defaultdict
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
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

        # Dictionary to store user model preferences
        self.user_models: Dict[int, str] = {}  # {user_id: model_name}

        # Available models
        self.available_models = {
            "nb": "Naive Bayes",
            "dBert": "Base DBert",
            "dBert_typos": "DBert Typos",
            "dBert_syns": "DBert Synonyms",
            "mBert": "Base MBert",
            "mBert_typos": "MBert Typos",
            "tBert": "Base TBert"
        }

        # Default model
        self.default_model = "nb"

        # Initialize Telegram application
        self.application = Application.builder() \
            .token(os.getenv("TELEGRAM_TOKEN")) \
            .build()

        # Register all handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("predict", self.predict))
        self.application.add_handler(CommandHandler("help", self.help))
        self.application.add_handler(CommandHandler("model", self.show_current_model))
        self.application.add_handler(CommandHandler("choose_model", self.choose_model))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_feedback)
        )

        self.last_activity = None
        self.is_sleeping = False
        self.sleep_timeout = 1800  # 30 minutes in seconds
        self.polling_task = None

        self.rate_limits = defaultdict(list)  # Track user request timestamps
        self.rate_lock = asyncio.Lock()  # Async lock for thread safety
        self.RATE_LIMIT = 5  # 5 requests
        self.TIME_WINDOW = 60  # 60 seconds

    async def activity_monitor(self):
        """Background task to manage sleep/wake states"""
        while True:
            await asyncio.sleep(60)  # Check every minute

            if self.last_activity and (time.time() - self.last_activity > self.sleep_timeout):
                if not self.is_sleeping:
                    logger.info("💤 Entering sleep mode to conserve resources")
                    await self.application.updater.stop()
                    self.is_sleeping = True
            else:
                if self.is_sleeping:
                    logger.info("🔋 Waking up from sleep mode")
                    await self.application.updater.start_polling()
                    self.is_sleeping = False

    async def record_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
        if self.is_sleeping:
            await self.wake_up()

    async def wake_up(self):
        """Force wake from sleep"""
        if self.is_sleeping:
            logger.info("🔔 Wake-up triggered by new activity")
            await self.application.updater.start_polling()
            self.is_sleeping = False
        self.last_activity = time.time()

    async def set_commands(self):
        """Set bot command suggestions"""
        commands = [
            ("start", "Start the bot"),
            ("predict", "Analyze text for sarcasm"),
            ("help", "Show help guide"),
            ("choose_model", "Select a different model"),
            ("model", "Show current model")
        ]
        await self.application.bot.set_my_commands(commands)

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send help instructions"""
        help_text = """
        🤖 <b>Sarcasm Detection Bot</b>

        <b>Commands:</b>
            /start - Start the bot
            /predict <i>text</i> - Analyze text for sarcasm (max 128 words)
            /choose_model - Select a different model
            /model - Show current model
            /help - Show this help message

        <b>Models:</b>
            nb - Naive Bayes (default)
            dBert_typos - DistillBert with Typos Augmentation
            dBert_syns - DistillBert with Synonym Augmentation
            dBert - Base DistillBert with No Augmentation
            mBert - Base MBert with No Augmentation
            mBert_typos - Base MBert with Typos Augmentation
            tBert - Base TBert with No Augmentation

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

    async def show_current_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show currently selected model"""
        user_id = update.effective_user.id
        current_model = self.user_models.get(user_id, self.default_model)
        model_name = self.available_models.get(current_model, "Unknown")
        await update.message.reply_text(f"Current model: {model_name}")

    async def choose_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show available models for selection"""
        # Create a list of KeyboardButton for each model
        model_buttons = [[KeyboardButton(model)] for model in self.available_models.values()]

        reply_markup = ReplyKeyboardMarkup(
            model_buttons,
            one_time_keyboard=True,
            resize_keyboard=True
        )
        await update.message.reply_text(
            "Please select a model:",
            reply_markup=reply_markup
        )

    async def set_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set selected model"""
        user_id = update.effective_user.id
        selected_model = update.message.text

        # Find matching model key
        for key, value in self.available_models.items():
            if value == selected_model:
                self.user_models[user_id] = key

                # Add warning for BERT models
                response_text = f"Model set to {selected_model}"
                if "dbert" in key.lower():  # Check if model name contains 'dbert'
                    response_text += "\n\n⚠️ Note: dBert is a big boy. Typical response time is ~18 seconds."

                await update.message.reply_text(
                    response_text,
                    reply_markup=ReplyKeyboardRemove()
                )
                return

        await update.message.reply_text(
            "Invalid model selection. Please use /choose_model to select from available options.",
            reply_markup=ReplyKeyboardRemove()
        )

    async def predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle prediction requests with retry logic"""
        user_id = update.effective_user.id
        text = ' '.join(context.args) if context.args else ""
        current_model = self.user_models.get(user_id, self.default_model)

        # Input validation
        if not text:
            await update.message.reply_text("Please provide text after /predict")
            return
        if len(text.split()) > 128:
            await update.message.reply_text("Please limit to 128 words")
            return

        # Retry configuration
        max_retries = 3
        # initial_timeout = 10
        # backoff_factor = 2

        # --- Rate Limiting Check ---
        async with self.rate_lock:
            now = time.time()
            # Cleanup old requests
            self.rate_limits[user_id] = [
                t for t in self.rate_limits[user_id]
                if now - t < self.TIME_WINDOW
            ]

            # Check if over limit
            if len(self.rate_limits[user_id]) >= self.RATE_LIMIT:
                wait_time = int(self.TIME_WINDOW - (now - self.rate_limits[user_id][0]))
                await update.message.reply_text(
                    f"⏳ Too many requests! Please wait {wait_time} seconds",
                    reply_markup=ReplyKeyboardRemove()
                )
                return

            # Record new request
            self.rate_limits[user_id].append(now)

        for attempt in range(max_retries):
            try:
                # Progressive timeout with backoff
                timeout = 18

                # Call Flask API with current timeout and model
                response = requests.post(
                    f"{os.getenv('API_URL')}/predict",
                    json={"text": text, "model": current_model},
                    timeout=timeout
                )

                # Handle HTTP errors
                if response.status_code == 502:
                    raise requests.exceptions.HTTPError("502 Bad Gateway (service may be waking up)")
                response.raise_for_status()

                data = response.json()

                # Validate API response structure
                if 'prediction' not in data:
                    raise ValueError("Invalid API response format")

                # Handle different response formats based on model type
                if current_model == "nb":
                    # Naive Bayes model
                    sarcasm_prob = data['prediction']
                else:
                    # DistilBERT models
                    sarcasm_prob = data['prediction']

                threshold = 0.5
                is_sarcasm = sarcasm_prob > threshold
                sarcasm_prob = sarcasm_prob if is_sarcasm else (1 - sarcasm_prob)
                confidence_score = sarcasm_prob * 100

                # Calculate response message
                response_msg = self._get_response(confidence_score, is_sarcasm)

                # Store interaction
                self.pending_interactions[user_id] = {
                    "raw_text": text,
                    "processed_text": data.get('processed_text', ''),
                    "confidence": sarcasm_prob,
                    "is_sarcasm": is_sarcasm,
                    "chat_id": update.effective_chat.id,
                    "model_used": current_model
                }

                # Send response
                await update.message.reply_text(
                    f"{response_msg}\n\n"
                    f"Confidence: {confidence_score:.1f}%\n"
                    f"Model: {self.available_models.get(current_model, 'Unknown')}"
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

    async def handle_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Process user feedback and save complete interaction"""
        user_id = update.effective_user.id
        if user_id not in self.pending_interactions:
            # Check if this is a model selection
            if update.message.text in self.available_models.values():
                await self.set_model(update, context)
                return
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
                "user_feedback": feedback,
                "model_used": interaction["model_used"]  # Add model to database
            }).execute()

            await update.message.reply_text(
                "✅ Thanks for your feedback! Please vote for our StEps26 team at https://uvents.nus.edu.sg/event/26th-steps/vote - CS4248, Team 15!",
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
            elif confidence >= 70:
                return "👍 Yeah, that's probably sarcasm"
            elif confidence >= 60:
                return "🤔 Might be sarcastic..."
            else:
                return "❓ Not entirely sure, but I'm leaning towards sarcastic"
        else:
            if confidence >= 80:
                return "✅ Seems genuine"
            elif confidence >= 70:
                return "💁‍♂️ Probably not sarcastic"
            else:
                return "⚖️ Could go either way, but I'm leaning towards not sarcastic..."

    def run(self):
        """Run the bot with smart sleep"""
        # Start activity monitor in background
        self.polling_task = self.application.run_polling()

        # Create event loop for background tasks
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run both bot and monitor
        tasks = [
            self.polling_task,
            loop.create_task(self.activity_monitor())
        ]

        try:
            loop.run_until_complete(asyncio.gather(*tasks))
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        finally:
            loop.close()

if __name__ == "__main__":
    bot = SarcasmBot()
    bot.run()