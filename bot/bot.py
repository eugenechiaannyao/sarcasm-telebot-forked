import os
import logging
import joblib
from telegram import Update
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

# --- Setup ---
load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load ML model
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Initialize Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)


# --- Bot Logic ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "🤖 Hi! I'm your sarcasm detection bot.\n\n"
        "Use /predict followed by text to analyze for sarcasm!\n"
        "Example: /predict Oh great, another meeting..."
    )


async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /predict command"""
    try:
        text = " ".join(context.args) if context.args else ""

        if not text:
            await update.message.reply_text("Please provide text after /predict")
            return

        if len(text.split()) > 18:
            await update.message.reply_text("Please limit to 18 words")
            return

        # Vectorize and predict
        X = vectorizer.transform([text])
        proba = model.predict_proba(X)[0]
        confidence = proba[1] * 100  # Probability of sarcasm

        # Get response based on confidence
        response = get_sarcasm_response(confidence)
        await update.message.reply_text(response)

        # Save to database
        save_interaction(
            user_id=update.effective_user.id,
            chat_id=update.effective_chat.id,
            text=text,
            confidence=confidence,
            is_sarcasm=confidence >= 50
        )

        # Ask for feedback
        await update.message.reply_text(
            "So... Was it actually sarcasm? Reply with 'yes' or 'no'"
        )
        context.user_data['awaiting_feedback'] = True
        context.user_data['last_prediction'] = {
            'text': text,
            'confidence': confidence
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        await update.message.reply_text("⚠️ Error processing your request")


async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user feedback on predictions"""
    if context.user_data.get('awaiting_feedback'):
        feedback = update.message.text.lower()

        if feedback in ('yes', 'no'):
            # Update previous record with feedback
            update_feedback(
                user_id=update.effective_user.id,
                text=context.user_data['last_prediction']['text'],
                actual_sarcasm=(feedback == 'yes')
            )
            await update.message.reply_text("Thanks for your feedback! 🙏")
        else:
            await update.message.reply_text("Please reply with 'yes' or 'no'")

        context.user_data['awaiting_feedback'] = False


def get_sarcasm_response(confidence: float) -> str:
    """Generate response based on confidence level"""
    if confidence >= 85:
        return "🔥 Most definitely, absolutely, positively SARCASM."
    elif confidence >= 75:
        return "💯 If this is not sarcasm, I don't know what to say..."
    elif confidence >= 65:
        return "👍 Yes that is sarcasm if I'm a good sarcasm bot. mmmmm"
    elif confidence >= 55:
        return "🤔 Yeap, this should be sarcasm, I'm QUITE sure..."
    elif confidence >= 50:
        return "❓ I think this SHOULD be sarcasm, I can't say for sure"
    else:
        return "❌ Doesn't seem sarcastic to me."


def save_interaction(user_id: int, chat_id: int, text: str, confidence: float, is_sarcasm: bool):
    """Save interaction to Supabase"""
    supabase.table("interactions").insert({
        "user_id": user_id,
        "chat_id": chat_id,
        "text": text,
        "confidence": confidence,
        "predicted_sarcasm": is_sarcasm,
        "actual_sarcasm": None  # Will be updated with feedback
    }).execute()


def update_feedback(user_id: int, text: str, actual_sarcasm: bool):
    """Update record with user feedback"""
    supabase.table("interactions").update({
        "actual_sarcasm": actual_sarcasm
    }).eq("user_id", user_id).eq("text", text).execute()


# --- Main ---
def main():
    """Start the bot"""
    bot = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()

    # Add handlers
    bot.add_handler(CommandHandler("start", start))
    bot.add_handler(CommandHandler("predict", predict))
    bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_feedback))

    logger.info("Bot started polling...")
    bot.run_polling()


if __name__ == "__main__":
    main()