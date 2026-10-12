# This Service is Depriciated


import logging

from openai import OpenAI

from config import OPEN_API_KEY, orgnId
from prompts import OPENAI_CHAT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)
client = None
if OPEN_API_KEY:
    try:
        client = OpenAI(api_key=OPEN_API_KEY, organization=orgnId)
        client.models.list()
    except Exception as e:
        client = None
        logger.warning(f"OpenAI client initialization failed: {e}")
else:
    logger.warning("OpenAI API key not found. Set OPEN_API_KEY in your .env file.")

def openai_response(message: str) -> str:
    if not client or not OPEN_API_KEY:
        logger.error("OpenAI client not initialized - missing API key")
        return "🤖 OpenAI service is not configured. Please set your OPEN_API_KEY in the .env file."

    if not message or not message.strip():
        return "❓ Please provide a message to get a response."
    
    try:
        logger.info(f"Sending request to OpenAI API for message: {message[:50]}...")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": OPENAI_CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": message.strip()},
            ],
            max_tokens=1000,
            temperature=0.7,
        )
        
        ai_response = response.choices[0].message.content
        logger.info("Successfully received response from OpenAI API")
        return ai_response or "🤔 I couldn't generate a response. Please try again."
        
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        if "quota" in str(e).lower() or "429" in str(e):
            return "🚫 OpenAI API quota exceeded. Please check your billing details or try again later."
        elif "unauthorized" in str(e).lower() or "401" in str(e):
            return "🔑 OpenAI API key is invalid. Please check your API key configuration."
        elif "rate_limit" in str(e).lower():
            return "⏱️ Rate limit exceeded. Please wait a moment before trying again."
        else:
            return f"❌ OpenAI service temporarily unavailable: {str(e)}"

