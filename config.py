
"""
Configuration file for the Basic Conversational Chatbot
=======================================================

This file contains configuration settings that can be easily modified
without changing the core chatbot code.
"""

import os

class ChatBotConfig:
    """Configuration settings for the chatbot"""

    # Logging settings
    ENABLE_LOGGING = True
    LOG_FILE = "chat_logs.json"
    SYSTEM_LOG_FILE = "chatbot_system.log"
    LOG_LEVEL = "INFO"

    # Flask settings
    FLASK_DEBUG = True
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5000
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'hackathon_chatbot_secret_2024')

    # NLP settings
    ENABLE_NLTK = False  # Set to True if NLTK is installed
    ENABLE_SPACY = False  # Set to True if spaCy is installed

    # Intent classification settings
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6

    # Response settings
    MAX_MESSAGE_LENGTH = 500

    # Analytics settings
    ENABLE_ANALYTICS = True
    ANALYTICS_RETENTION_DAYS = 30

    # API settings (for future extensions)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
    WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY', None)

    # Chatbot personality settings
    BOT_NAME = "ChatBot"
    BOT_PERSONALITY = "friendly"  # friendly, professional, casual, humorous

    # Feature flags for experimental features
    ENABLE_SENTIMENT_ANALYSIS = False
    ENABLE_CONTEXT_MEMORY = False
    ENABLE_LEARNING_MODE = False


# Custom responses for different personalities
PERSONALITY_RESPONSES = {
    "friendly": {
        "greeting": [
            "Hello! How can I help you today? 😊",
            "Hi there! What would you like to know?",
            "Hey! I'm here to assist you.",
            "Greetings! How may I help you?"
        ],
        "unknown": [
            "I'm not quite sure about that. Could you try rephrasing?",
            "Hmm, I don't understand. Can you ask me something else?",
            "I'm still learning! Try asking about AI, jokes, or type 'help'."
        ]
    },
    "professional": {
        "greeting": [
            "Good day. How may I assist you?",
            "Hello. What information do you require?",
            "Greetings. How can I be of service?",
            "Welcome. What can I help you with today?"
        ],
        "unknown": [
            "I do not have information on that topic.",
            "That query is outside my current knowledge base.",
            "I cannot provide assistance with that request."
        ]
    },
    "casual": {
        "greeting": [
            "Hey! What's up?",
            "Yo! How's it going?",
            "What's happening?",
            "Hey there! What can I do for ya?"
        ],
        "unknown": [
            "No clue about that, sorry!",
            "That's not ringing any bells...",
            "Beats me! Try something else?"
        ]
    },
    "humorous": {
        "greeting": [
            "Hello there, human! Ready for some fun? 🤖",
            "Greetings, carbon-based life form!",
            "Hey! Warning: May contain traces of artificial intelligence.",
            "Hello! I'm like Siri, but with more personality and fewer features!"
        ],
        "unknown": [
            "Error 404: Clue not found! 🤷‍♂️",
            "My brain.exe has stopped working on that one.",
            "That's more mysterious than why humans need sleep!"
        ]
    }
}

# Extended intent patterns (can be loaded from external file)
EXTENDED_INTENTS = {
    "programming": {
        "patterns": ["python", "programming", "code", "coding", "software", "development", "algorithm"],
        "responses": [
            "I love talking about programming! Python is my favorite language. What would you like to know?",
            "Programming is fascinating! Are you working on any coding projects?",
            "Code is poetry! What programming topic interests you?"
        ]
    },
    "motivation": {
        "patterns": ["motivate me", "inspiration", "feeling down", "encouragement", "support"],
        "responses": [
            "You've got this! Every expert was once a beginner. 💪",
            "Remember: Progress, not perfection. You're doing great!",
            "Challenges are just opportunities in disguise. Keep going!",
            "Believe in yourself! You're capable of amazing things!"
        ]
    },
    "compliment": {
        "patterns": ["you are good", "you are smart", "you are helpful", "you are great", "you are awesome"],
        "responses": [
            "Thank you! That makes my circuits happy! 😊",
            "You're too kind! I'm just doing my best to help.",
            "Aww, thanks! You're pretty awesome yourself!",
            "That means a lot! I try to be helpful."
        ]
    }
}
