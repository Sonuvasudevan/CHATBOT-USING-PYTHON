
"""
Example AI/ML Extensions for the Basic Conversational Chatbot
=============================================================

This file demonstrates how to extend the chatbot with advanced AI/ML features.
These are examples and templates for future implementation.

Usage:
    from extensions import SentimentAnalyzer, MLIntentClassifier

    # Initialize extensions
    sentiment = SentimentAnalyzer()
    ml_classifier = MLIntentClassifier()

    # Use in chatbot
    mood = sentiment.analyze("I'm feeling great today!")
    intent = ml_classifier.predict("What's the weather like?")
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Placeholder imports - install these for real implementation
try:
    # For sentiment analysis
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    # For ML-based classification
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    # For more advanced NLP
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class SentimentAnalyzer:
    """
    Advanced sentiment analysis for user messages.
    Can be used to adapt bot responses based on user emotion.
    """

    def __init__(self):
        self.initialized = False
        self.setup()

    def setup(self):
        """Initialize sentiment analysis tools"""
        if TEXTBLOB_AVAILABLE:
            self.initialized = True
            print("✅ TextBlob sentiment analyzer ready")
        else:
            print("⚠️  TextBlob not available. Using rule-based sentiment.")

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text and return scores.

        Args:
            text (str): Text to analyze

        Returns:
            Dict with polarity (-1 to 1) and subjectivity (0 to 1)
        """
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'label': self._get_sentiment_label(blob.sentiment.polarity)
            }
        else:
            return self._rule_based_sentiment(text)

    def _get_sentiment_label(self, polarity: float) -> str:
        """Convert polarity score to label"""
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"

    def _rule_based_sentiment(self, text: str) -> Dict[str, float]:
        """Simple rule-based sentiment analysis fallback"""
        positive_words = [
            'good', 'great', 'excellent', 'awesome', 'wonderful', 'fantastic',
            'amazing', 'perfect', 'love', 'like', 'happy', 'excited', 'thrilled'
        ]
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad',
            'angry', 'frustrated', 'disappointed', 'annoyed', 'upset'
        ]

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        # Simple scoring
        total_words = len(text_lower.split())
        polarity = (pos_count - neg_count) / max(total_words, 1)

        return {
            'polarity': max(-1.0, min(1.0, polarity)),
            'subjectivity': 0.5,  # Default
            'label': self._get_sentiment_label(polarity)
        }


class MLIntentClassifier:
    """
    Machine Learning-based intent classification using scikit-learn.
    Provides more accurate intent detection than keyword matching.
    """

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = {}
        self.is_trained = False

    def create_training_data(self) -> Tuple[List[str], List[str]]:
        """
        Create training data for the ML model.
        In a real implementation, this would load from a larger dataset.
        """
        training_texts = [
            # Greetings
            "hello", "hi", "hey", "good morning", "good afternoon", "greetings",
            "howdy", "what's up", "yo", "hiya",

            # Farewells
            "goodbye", "bye", "see you", "farewell", "take care", "catch you later",
            "adios", "au revoir", "ciao", "until next time",

            # Questions about AI
            "what is artificial intelligence", "explain AI", "how does AI work",
            "what can AI do", "AI capabilities", "machine learning definition",
            "artificial intelligence explanation", "AI technology",

            # Help requests
            "help me", "I need help", "what can you do", "how do you work",
            "show me options", "list features", "available commands",

            # Jokes
            "tell me a joke", "make me laugh", "something funny", "humor",
            "comedy", "be funny", "joke please", "entertain me",

            # Time/Date
            "what time is it", "current time", "what's the time", "time please",
            "what day is it", "today's date", "current date",

            # Personal questions
            "who are you", "what's your name", "introduce yourself",
            "tell me about yourself", "your identity", "what are you"
        ]

        training_labels = [
            # Greetings (10)
            "greeting", "greeting", "greeting", "greeting", "greeting",
            "greeting", "greeting", "greeting", "greeting", "greeting",

            # Farewells (10)
            "goodbye", "goodbye", "goodbye", "goodbye", "goodbye",
            "goodbye", "goodbye", "goodbye", "goodbye", "goodbye",

            # AI questions (8)
            "what_is_ai", "what_is_ai", "what_is_ai", "what_is_ai",
            "what_is_ai", "what_is_ai", "what_is_ai", "what_is_ai",

            # Help requests (7)
            "help", "help", "help", "help", "help", "help", "help",

            # Jokes (8)
            "joke", "joke", "joke", "joke", "joke", "joke", "joke", "joke",

            # Time queries (7)
            "time", "time", "time", "time", "time", "time", "time",

            # Personal questions (6)
            "name", "name", "name", "name", "name", "name"
        ]

        return training_texts, training_labels

    def train(self):
        """Train the ML intent classifier"""
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn not available. Cannot train ML classifier.")
            return

        # Get training data
        texts, labels = self.create_training_data()

        # Create pipeline with TF-IDF vectorizer and Naive Bayes classifier
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
            ('clf', MultinomialNB())
        ])

        # Train the model
        self.model.fit(texts, labels)
        self.is_trained = True

        print(f"✅ ML Intent Classifier trained on {len(texts)} examples")

    def predict_intent(self, text: str) -> Tuple[str, float]:
        """
        Predict intent for given text using the trained ML model.

        Args:
            text (str): User input text

        Returns:
            Tuple of (intent, confidence_score)
        """
        if not self.is_trained:
            return "unknown", 0.0

        # Get prediction and probability
        prediction = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        confidence = max(probabilities)

        return prediction, confidence

    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if self.is_trained and SKLEARN_AVAILABLE:
            joblib.dump(self.model, filepath)
            print(f"✅ Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from disk"""
        if SKLEARN_AVAILABLE:
            try:
                self.model = joblib.load(filepath)
                self.is_trained = True
                print(f"✅ Model loaded from {filepath}")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")


class ContextManager:
    """
    Manages conversation context and memory across multiple exchanges.
    Allows the chatbot to remember previous topics and provide contextual responses.
    """

    def __init__(self, max_context_length: int = 10):
        self.conversation_history = []
        self.current_context = {}
        self.max_context_length = max_context_length
        self.topics = set()

    def add_exchange(self, user_input: str, bot_response: str, intent: str = None):
        """Add a conversation exchange to context memory"""
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'intent': intent
        }

        self.conversation_history.append(exchange)

        # Keep only recent context
        if len(self.conversation_history) > self.max_context_length:
            self.conversation_history.pop(0)

        # Extract topics
        if intent:
            self.topics.add(intent)

    def get_context_summary(self) -> str:
        """Get a summary of recent conversation context"""
        if not self.conversation_history:
            return "No previous context."

        recent_topics = list(self.topics)[-3:]  # Last 3 unique topics
        return f"Recent topics: {', '.join(recent_topics)}"

    def should_reference_context(self, current_input: str) -> bool:
        """Determine if current input should reference previous context"""
        context_triggers = [
            "what did we talk about", "earlier", "before", "previous",
            "you mentioned", "we discussed", "continuing", "also"
        ]

        return any(trigger in current_input.lower() for trigger in context_triggers)


class PersonalityEngine:
    """
    Adapts chatbot responses based on configured personality and user interaction style.
    """

    def __init__(self, personality: str = "friendly"):
        self.personality = personality
        self.user_profile = {
            'interaction_count': 0,
            'preferred_topics': [],
            'communication_style': 'neutral'
        }

    def adapt_response(self, base_response: str, user_sentiment: str) -> str:
        """Adapt response based on personality and user sentiment"""

        # Adjust for user sentiment
        if user_sentiment == "negative" and self.personality == "friendly":
            base_response = "I'm sorry you're feeling that way. " + base_response
        elif user_sentiment == "positive" and self.personality == "friendly":
            base_response = "I'm glad you're in good spirits! " + base_response

        # Add personality markers
        if self.personality == "humorous":
            base_response += " 😄"
        elif self.personality == "professional":
            base_response = base_response.replace("!", ".")

        return base_response

    def update_user_profile(self, intent: str, sentiment: str):
        """Update user profile based on interaction"""
        self.user_profile['interaction_count'] += 1

        if intent not in self.user_profile['preferred_topics']:
            self.user_profile['preferred_topics'].append(intent)

        # Keep only recent topics
        if len(self.user_profile['preferred_topics']) > 5:
            self.user_profile['preferred_topics'].pop(0)


# Example usage and integration
class ExtendedChatBot:
    """
    Extended chatbot that integrates all the AI/ML extensions.
    This shows how to combine the extensions with the base chatbot.
    """

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ml_classifier = MLIntentClassifier()
        self.context_manager = ContextManager()
        self.personality_engine = PersonalityEngine()

        # Train ML classifier
        self.ml_classifier.train()

    def process_advanced_message(self, user_input: str) -> Dict:
        """
        Process message using all available AI/ML extensions.

        Returns:
            Dict with response and analysis details
        """
        # Analyze sentiment
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(user_input)

        # Classify intent using ML
        intent, confidence = self.ml_classifier.predict_intent(user_input)

        # Check if we should reference context
        reference_context = self.context_manager.should_reference_context(user_input)

        # Generate base response (would integrate with main chatbot)
        base_response = f"I detected '{intent}' intent with {confidence:.2f} confidence."

        # Adapt response using personality engine
        adapted_response = self.personality_engine.adapt_response(
            base_response, 
            sentiment_result['label']
        )

        # Update context and user profile
        self.context_manager.add_exchange(user_input, adapted_response, intent)
        self.personality_engine.update_user_profile(intent, sentiment_result['label'])

        return {
            'response': adapted_response,
            'sentiment': sentiment_result,
            'intent': intent,
            'confidence': confidence,
            'context_used': reference_context
        }


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing AI/ML Extensions...")

    # Test sentiment analysis
    sentiment = SentimentAnalyzer()
    test_texts = ["I love this chatbot!", "This is terrible", "How are you?"]

    for text in test_texts:
        result = sentiment.analyze_sentiment(text)
        print(f"Text: '{text}' -> Sentiment: {result}")

    # Test ML intent classification
    if SKLEARN_AVAILABLE:
        classifier = MLIntentClassifier()
        classifier.train()

        test_inputs = ["Hi there!", "Tell me a joke", "What is AI?"]
        for inp in test_inputs:
            intent, conf = classifier.predict_intent(inp)
            print(f"Input: '{inp}' -> Intent: {intent} (confidence: {conf:.2f})")

    print("\n✅ Extension testing completed!")
