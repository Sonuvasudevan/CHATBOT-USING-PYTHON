# Create a comprehensive basic chatbot for hackathon experimentation
# Let's start with the core chatbot class and functionality

chatbot_core_code = '''
"""
Basic Conversational Chatbot for Hackathon Experimentation
===========================================================

A modular, extensible chatbot framework using keyword matching and simple
natural language processing. Designed for easy extension with AI/ML features.

Author: Hackathon Participant
Version: 1.0
"""

import json
import re
import random
import datetime
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os

# Optional NLP imports - install with: pip install nltk spacy
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    
    # Download required NLTK data (run once)
    # nltk.download('punkt')
    # nltk.download('stopwords') 
    # nltk.download('wordnet')
    
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Using basic text processing.")

try:
    import spacy
    # Load English language model: python -m spacy download en_core_web_sm
    # nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy not available. Using basic text processing.")


@dataclass
class ConversationLog:
    """Data structure for logging conversations"""
    timestamp: str
    user_input: str
    bot_response: str
    intent: Optional[str] = None
    confidence: Optional[float] = None


class TextProcessor:
    """Handles text preprocessing and NLP features"""
    
    def __init__(self):
        self.lemmatizer = None
        self.stop_words = set()
        
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except:
                print("NLTK data not found. Run setup_nltk() method first.")
    
    def setup_nltk(self):
        """Download required NLTK data - run once"""
        if NLTK_AVAILABLE:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower().strip()
        # Remove special characters except basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\\s\\?\\.!,]', '', text)
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except:
                pass
        # Fallback: simple split
        return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove common English stopwords"""
        if self.stop_words:
            return [word for word in tokens if word.lower() not in self.stop_words]
        # Fallback: basic stopwords
        basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        return [word for word in tokens if word.lower() not in basic_stopwords]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Reduce words to their base forms"""
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(word) for word in tokens]
        # Fallback: basic stemming
        return [word.rstrip('s').rstrip('ed').rstrip('ing') for word in tokens]
    
    def process(self, text: str) -> List[str]:
        """Complete text processing pipeline"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return tokens


class IntentClassifier:
    """Simple intent classification using keyword matching"""
    
    def __init__(self):
        self.intents = self._load_intents()
    
    def _load_intents(self) -> Dict[str, Dict]:
        """Define intent patterns and responses"""
        return {
            'greeting': {
                'patterns': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
                'responses': [
                    "Hello! How can I help you today?",
                    "Hi there! What would you like to know?",
                    "Hey! I'm here to assist you.",
                    "Greetings! How may I assist you?"
                ],
                'confidence_threshold': 0.7
            },
            'goodbye': {
                'patterns': ['bye', 'goodbye', 'see you', 'farewell', 'exit', 'quit'],
                'responses': [
                    "Goodbye! Have a great day!",
                    "See you later!",
                    "Farewell! Thanks for chatting!",
                    "Bye! Come back anytime!"
                ],
                'confidence_threshold': 0.8
            },
            'how_are_you': {
                'patterns': ['how are you', 'how do you do', 'how are things', 'what\\'s up'],
                'responses': [
                    "I'm doing great! Thanks for asking. How are you?",
                    "I'm functioning perfectly! How can I help you today?",
                    "All systems operational! What brings you here?",
                    "I'm excellent! Ready to chat and help out!"
                ],
                'confidence_threshold': 0.6
            },
            'what_is_ai': {
                'patterns': ['what is ai', 'artificial intelligence', 'what is artificial intelligence', 'explain ai', 'ai definition'],
                'responses': [
                    "AI (Artificial Intelligence) is the simulation of human intelligence in machines that are programmed to think and learn like humans.",
                    "Artificial Intelligence refers to computer systems that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, and decision-making.",
                    "AI is a branch of computer science that aims to create intelligent machines capable of learning, reasoning, and problem-solving."
                ],
                'confidence_threshold': 0.8
            },
            'joke': {
                'patterns': ['tell me a joke', 'joke', 'make me laugh', 'something funny', 'humor'],
                'responses': [
                    "Why don't scientists trust atoms? Because they make up everything! 😄",
                    "I told my computer a joke about UDP... but it didn't get it. 😂",
                    "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
                    "How many programmers does it take to change a light bulb? None, that's a hardware problem! 💡"
                ],
                'confidence_threshold': 0.7
            },
            'help': {
                'patterns': ['help', 'what can you do', 'commands', 'options', 'features'],
                'responses': [
                    "I can help you with:\\n• General conversation and greetings\\n• Answer questions about AI\\n• Tell jokes\\n• Provide information on various topics\\n• Type 'help' anytime for this menu"
                ],
                'confidence_threshold': 0.8
            },
            'name': {
                'patterns': ['what is your name', 'who are you', 'your name', 'introduce yourself'],
                'responses': [
                    "I'm ChatBot, your friendly AI assistant!",
                    "You can call me ChatBot. I'm here to help and chat!",
                    "I'm ChatBot, a conversational AI built for hackathons and experimentation!"
                ],
                'confidence_threshold': 0.7
            },
            'weather': {
                'patterns': ['weather', 'temperature', 'forecast', 'climate'],
                'responses': [
                    "I don't have access to real-time weather data, but you can check your local weather service or weather apps for accurate forecasts!",
                    "For current weather information, I recommend checking weather.com or your local meteorological service.",
                    "I'd love to help with weather, but I don't have access to live weather data. Try a weather app or website!"
                ],
                'confidence_threshold': 0.6
            },
            'time': {
                'patterns': ['time', 'what time is it', 'current time', 'clock'],
                'responses': [
                    f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}",
                    f"Right now it's {datetime.datetime.now().strftime('%I:%M %p')}",
                    f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                ],
                'confidence_threshold': 0.8
            }
        }
    
    def calculate_similarity(self, user_input: str, pattern: str) -> float:
        """Calculate similarity between user input and intent pattern"""
        # Simple word overlap scoring
        user_words = set(user_input.lower().split())
        pattern_words = set(pattern.lower().split())
        
        if not pattern_words:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = len(user_words.intersection(pattern_words))
        union = len(user_words.union(pattern_words))
        
        return intersection / union if union > 0 else 0.0
    
    def classify_intent(self, user_input: str) -> Tuple[str, float]:
        """Classify user intent and return confidence score"""
        best_intent = 'unknown'
        best_confidence = 0.0
        
        for intent, data in self.intents.items():
            max_similarity = 0.0
            
            for pattern in data['patterns']:
                similarity = self.calculate_similarity(user_input, pattern)
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity > best_confidence:
                best_confidence = max_similarity
                best_intent = intent
        
        # Check if confidence meets threshold
        if best_intent != 'unknown':
            threshold = self.intents[best_intent]['confidence_threshold']
            if best_confidence < threshold:
                best_intent = 'unknown'
                best_confidence = 0.0
        
        return best_intent, best_confidence
    
    def get_response(self, intent: str) -> str:
        """Get a random response for the given intent"""
        if intent in self.intents:
            return random.choice(self.intents[intent]['responses'])
        return "I'm not sure how to respond to that. Type 'help' to see what I can do!"


class ConversationLogger:
    """Handles logging of conversations for analysis"""
    
    def __init__(self, log_file: str = "chat_logs.json"):
        self.log_file = log_file
        self.conversation_history = []
        self.setup_logging()
    
    def setup_logging(self):
        """Setup Python logging for system events"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('chatbot_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Chatbot logging system initialized")
    
    def log_conversation(self, user_input: str, bot_response: str, 
                        intent: str = None, confidence: float = None):
        """Log a conversation exchange"""
        log_entry = ConversationLog(
            timestamp=datetime.datetime.now().isoformat(),
            user_input=user_input,
            bot_response=bot_response,
            intent=intent,
            confidence=confidence
        )
        
        self.conversation_history.append(log_entry)
        self._save_to_file()
        
        self.logger.info(f"Conversation logged - Intent: {intent}, Confidence: {confidence}")
    
    def _save_to_file(self):
        """Save conversation history to JSON file"""
        try:
            # Convert dataclass objects to dictionaries
            log_data = [
                {
                    'timestamp': log.timestamp,
                    'user_input': log.user_input,
                    'bot_response': log.bot_response,
                    'intent': log.intent,
                    'confidence': log.confidence
                }
                for log in self.conversation_history
            ]
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save conversation log: {e}")
    
    def load_conversation_history(self):
        """Load previous conversation history from file"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversation_history = [
                        ConversationLog(**entry) for entry in data
                    ]
                    self.logger.info(f"Loaded {len(self.conversation_history)} previous conversations")
        except Exception as e:
            self.logger.warning(f"Could not load conversation history: {e}")
    
    def get_analytics(self) -> Dict:
        """Get conversation analytics"""
        if not self.conversation_history:
            return {"total_conversations": 0}
        
        intents = [log.intent for log in self.conversation_history if log.intent]
        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        confidences = [log.confidence for log in self.conversation_history if log.confidence]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "total_conversations": len(self.conversation_history),
            "intent_distribution": intent_counts,
            "average_confidence": round(avg_confidence, 3),
            "most_common_intent": max(intent_counts.items(), key=lambda x: x[1])[0] if intent_counts else None
        }


class ChatBot:
    """Main ChatBot class - orchestrates all components"""
    
    def __init__(self, enable_logging: bool = True, log_file: str = "chat_logs.json"):
        self.text_processor = TextProcessor()
        self.intent_classifier = IntentClassifier()
        self.enable_logging = enable_logging
        
        if enable_logging:
            self.logger = ConversationLogger(log_file)
            self.logger.load_conversation_history()
            print("💬 Conversation logging enabled")
        else:
            self.logger = None
        
        self.session_start = datetime.datetime.now()
        print("🤖 ChatBot initialized successfully!")
        print("💡 Type 'help' to see available commands")
        print("🔄 Type 'analytics' to see conversation statistics")
        print("🚪 Type 'quit' or 'exit' to end the session\\n")
    
    def process_message(self, user_input: str) -> str:
        """Process user message and return bot response"""
        # Handle special commands
        if user_input.lower() in ['quit', 'exit', 'bye']:
            return self._handle_goodbye()
        
        if user_input.lower() == 'analytics':
            return self._show_analytics()
        
        # Classify intent and generate response
        intent, confidence = self.intent_classifier.classify_intent(user_input)
        response = self.intent_classifier.get_response(intent)
        
        # Log conversation
        if self.enable_logging:
            self.logger.log_conversation(user_input, response, intent, confidence)
        
        return response
    
    def _handle_goodbye(self) -> str:
        """Handle goodbye messages and session termination"""
        session_duration = datetime.datetime.now() - self.session_start
        goodbye_msg = f"Thanks for chatting! Session duration: {str(session_duration).split('.')[0]}"
        
        if self.enable_logging:
            analytics = self.logger.get_analytics()
            goodbye_msg += f"\\nConversations this session: {analytics['total_conversations']}"
        
        return goodbye_msg
    
    def _show_analytics(self) -> str:
        """Show conversation analytics"""
        if not self.enable_logging:
            return "Analytics not available - logging is disabled."
        
        analytics = self.logger.get_analytics()
        
        analytics_msg = "📊 **Conversation Analytics**\\n"
        analytics_msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\\n"
        analytics_msg += f"Total Conversations: {analytics['total_conversations']}\\n"
        analytics_msg += f"Average Confidence: {analytics.get('average_confidence', 0):.1%}\\n"
        analytics_msg += f"Most Common Intent: {analytics.get('most_common_intent', 'N/A')}\\n"
        
        if analytics.get('intent_distribution'):
            analytics_msg += "\\nIntent Distribution:\\n"
            for intent, count in analytics['intent_distribution'].items():
                analytics_msg += f"  • {intent}: {count}\\n"
        
        return analytics_msg
    
    def chat_loop(self):
        """Run interactive chat session in terminal"""
        print("🚀 Starting chat session...")
        print("="*50)
        
        while True:
            try:
                user_input = input("\\n👤 You: ").strip()
                
                if not user_input:
                    print("🤖 Bot: Please enter a message!")
                    continue
                
                response = self.process_message(user_input)
                print(f"🤖 Bot: {response}")
                
                # Check for exit conditions
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                    
            except KeyboardInterrupt:
                print("\\n\\n👋 Chat session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("🔄 Please try again.")


# Extension point for AI/ML features
class AIExtensions:
    """Framework for adding advanced AI/ML features"""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.advanced_classifier = None
    
    def setup_sentiment_analysis(self):
        """Setup sentiment analysis (placeholder for future ML model)"""
        # Placeholder for sentiment analysis model
        # Could integrate with TextBlob, VADER, or custom model
        pass
    
    def setup_advanced_intent_classification(self):
        """Setup advanced ML-based intent classification"""
        # Placeholder for ML model (sklearn, transformers, etc.)
        # Could train on larger dataset with more sophisticated features
        pass
    
    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of user input"""
        # Placeholder implementation
        # In real implementation, would use trained model
        positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'like', 'awesome', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'sad', 'angry']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"


if __name__ == "__main__":
    # Initialize and run the chatbot
    chatbot = ChatBot(enable_logging=True)
    chatbot.chat_loop()
'''

# Save the main chatbot code
with open("chatbot_core.py", "w", encoding='utf-8') as f:
    f.write(chatbot_core_code)

print("✅ Created chatbot_core.py - Main chatbot implementation")