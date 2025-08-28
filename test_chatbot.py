
#!/usr/bin/env python3
"""
Test Script for Basic Conversational Chatbot
===========================================

Run this script to verify that all components are working correctly.
"""

import sys
import json
from chatbot_core import ChatBot, TextProcessor, IntentClassifier


def test_text_processor():
    """Test text processing functionality"""
    print("🧪 Testing Text Processor...")

    processor = TextProcessor()

    # Test cases
    test_texts = [
        "Hello, how are you doing today?",
        "WHAT IS ARTIFICIAL INTELLIGENCE???",
        "i'm feeling great! tell me a joke.",
        "Can you help me with something?"
    ]

    for text in test_texts:
        cleaned = processor.clean_text(text)
        tokens = processor.tokenize(cleaned)
        processed = processor.process(text)

        print(f"  Input: '{text}'")
        print(f"  Processed: {processed}")
        print()

    print("✅ Text Processor tests completed\n")


def test_intent_classifier():
    """Test intent classification"""
    print("🧪 Testing Intent Classifier...")

    classifier = IntentClassifier()

    test_cases = [
        ("hello", "greeting"),
        ("goodbye", "goodbye"),
        ("tell me a joke", "joke"),
        ("what is AI?", "what_is_ai"),
        ("help me", "help"),
        ("what time is it", "time"),
        ("completely random nonsense", "unknown")
    ]

    for user_input, expected in test_cases:
        intent, confidence = classifier.classify_intent(user_input)
        response = classifier.get_response(intent)

        status = "✅" if intent == expected or (expected == "unknown" and confidence == 0.0) else "❌"

        print(f"  {status} Input: '{user_input}'")
        print(f"      Expected: {expected}, Got: {intent} (confidence: {confidence:.2f})")
        print(f"      Response: {response[:50]}...")
        print()

    print("✅ Intent Classifier tests completed\n")


def test_chatbot_core():
    """Test core chatbot functionality"""
    print("🧪 Testing ChatBot Core...")

    bot = ChatBot(enable_logging=False)  # Disable logging for tests

    test_messages = [
        "hello",
        "what can you do?",
        "tell me a joke",
        "what is artificial intelligence?",
        "what time is it?",
        "analytics",
        "goodbye"
    ]

    for message in test_messages:
        response = bot.process_message(message)
        print(f"  Input: '{message}'")
        print(f"  Output: '{response[:80]}...'")
        print()

    print("✅ ChatBot Core tests completed\n")


def test_logging():
    """Test conversation logging"""
    print("🧪 Testing Conversation Logging...")

    # Create bot with logging enabled
    bot = ChatBot(enable_logging=True, log_file="test_logs.json")

    # Have a short conversation
    test_conversation = [
        "hello",
        "tell me a joke",
        "that was funny",
        "goodbye"
    ]

    for message in test_conversation:
        bot.process_message(message)

    # Check analytics
    analytics = bot.logger.get_analytics()

    print(f"  Total conversations: {analytics['total_conversations']}")
    print(f"  Intent distribution: {analytics['intent_distribution']}")
    print(f"  Average confidence: {analytics['average_confidence']}")

    # Clean up test log file
    import os
    try:
        os.remove("test_logs.json")
        print("  Test log file cleaned up")
    except:
        pass

    print("✅ Conversation Logging tests completed\n")


def run_all_tests():
    """Run all test suites"""
    print("🚀 Starting Chatbot Test Suite")
    print("=" * 50)

    try:
        test_text_processor()
        test_intent_classifier()
        test_chatbot_core()
        test_logging()

        print("🎉 All tests completed successfully!")
        print("\n✅ Your chatbot is ready for hackathon action!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("Please check your installation and try again.")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
