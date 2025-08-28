
#!/usr/bin/env python3
"""
Terminal Chat Interface for Basic Conversational Chatbot
========================================================

Simple script to run the chatbot in terminal mode for quick testing.

Usage:
    python chat_terminal.py

    or

    python chat_terminal.py --personality casual
    python chat_terminal.py --no-logging
    python chat_terminal.py --help
"""

import argparse
import sys
from chatbot_core import ChatBot
from config import ChatBotConfig


def create_arg_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Basic Conversational Chatbot - Terminal Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python chat_terminal.py                     # Run with default settings
    python chat_terminal.py --no-logging       # Disable conversation logging
    python chat_terminal.py --personality casual  # Use casual personality
    python chat_terminal.py --log-file my_chat.json  # Custom log file
        """
    )

    parser.add_argument(
        '--personality', 
        choices=['friendly', 'professional', 'casual', 'humorous'],
        default='friendly',
        help='Set chatbot personality (default: friendly)'
    )

    parser.add_argument(
        '--no-logging',
        action='store_true',
        help='Disable conversation logging'
    )

    parser.add_argument(
        '--log-file',
        default='chat_logs.json',
        help='Custom log file path (default: chat_logs.json)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Basic Conversational Chatbot v1.0'
    )

    return parser


def print_welcome():
    """Print welcome message"""
    print("=" * 60)
    print("🤖 BASIC CONVERSATIONAL CHATBOT")
    print("=" * 60)
    print("Welcome to your hackathon chatbot!")
    print("This is a modular, extensible chatbot framework.")
    print()
    print("✨ Features:")
    print("  • Natural language processing")
    print("  • Intent classification")
    print("  • Conversation logging")
    print("  • Extensible architecture")
    print("  • Multiple personalities")
    print()
    print("💡 Try these commands:")
    print("  • 'hello' - Greet the bot")
    print("  • 'help' - See available features")
    print("  • 'tell me a joke' - Get a programming joke")
    print("  • 'what is AI?' - Learn about artificial intelligence")
    print("  • 'analytics' - View conversation statistics")
    print("  • 'quit' - Exit the chat")
    print()


def print_settings(args):
    """Print current settings"""
    print("⚙️  Current Settings:")
    print(f"   Personality: {args.personality}")
    print(f"   Logging: {'Disabled' if args.no_logging else 'Enabled'}")
    if not args.no_logging:
        print(f"   Log file: {args.log_file}")
    print()


def main():
    """Main function to run the terminal chatbot"""

    # Parse command line arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    # Print welcome message
    print_welcome()
    print_settings(args)

    try:
        # Initialize chatbot with settings
        enable_logging = not args.no_logging

        print("🔄 Initializing chatbot...")
        chatbot = ChatBot(
            enable_logging=enable_logging,
            log_file=args.log_file
        )

        # Set personality (this would be implemented in a more advanced version)
        # chatbot.set_personality(args.personality)

        print(f"✅ Chatbot ready with {args.personality} personality!")
        print()

        # Start chat loop
        chatbot.chat_loop()

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for chatting!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error initializing chatbot: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
