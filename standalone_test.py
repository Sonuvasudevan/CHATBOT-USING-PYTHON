from chatbot_core import ChatBot

if __name__ == "__main__":
    chatbot = ChatBot(enable_logging=True, log_file="web_chat_logs.json")
    print("ChatBot Standalone Test")

    # Sample input/output
    sample_inputs = ["hi", "hello", "bye", "how are you?"]
    print("\nSample Input/Output:")
    for msg in sample_inputs:
        print(f"You: {msg}")
        print(f"Bot: {chatbot.get_response(msg)}")

    print("\nInteractive Mode:")
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in ["exit", "quit", "bye"]:
            print("Bot:", chatbot.get_response("bye"))
            break
        bot_response = chatbot.get_response(user_message)
        print("Bot:", bot_response)
