import json
import os

from datetime import datetime

class Logger:
	def __init__(self, log_file):
		self.log_file = log_file

	def get_analytics(self):
		if os.path.exists(self.log_file):
			with open(self.log_file, "r", encoding="utf-8") as f:
				try:
					logs = json.load(f)
				except Exception:
					logs = []
		else:
			logs = []
		return {
			"total_messages": len(logs),
			"messages": logs[-10:] if logs else []
		}

class ChatBot:
	def __init__(self, enable_logging=False, log_file=None):
		self.responses = {
			"hi": "Hello! How can I help you today?",
			"hello": "Hi there! What can I do for you?",
			"bye": "Goodbye! Have a great day!",
			"how are you": "I'm doing great, thank you for asking! How can I assist you?",
			"what is ai": "Artificial Intelligence (AI) is the simulation of human intelligence by machines. It includes capabilities like learning, reasoning, and self-correction.",
			"help": "I can help you with:\n- General greetings and conversation\n- Information about AI\n- Telling jokes\n- Basic assistance\nJust ask me anything!",
			"tell me a joke": "Why don't programmers like nature? It has too many bugs! 😄",
			"who are you": "I'm a friendly chatbot assistant created to help and chat with you!",
			"thank you": "You're welcome! Is there anything else I can help you with?",
			"analytics": "To view analytics, please visit the analytics page at /analytics",
		}
		self.enable_logging = enable_logging
		self.log_file = log_file
		self.session_start = datetime.now()
		self.logger = Logger(log_file) if enable_logging and log_file else None

	def get_response(self, message):
		message = message.lower().strip()
		
		# Exact match
		if message in self.responses:
			response = self.responses[message]
		else:
			# Partial match
			for key in self.responses:
				if key in message:
					response = self.responses[key]
					break
			else:
				# No match found
				if "?" in message:
					response = "That's an interesting question! While I try to help with basic queries, I might need more training to answer this specific question."
				elif any(word in message for word in ["thanks", "thank"]):
					response = "You're welcome! Feel free to ask me anything else!"
				else:
					response = "I'm not sure about that. Try asking for 'help' to see what I can do!"

		if self.enable_logging and self.log_file:
			self.log_message(message, response)
		return response

	def process_message(self, message):
		return self.get_response(message)

	def log_message(self, user_message, bot_response):
		log_entry = {
			"user": user_message,
			"bot": bot_response
		}
		if os.path.exists(self.log_file):
			with open(self.log_file, "r", encoding="utf-8") as f:
				try:
					logs = json.load(f)
				except Exception:
					logs = []
		else:
			logs = []
		logs.append(log_entry)
		with open(self.log_file, "w", encoding="utf-8") as f:
			json.dump(logs, f, indent=2)
