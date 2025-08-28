# Create Flask web interface for the chatbot

flask_app_code = '''
"""
Flask Web Interface for Basic Conversational Chatbot
====================================================

A simple web interface for the chatbot using Flask, HTML, CSS, and JavaScript.
Provides a user-friendly chat interface accessible via web browser.

Usage:
    python flask_app.py

Then visit: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from chatbot_core import ChatBot
import json
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'hackathon_chatbot_secret_key_2024'

# Initialize chatbot instance
chatbot = ChatBot(enable_logging=True, log_file="web_chat_logs.json")

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages from the web interface"""
    try:
        # Get user message from request
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'error': 'Empty message',
                'response': 'Please enter a message!'
            }), 400
        
        # Process message through chatbot
        bot_response = chatbot.process_message(user_message)
        
        # Return response
        return jsonify({
            'user_message': user_message,
            'bot_response': bot_response,
            'timestamp': chatbot.session_start.isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'response': 'Sorry, I encountered an error processing your message.'
        }), 500

@app.route('/analytics')
def analytics():
    """Serve analytics page"""
    try:
        if chatbot.enable_logging:
            analytics_data = chatbot.logger.get_analytics()
            return jsonify(analytics_data)
        else:
            return jsonify({'error': 'Analytics not available - logging disabled'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'chatbot_initialized': True,
        'logging_enabled': chatbot.enable_logging
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Starting Flask Chatbot Web Interface...")
    print("💻 Visit: http://localhost:5000")
    print("📊 Analytics: http://localhost:5000/analytics")
    print("❤️  Health Check: http://localhost:5000/health")
    print("🔄 Press Ctrl+C to stop the server\\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
'''

# Save Flask app
with open("flask_app.py", "w", encoding='utf-8') as f:
    f.write(flask_app_code)

print("✅ Created flask_app.py - Flask web interface")

# Create HTML template for the chat interface
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hackathon ChatBot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }

        .chat-header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }

        .chat-header p {
            font-size: 14px;
            opacity: 0.9;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }

        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }

        .message.user {
            justify-content: flex-end;
        }

        .message.bot {
            justify-content: flex-start;
        }

        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
            line-height: 1.4;
        }

        .message.user .message-content {
            background: #667eea;
            color: white;
            border-bottom-right-radius: 4px;
        }

        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #e1e8ed;
            border-bottom-left-radius: 4px;
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
            margin: 0 10px;
            flex-shrink: 0;
        }

        .message.user .message-avatar {
            background: #667eea;
            order: 2;
        }

        .message.bot .message-avatar {
            background: #764ba2;
            order: 1;
        }

        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e1e8ed;
        }

        .chat-input-form {
            display: flex;
            gap: 10px;
        }

        .chat-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e1e8ed;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }

        .chat-input:focus {
            border-color: #667eea;
        }

        .send-button {
            padding: 12px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s;
        }

        .send-button:hover {
            transform: translateY(-2px);
        }

        .send-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .typing-indicator {
            display: none;
            align-items: center;
            margin: 10px 0;
        }

        .typing-indicator .dots {
            margin-left: 50px;
        }

        .typing-indicator .dot {
            height: 8px;
            width: 8px;
            background-color: #667eea;
            border-radius: 50%;
            display: inline-block;
            margin: 0 2px;
            animation: typing 1.4s infinite ease-in-out;
        }

        .typing-indicator .dot:nth-child(1) {
            animation-delay: -0.32s;
        }

        .typing-indicator .dot:nth-child(2) {
            animation-delay: -0.16s;
        }

        @keyframes typing {
            0%, 80%, 100% {
                transform: scale(0);
                opacity: 0.5;
            }
            40% {
                transform: scale(1);
                opacity: 1;
            }
        }

        .quick-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 15px;
            padding: 0 10px;
        }

        .quick-action-btn {
            padding: 8px 12px;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 15px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .quick-action-btn:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }

        .status-bar {
            padding: 8px 20px;
            background: #f8f9fa;
            border-top: 1px solid #e1e8ed;
            font-size: 12px;
            color: #666;
            text-align: center;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .chat-container {
                width: 95%;
                height: 95vh;
                border-radius: 10px;
            }

            .message-content {
                max-width: 85%;
            }

            .chat-header h1 {
                font-size: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🤖 Hackathon ChatBot</h1>
            <p>Your friendly AI assistant for experimentation</p>
        </div>

        <div class="quick-actions">
            <button class="quick-action-btn" onclick="sendQuickMessage('hello')">👋 Hello</button>
            <button class="quick-action-btn" onclick="sendQuickMessage('help')">❓ Help</button>
            <button class="quick-action-btn" onclick="sendQuickMessage('tell me a joke')">😄 Tell a joke</button>
            <button class="quick-action-btn" onclick="sendQuickMessage('what is AI?')">🤔 What is AI?</button>
            <button class="quick-action-btn" onclick="sendQuickMessage('analytics')">📊 Analytics</button>
        </div>

        <div class="chat-messages" id="chatMessages">
            <!-- Welcome message -->
            <div class="message bot">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    Hello! I'm your chatbot assistant. I can help with greetings, answer questions about AI, tell jokes, and much more! 
                    <br><br>
                    Try typing "help" to see what I can do, or use the quick action buttons above.
                </div>
            </div>
        </div>

        <div class="typing-indicator" id="typingIndicator">
            <div class="dots">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
            </div>
        </div>

        <div class="chat-input-container">
            <form class="chat-input-form" id="chatForm">
                <input type="text" id="chatInput" class="chat-input" 
                       placeholder="Type your message here..." 
                       autocomplete="off" maxlength="500">
                <button type="submit" class="send-button" id="sendButton">Send</button>
            </form>
        </div>

        <div class="status-bar">
            <span id="statusText">Ready to chat! 💬</span>
        </div>
    </div>

    <script>
        class ChatInterface {
            constructor() {
                this.chatMessages = document.getElementById('chatMessages');
                this.chatInput = document.getElementById('chatInput');
                this.sendButton = document.getElementById('sendButton');
                this.chatForm = document.getElementById('chatForm');
                this.typingIndicator = document.getElementById('typingIndicator');
                this.statusText = document.getElementById('statusText');
                
                this.setupEventListeners();
                this.chatInput.focus();
            }

            setupEventListeners() {
                this.chatForm.addEventListener('submit', (e) => {
                    e.preventDefault();
                    this.sendMessage();
                });

                this.chatInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.sendMessage();
                    }
                });

                // Auto-resize input
                this.chatInput.addEventListener('input', () => {
                    this.updateStatus(`Typing... (${this.chatInput.value.length}/500 characters)`);
                });

                this.chatInput.addEventListener('blur', () => {
                    this.updateStatus('Ready to chat! 💬');
                });
            }

            async sendMessage() {
                const message = this.chatInput.value.trim();
                
                if (!message) {
                    this.shake(this.chatInput);
                    return;
                }

                // Add user message to chat
                this.addMessage(message, 'user');
                this.chatInput.value = '';
                this.toggleInput(false);
                this.showTyping();

                try {
                    // Send message to backend
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ message: message })
                    });

                    const data = await response.json();

                    if (response.ok) {
                        // Add bot response
                        setTimeout(() => {
                            this.hideTyping();
                            this.addMessage(data.bot_response, 'bot');
                            this.toggleInput(true);
                            this.chatInput.focus();
                        }, 500); // Simulate thinking time
                    } else {
                        throw new Error(data.error || 'Unknown error');
                    }

                } catch (error) {
                    this.hideTyping();
                    this.addMessage(`Sorry, I encountered an error: ${error.message}`, 'bot');
                    this.toggleInput(true);
                    this.updateStatus('Error occurred. Please try again.');
                }
            }

            addMessage(content, sender) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                
                const avatar = sender === 'user' ? '👤' : '🤖';
                
                messageDiv.innerHTML = `
                    <div class="message-avatar">${avatar}</div>
                    <div class="message-content">${this.formatMessage(content)}</div>
                `;

                this.chatMessages.appendChild(messageDiv);
                this.scrollToBottom();
            }

            formatMessage(content) {
                // Convert newlines to <br> tags
                return content.replace(/\\n/g, '<br>');
            }

            showTyping() {
                this.typingIndicator.style.display = 'flex';
                this.updateStatus('Bot is typing...');
                this.scrollToBottom();
            }

            hideTyping() {
                this.typingIndicator.style.display = 'none';
                this.updateStatus('Ready to chat! 💬');
            }

            toggleInput(enabled) {
                this.chatInput.disabled = !enabled;
                this.sendButton.disabled = !enabled;
                
                if (enabled) {
                    this.chatInput.placeholder = "Type your message here...";
                } else {
                    this.chatInput.placeholder = "Please wait...";
                }
            }

            scrollToBottom() {
                setTimeout(() => {
                    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
                }, 100);
            }

            updateStatus(message) {
                this.statusText.textContent = message;
            }

            shake(element) {
                element.style.animation = 'none';
                element.offsetHeight; // Trigger reflow
                element.style.animation = 'shake 0.5s';
                
                setTimeout(() => {
                    element.style.animation = '';
                }, 500);
            }
        }

        // Quick message functionality
        function sendQuickMessage(message) {
            document.getElementById('chatInput').value = message;
            chatInterface.sendMessage();
        }

        // Initialize chat interface when page loads
        let chatInterface;
        
        document.addEventListener('DOMContentLoaded', () => {
            chatInterface = new ChatInterface();
        });

        // Add shake animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes shake {
                0%, 100% { transform: translateX(0); }
                25% { transform: translateX(-5px); }
                75% { transform: translateX(5px); }
            }
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>
'''

# Create templates directory and save HTML
os.makedirs('templates', exist_ok=True)
with open("templates/chat.html", "w", encoding='utf-8') as f:
    f.write(html_template)

print("✅ Created templates/chat.html - Web interface template")