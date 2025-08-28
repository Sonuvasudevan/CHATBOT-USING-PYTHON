
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
    print("🔄 Press Ctrl+C to stop the server\n")

    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
