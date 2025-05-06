from flask import Flask, render_template, request, jsonify
from chatbot import FAQChatbot
import os

app = Flask(__name__)

# Initialize the chatbot
chatbot = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    response = chatbot.get_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    # Check if we have a model file
    if os.path.exists('chatbot_model.pkl'):
        chatbot = FAQChatbot.load_model('chatbot_model.pkl')
    else:
        # If not, train a new model
        chatbot = FAQChatbot('intents.json')
        chatbot.save_model('chatbot_model.pkl')
    
    app.run(debug=True)