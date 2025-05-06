import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class FAQChatbot:
    def __init__(self, intents_file):
        """Initialize the chatbot with intents file."""
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open(intents_file).read())
        self.vectorizer = None
        self.questions = []
        self.labels = []
        self.responses = {}
        self.process_intents()
        self.train_model()
        
    def process_intents(self):
        """Process the intents data for training."""
        # Extract questions and labels from intents
        for intent in self.intents['intents']:
            self.responses[intent['tag']] = intent['responses']
            for pattern in intent['patterns']:
                self.questions.append(pattern)
                self.labels.append(intent['tag'])
    
    def preprocess_text(self, text):
        """Preprocess text by tokenizing and lemmatizing."""
        tokens = nltk.word_tokenize(text.lower())
        return ' '.join([self.lemmatizer.lemmatize(word) for word in tokens])
    
    def train_model(self):
        """Train the chatbot model using TF-IDF vectorization."""
        # Preprocess all questions
        processed_questions = [self.preprocess_text(q) for q in self.questions]
        
        # Create and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(processed_questions)
        
        # Transform questions to TF-IDF features
        self.question_vectors = self.vectorizer.transform(processed_questions)
        
        print("Model trained successfully!")
    
    def save_model(self, filename):
        """Save the trained model to a file."""
        model_data = {
            'vectorizer': self.vectorizer,
            'question_vectors': self.question_vectors,
            'labels': self.labels,
            'responses': self.responses
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load_model(cls, model_file):
        """Load a trained model from file."""
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        chatbot = cls.__new__(cls)
        chatbot.vectorizer = model_data['vectorizer']
        chatbot.question_vectors = model_data['question_vectors']
        chatbot.labels = model_data['labels']
        chatbot.responses = model_data['responses']
        chatbot.lemmatizer = WordNetLemmatizer()
        
        return chatbot
    
    def predict_intent(self, user_input, threshold=0.3):
        """Predict the intent of user input."""
        # Preprocess user input
        processed_input = self.preprocess_text(user_input)
        
        # Vectorize the input
        input_vector = self.vectorizer.transform([processed_input])
        
        # Calculate similarities
        similarities = cosine_similarity(input_vector, self.question_vectors)[0]
        
        # Find the index of highest similarity
        max_sim_idx = np.argmax(similarities)
        
        # Check if similarity is above threshold
        if similarities[max_sim_idx] >= threshold:
            return self.labels[max_sim_idx]
        else:
            return "unknown"
    
    def get_response(self, user_input):
        """Generate a response based on user input."""
        intent_tag = self.predict_intent(user_input)
        
        if intent_tag == "unknown":
            return "I'm sorry, I don't understand your question. Could you rephrase or ask something else?"
        
        # Return a random response for the predicted intent
        return random.choice(self.responses[intent_tag])
    
    def chat(self):
        """Run the chatbot in interactive mode."""
        print("Chatbot is ready! (type 'quit' to exit)")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("Chatbot: Goodbye!")
                break
            
            response = self.get_response(user_input)
            print(f"Chatbot: {response}")


# Example usage
if __name__ == "__main__":
    # You can either train a new model:
    # chatbot = FAQChatbot("intents.json")
    # chatbot.save_model("chatbot_model.pkl")
    
    # Or load a pre-trained model:
    # chatbot = FAQChatbot.load_model("chatbot_model.pkl")
    
    # For demo purposes, let's train a new model with a simple intents file
    chatbot = FAQChatbot("intents.json")
    chatbot.chat()