from flask import Flask, render_template, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load your data and model (copy your notebook logic here)
df = pd.read_csv('Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')
queries = df['instruction'].astype(str).tolist()
responses = df['response'].astype(str).tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')
query_embeddings = model.encode(queries, show_progress_bar=True)
dimension = query_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(query_embeddings).astype('float32'))

def get_response(user_input, top_k=1):
    user_emb = model.encode([user_input])
    D, I = index.search(np.array(user_emb).astype('float32'), top_k)
    return [responses[i] for i in I[0]]

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("message")
    # Check for greetings
    if user_input.strip().lower() in ["hi", "hello!"]:
        bot_response = "Hi! How can I help you?"
    elif user_input.strip().lower() in ["hello", "hi!"]:
        bot_response = "Hi! I am at your service?"
    elif user_input.strip().lower() in [ "hey"]:
        bot_response = "Hello! Feel free to ask anything regarding us?"
    else:
        bot_response = get_response(user_input, top_k=1)[0]
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
