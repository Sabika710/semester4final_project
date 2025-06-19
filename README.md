# 💬 AI-Powered Customer Service Chatbot

A simple AI-based customer service chatbot built using **Flask**, **FAISS**, and **Sentence Transformers**. The bot provides responses to user queries using semantic similarity from a dataset of real-world customer support interactions.

---

## 📁 Project Structure

├── app.py # Main Flask application
├── template/chat.html # Frontend chat interface
├── static/style.css # Styling for the chat UI
├── Bitext_Sample_...csv # Dataset containing 27K+ Q&A pairs
├── chatbot_env/ # Conda environment for managing dependencies


---

## 🧠 How It Works

- Loads a CSV file of customer questions (`instruction`) and responses (`response`).
- Embeds all questions using the `all-MiniLM-L6-v2` model from `sentence-transformers`.
- Uses **FAISS** for fast semantic search to find the most relevant answer.
- The user query is compared against all existing questions, and the best-matching answer is returned.

---

## 🚀 Getting Started

### 1. Clone the Repository

bash
git clone https://github.com/your-username/customer-service-chatbot.git
cd customer-service-chatbot

### 2. Activate Environment
bash
Copy
Edit
conda activate env_chatbot

### 3. Install Dependencies
pip install flask pandas sentence-transformers faiss-cpu

### 4. Run the App
bash
Copy
Edit
python app.py
