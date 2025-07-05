from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain_core.exceptions import OutputParserException
import os
import time

app = Flask(__name__)

# ==== API Keys and Models ====
GEMINI_API_KEYS = [
    "AIzaSyD6SBsvAeO8nuaNiT4SLCI27cCzaR4-tMk",
    "AIzaSyAzOJqtVxh6OHRMzoV8AeU1_CALdkBskKg",
]

GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash"
]

# Global index trackers
current_key_index = 0
current_model_index = 0

def get_llm(api_key, model_name):
    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.7,
        convert_system_message_to_human=True
    )

def try_chat_with_failover(user_message):
    global current_key_index, current_model_index

    max_attempts = len(GEMINI_API_KEYS) * len(GEMINI_MODELS)
    attempts = 0

    while attempts < max_attempts:
        api_key = GEMINI_API_KEYS[current_key_index]
        model = GEMINI_MODELS[current_model_index]

        try:
            llm = get_llm(api_key, model)
            response = llm.invoke([HumanMessage(content=user_message)])
            return response.content

        except (OutputParserException, ValueError):
            current_model_index = (current_model_index + 1) % len(GEMINI_MODELS)

        except Exception as e:
            current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
            current_model_index = 0

        attempts += 1
        time.sleep(1)

    return "❌ All API keys and models failed. Try again later."


# ==== Flask Route ====
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "Missing 'message' in request."}), 400

    reply = try_chat_with_failover(user_message)
    return jsonify({"reply": reply})


# ==== Run Server ====
if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=10000)
