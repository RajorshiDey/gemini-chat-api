from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain_core.exceptions import OutputParserException
import os
import time
from flask_cors import CORS



app = Flask(__name__)

CORS(app)

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

    total_keys = len(GEMINI_API_KEYS)
    total_models = len(GEMINI_MODELS)

    for key_offset in range(total_keys):
        for model_offset in range(total_models):
            key_index = (current_key_index + key_offset) % total_keys
            model_index = (current_model_index + model_offset) % total_models

            api_key = GEMINI_API_KEYS[key_index]
            model = GEMINI_MODELS[model_index]

            print(f"🔁 Trying key {key_index + 1}/{total_keys} | Model: {model}")

            try:
                llm = get_llm(api_key, model)
                response = llm.invoke([HumanMessage(content=user_message)])
                
                # ✅ Update global state on success
                current_key_index = key_index
                current_model_index = model_index
                return response.content
            
            except OutputParserException:
                print(f"⚠️ Output parsing failed on {model}, trying next model.")
                continue

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "ResourceExhausted" in error_str:
                    print(f"❌ Rate limit hit for {model} using key {key_index}. Trying next key.")
                    continue
                else:
                    print(f"❌ Unexpected error: {e}")
                    continue  # Don't break; keep trying other key+model pairs

            time.sleep(1) # Optional: short delay to respect quotas

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
