import warnings
warnings.filterwarnings("ignore")

from elasticsearch import Elasticsearch
import google.generativeai as genai
from logger import log_info, log_error
from dotenv import load_dotenv
import os

# -------------------------
# LOAD ENV VARIABLES
# -------------------------
load_dotenv()

ES_HOST = os.getenv("ES_HOST")
ES_USER = os.getenv("ES_USER")
ES_PASS = os.getenv("ES_PASS")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -------------------------
# INITIALIZE GEMINI
# -------------------------
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    log_info("Gemini model initialized")
except Exception as e:
    log_error(f"Gemini initialization failed: {e}")
    exit()

# -------------------------
# CONNECT TO ELASTICSEARCH
# -------------------------
try:
    es = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASS),
        verify_certs=False
    )

    if es.ping():
        log_info("Connected to Elasticsearch (LLM assistant)")
    else:
        log_error("Elasticsearch ping failed")
        exit()

except Exception as e:
    log_error(f"Elasticsearch connection failed: {e}")
    exit()

# -------------------------
# FETCH LOGS FUNCTION
# -------------------------
def get_logs(query_text):
    try:
        log_info(f"Fetching logs for query: {query_text}")

        res = es.search(
            index="ai-log-anomalies",
            size=10,
            query={
                "multi_match": {
                    "query": query_text,
                    "fields": ["message", "severity"]
                }
            }
        )

        logs = [
            f"{hit['_source'].get('timestamp')} | {hit['_source'].get('severity')} | {hit['_source'].get('message')}"
            for hit in res["hits"]["hits"]
        ]

        # Fallback
        if not logs:
            log_info("No direct match found, fetching recent logs")

            res = es.search(
                index="ai-log-anomalies",
                size=5,
                query={"match_all": {}}
            )

            logs = [
                f"{hit['_source'].get('timestamp')} | {hit['_source'].get('severity')} | {hit['_source'].get('message')}"
                for hit in res["hits"]["hits"]
            ]

        log_info(f"Fetched {len(logs)} logs")
        return logs

    except Exception as e:
        log_error(f"Error fetching logs: {e}")
        return []

# -------------------------
# GEMINI RESPONSE FUNCTION
# -------------------------
def ask_gemini(user_query, logs, history):

    try:
        context = "\n".join(logs)

        # 🔥 Add last 3 queries as memory
        history_context = "\n".join(history[-3:]) if history else "None"

        prompt = f"""
You are an expert Site Reliability Engineer (SRE).

Your job is to analyze system logs and provide clear, structured insights.

Previous Questions:
{history_context}

Analyze the logs and respond STRICTLY in this format:

Issue:
Cause:
Severity (CRITICAL/HIGH/MEDIUM/LOW):
Trend (Increasing/Decreasing/Stable):
Recommended Action:

Logs:
{context}

User Question:
{user_query}
"""

        response = model.generate_content(prompt)

        log_info("Gemini response generated")

        return response.text.strip()

    except Exception as e:
        log_error(f"Gemini error: {e}")
        print("DEBUG ERROR:", e)
        return "Error generating AI response."

# -------------------------
# MAIN CHAT LOOP
# -------------------------
print("\n🤖 Gemini AI Log Assistant Started (type 'exit' to quit)\n")
log_info("AI assistant (Gemini) started")

chat_history = []

while True:
    try:
        user_query = input("Ask: ").strip()

        if not user_query:
            continue

        log_info(f"User query: {user_query}")

        if user_query.lower() in ["exit", "quit"]:
            log_info("Gemini assistant exited by user")
            print("Exiting AI assistant...")
            break

        logs = get_logs(user_query)

        if not logs:
            print("❌ No logs available.\n")
            log_info("No logs returned")
            continue

        # 🔥 Pass history
        answer = ask_gemini(user_query, logs, chat_history)

        print("\n🧠 AI Answer:\n")
        print(answer)
        print("\n" + "=" * 60 + "\n")

        # 🔥 Store history AFTER response
        chat_history.append(user_query)

    except Exception as e:
        log_error(f"Assistant runtime error: {e}")
        print("Unexpected error occurred.")
