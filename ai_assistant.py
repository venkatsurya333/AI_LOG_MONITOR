"""
ai_assistant.py — Intelligent AI Log Assistant (Gemini powered)

Features:
- Small talk handling (hi, hello, etc.)
- Intent classification (trend, security, errors)
- Context-aware log analysis
- Structured SRE-style responses
"""

import google.generativeai as genai
import pandas as pd
from config import GEMINI_API_KEY, ANOMALY_RESULTS_CSV
from logger import log_info, log_error


# ─────────────────────────────────────────────
# CONFIGURE GEMINI
# ─────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")


# ─────────────────────────────────────────────
# SMALL TALK DETECTION
# ─────────────────────────────────────────────
def is_small_talk(query):
    small_talk = [
        "hi", "hello", "hey", "how are you",
        "good morning", "good evening", "what's up"
    ]
    return query.lower().strip() in small_talk


# ─────────────────────────────────────────────
# INTENT CLASSIFICATION
# ─────────────────────────────────────────────
def classify_query(query):
    q = query.lower()

    if any(k in q for k in ["latest logs", "recent logs", "show logs", "last logs"]):
        return "logs"

    if any(k in q for k in ["trend", "increase", "pattern", "spike"]):
        return "trend"

    if any(k in q for k in ["error", "fail", "issue", "problem"]):
        return "error"

    if any(k in q for k in ["ssh", "login", "attack", "security"]):
        return "security"

    if any(k in q for k in ["summary", "overview", "all anomalies"]):
        return "summary"
    if "how many" in q:
        return "count"
    if "critical" in q:
        return "critical"

    return "general"
# ─────────────────────────────────────────────
# LOAD RECENT ANOMALIES
# ─────────────────────────────────────────────
def load_recent_logs(n=5):
    try:
        df = pd.read_csv(ANOMALY_RESULTS_CSV)
        df = df[df["anomaly"] == -1]

        if df.empty:
            return []

        logs = df.sort_values("anomaly_score_norm", ascending=False).head(n)
        return logs["message"].tolist()

    except Exception as e:
        log_error(f"Error loading logs: {e}")
        return []


# ─────────────────────────────────────────────
# BUILD PROMPT BASED ON INTENT
# ─────────────────────────────────────────────
def build_prompt(user_query, logs, intent):
    context = "\n".join(logs)

    base = f"""
You are an expert Site Reliability Engineer (SRE).

Analyze the logs and answer the user's question.

Logs:
{context}

User Question:
{user_query}

Give structured output:

Issue:
<main problem>

Cause:
<root cause>

Severity:
<CRITICAL/HIGH/MEDIUM/LOW with reason>

Trend:
<Increasing/Stable/Decreasing>

Recommended Action:
1. Immediate
2. Short-term
3. Long-term
"""

    # Intent-based tuning
    if intent == "trend":
        base += "\nFocus more on patterns, spikes, and time-based behavior."

    elif intent == "security":
        base += "\nFocus on security threats like brute force, unauthorized access."

    elif intent == "error":
        base += "\nFocus on system errors, failures, and root causes."

    elif intent == "summary":
        base += "\nProvide a concise summary of all anomalies."

    return base


# ─────────────────────────────────────────────
# ASK GEMINI
# ─────────────────────────────────────────────
def ask_gemini(user_query):
    try:
        logs = load_recent_logs()

        if not logs:
            return "No anomaly data available."

        intent = classify_query(user_query)

        # ✅ NEW: Direct log response
        if intent == "logs":
            response = "\n📄 Latest Anomalies:\n\n"
            for i, log in enumerate(logs, 1):
                response += f"{i}. {log[:150]}\n\n"
            return response

        prompt = build_prompt(user_query, logs, intent)
        response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        log_error(f"Gemini error: {e}")
        return "AI assistant failed to generate response."
# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def run_assistant():
    print("\n🤖  Gemini AI Log Assistant  (type 'exit' to quit)\n")
    log_info("AI assistant (Gemini) started")

    while True:
        user_query = input("Ask: ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting assistant...")
            break

        # Small talk handling
        if is_small_talk(user_query):
            print("\n🤖 Hello! Ask me about logs, anomalies, or system issues.\n")
            continue

        log_info(f"User query: {user_query}")

        response = ask_gemini(user_query)

        print("\n🧠 AI Analysis:\n")
        print(response)
        print("\n" + "═" * 70 + "\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_assistant()
