"""
analyzer.py
-----------
All the "brains" of FinSight AI lives here:
1. Rule-based transaction categorization (keyword matching)
2. Summary statistics (spend by category, by month)
3. Anomaly detection (z-score based outlier spends)
4. Forecasting next month's spend (simple linear regression)
5. AI-generated insights + natural language Q&A (Groq LLM, with safe fallback)

Kept intentionally simple (no heavy NLP libs) so it's easy to explain in an interview.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import requests

# --------------------------------------------------------------------------
# 1. CATEGORIZATION
# --------------------------------------------------------------------------
# Real-world equivalent: banks/fintech apps (e.g. Mint, Cred) use ML classifiers
# trained on millions of transactions. Here we use keyword rules — fast,
# explainable, zero training cost — a very common MVP approach in real fintech.
CATEGORY_KEYWORDS = {
    "Food": ["swiggy", "zomato", "restaurant", "cafe", "food", "dominos"],
    "Transport": ["uber", "ola", "fuel", "petrol", "metro", "irctc"],
    "Shopping": ["amazon", "flipkart", "myntra", "shopping", "mall"],
    "Bills & Utilities": ["electricity", "recharge", "bill", "broadband", "gas"],
    "Entertainment": ["netflix", "spotify", "movie", "bookmyshow", "prime"],
    "Income": ["salary", "credit", "refund", "interest"],
}


def categorize_transaction(description: str) -> str:
    """Match a transaction description against keyword dictionary."""
    desc = str(description).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in desc for keyword in keywords):
            return category
    return "Other"


def load_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the uploaded CSV and add derived columns.
    Expected columns: date, description, amount (negative = expense, positive = income)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])
    df["category"] = df["description"].apply(categorize_transaction)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


# --------------------------------------------------------------------------
# 2. SUMMARY
# --------------------------------------------------------------------------
def compute_summary(df: pd.DataFrame) -> dict:
    expenses = df[df["amount"] < 0].copy()
    income = df[df["amount"] > 0].copy()
    expenses["amount"] = expenses["amount"].abs()

    category_totals = (
        expenses.groupby("category")["amount"].sum().sort_values(ascending=False).round(2)
    )
    monthly_totals = expenses.groupby("month")["amount"].sum().round(2)

    return {
        # Cast numpy types -> native Python types, otherwise FastAPI's JSON
        # encoder will throw errors on np.int64 / np.float64.
        "total_spend": float(round(expenses["amount"].sum(), 2)),
        "total_income": float(round(income["amount"].sum(), 2)),
        "net_savings": float(round(income["amount"].sum() - expenses["amount"].sum(), 2)),
        "category_breakdown": {k: float(v) for k, v in category_totals.to_dict().items()},
        "monthly_spend": {k: float(v) for k, v in monthly_totals.to_dict().items()},
        "top_category": category_totals.idxmax() if not category_totals.empty else None,
    }


# --------------------------------------------------------------------------
# 3. ANOMALY DETECTION
# --------------------------------------------------------------------------
def detect_anomalies(df: pd.DataFrame, z_threshold: float = 2.0) -> list:
    """
    Flags transactions whose amount deviates far from the mean spend
    *within their own category* (z-score method).
    This is a classic, interview-friendly explainable anomaly detection technique.
    """
    expenses = df[df["amount"] < 0].copy()
    expenses["amount"] = expenses["amount"].abs()
    flagged = []

    for category, group in expenses.groupby("category"):
        if len(group) < 3:
            continue  # not enough data to judge what's "normal"
        mean, std = group["amount"].mean(), group["amount"].std()
        if std == 0:
            continue
        group = group.copy()
        group["zscore"] = (group["amount"] - mean) / std
        outliers = group[group["zscore"].abs() > z_threshold]
        for _, row in outliers.iterrows():
            flagged.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "description": row["description"],
                "amount": float(round(row["amount"], 2)),
                "category": category,
                "zscore": float(round(row["zscore"], 2)),
            })
    return flagged


# --------------------------------------------------------------------------
# 4. FORECASTING
# --------------------------------------------------------------------------
def forecast_next_month(df: pd.DataFrame) -> dict:
    """
    Simple linear regression on monthly totals: month_index -> total_spend.
    Not meant to be state-of-the-art — meant to be EXPLAINABLE in an interview:
    "I used linear regression treating each month as a time step, and predicted
    the next step. With more data I'd use ARIMA/Prophet."
    """
    expenses = df[df["amount"] < 0].copy()
    expenses["amount"] = expenses["amount"].abs()
    monthly = expenses.groupby("month")["amount"].sum().sort_index()

    if len(monthly) < 2:
        return {"forecast": None, "note": "Need at least 2 months of data to forecast."}

    X = np.arange(len(monthly)).reshape(-1, 1)  # 0, 1, 2, ... for each month
    y = monthly.values

    model = LinearRegression()
    model.fit(X, y)
    next_index = np.array([[len(monthly)]])
    prediction = model.predict(next_index)[0]

    return {
        "forecast": float(round(max(prediction, 0), 2)),  # spend can't be negative
        "based_on_months": list(monthly.index),
        "trend": "increasing" if model.coef_[0] > 0 else "decreasing",
    }


# --------------------------------------------------------------------------
# 5. AI INSIGHTS + Q&A (Groq LLM with graceful fallback)
# --------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _call_llm(prompt: str) -> str | None:
    """Calls Groq's Llama model. Returns None if no key set or call fails."""
    if not GROQ_API_KEY:
        return None
    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 300,
            },
            timeout=15,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        return None  # fail gracefully -> caller falls back to rule-based text


def generate_insights(summary: dict) -> str:
    """
    Produces a natural-language summary of spending.
    Uses LLM if GROQ_API_KEY is set, else builds a rule-based sentence.
    This mirrors how real fintech apps (e.g. Cleo, Wealthfront) generate
    'AI insights' cards for users.
    """
    prompt = (
        f"You are a financial assistant. Given this spending summary: {summary}, "
        "write 3 short, friendly, actionable insights (bullet points) for the user."
    )
    llm_response = _call_llm(prompt)
    if llm_response:
        return llm_response

    # Fallback: rule-based insight generation (no API key needed to demo)
    top_cat = summary.get("top_category")
    total = summary.get("total_spend", 0)
    savings = summary.get("net_savings", 0)
    lines = [f"- Your highest spending category is **{top_cat}**."]
    lines.append(f"- Total tracked spend: ₹{total}.")
    if savings < 0:
        lines.append("- You spent more than you earned this period — consider reviewing discretionary categories.")
    else:
        lines.append(f"- You saved ₹{round(savings, 2)} this period. Nice work!")
    return "\n".join(lines)


def answer_question(df: pd.DataFrame, summary: dict, question: str) -> str:
    """
    Natural language Q&A over the data.
    Strategy: compute the real numbers safely with pandas first (no arbitrary
    code execution -> no security risk), then let the LLM phrase the answer
    nicely. If no LLM key, return the raw computed answer directly.
    """
    q = question.lower()

    # crude category detection from the question
    matched_category = None
    for category in CATEGORY_KEYWORDS:
        if category.lower() in q:
            matched_category = category
            break

    if matched_category:
        expenses = df[(df["amount"] < 0) & (df["category"] == matched_category)]
        total = round(expenses["amount"].abs().sum(), 2)
        raw_answer = f"You spent ₹{total} on {matched_category}."
    else:
        raw_answer = f"Total spend: ₹{summary['total_spend']}. Total income: ₹{summary['total_income']}."

    prompt = f"User asked: '{question}'. The factual answer is: {raw_answer}. Rephrase this in one friendly sentence."
    llm_response = _call_llm(prompt)
    return llm_response or raw_answer
