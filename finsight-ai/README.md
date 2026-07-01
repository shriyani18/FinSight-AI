# FinSight AI — Smart Expense Analyzer & Forecaster

An AI-powered personal finance analyzer. Upload a bank statement CSV and get:
- Automatic transaction categorization
- Spend summary + category breakdown (pie chart)
- Anomaly detection (unusual transactions, z-score based)
- Next-month spend forecast (linear regression)
- AI-generated natural language insights (Groq/Llama 3.3, with rule-based fallback)
- Ask-a-question chat over your own finance data

## Tech Stack
- **Backend:** FastAPI (Python)
- **Data processing:** pandas, numpy
- **ML:** scikit-learn (LinearRegression for forecasting, z-score for anomalies)
- **GenAI:** Groq API (Llama 3.3) — optional, falls back gracefully without a key
- **Frontend:** Plain HTML/JS + Chart.js (no build step needed)

## Folder Structure
```
finsight-ai/
├── main.py              # FastAPI app: routes only, thin controller layer
├── analyzer.py          # All business logic: categorization, ML, AI insights
├── requirements.txt     # Python dependencies
├── sample_data.csv      # Sample bank statement to test with
├── render.yaml          # One-click Render deployment config
├── .env.example         # Environment variable reference
└── static/
    └── index.html       # Single-page frontend (upload, charts, chat)
```

## Run Locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
Visit `http://localhost:8000` and upload `sample_data.csv`.

(Optional) Add a free Groq API key to a `.env` file to enable real LLM-generated
insights instead of the rule-based fallback:
```
GROQ_API_KEY=your_key_here
```

## Deploy on Render (recommended — single service, no split needed)
1. Push this project to a GitHub repo.
2. Go to [render.com](https://render.com) → New → Web Service → connect your repo.
3. Render auto-detects `render.yaml`. Otherwise set manually:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. (Optional) Add `GROQ_API_KEY` under Environment tab.
5. Deploy — Render gives you a live URL (e.g. `finsight-ai.onrender.com`).

This works as a single service because FastAPI serves both the API (`/api/...`)
and the frontend (`/`, `/static/...`) itself — no CORS setup needed in production.

## Deploy on Vercel (alternative — frontend/backend split)
Vercel is built for serverless functions and static frontends, not long-running
Python apps with in-memory state — so for this project's architecture, **Render is
the better fit**. If you still want to use Vercel:
1. Host `static/index.html` on Vercel as a static site.
2. Host `main.py` + `analyzer.py` on Render (as above) as the API.
3. In `index.html`, change `const API_BASE = ""` to your Render URL,
   e.g. `const API_BASE = "https://finsight-ai.onrender.com"`.
4. CORS is already enabled in `main.py` (`allow_origins=["*"]`) so this split works.

## Interview Talking Points
- **Why FastAPI over Flask?** Async support, automatic OpenAPI docs, Pydantic
  validation out of the box — better for production APIs.
- **Why rule-based categorization instead of an ML classifier?** Explainable,
  zero training data needed, instant to ship — a real MVP pattern in fintech
  before collecting enough labeled data for a real classifier.
- **Why linear regression for forecasting?** Simple, explainable baseline.
  With more historical data, this could be swapped for ARIMA/Prophet/LSTM
  without changing the API contract (`forecast_next_month()` is isolated).
- **How is this "AI" and not just data analysis?** The `generate_insights()`
  and `answer_question()` functions call an LLM (Groq/Llama 3.3) to turn raw
  numbers into natural language explanations and answer free-form questions —
  and the app degrades gracefully to rule-based text if no API key is present,
  which is a good production-resilience pattern to mention.
- **Security consideration:** the "ask a question" feature computes numbers
  in pandas first, then only asks the LLM to *phrase* the answer — it never
  lets the LLM generate/execute arbitrary code, avoiding injection risks.
