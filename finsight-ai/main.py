"""
main.py
-------
FastAPI application entry point for FinSight AI.

Design notes (good to mention in an interview):
- Stateless-ish demo: the uploaded dataframe is kept in a simple in-memory
  store keyed by a session id (cookie). For production you'd persist to a
  real DB (Postgres) or object storage (S3) per user.
- CORS enabled so a separately-hosted frontend (e.g. on Vercel) can call this API.
- Endpoints are thin — all real logic lives in analyzer.py (separation of concerns).
"""

import io
import uuid
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from analyzer import (
    load_and_prepare,
    compute_summary,
    detect_anomalies,
    forecast_next_month,
    generate_insights,
    answer_question,
)

app = FastAPI(title="FinSight AI", description="AI-powered personal finance analyzer")

# Allow requests from any frontend (tighten this to your actual domain in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory "database" -> {session_id: dataframe}. Simple & fine for a demo/interview project.
SESSION_STORE: dict[str, pd.DataFrame] = {}


class Question(BaseModel):
    question: str


def get_df(session_id: str | None) -> pd.DataFrame:
    """Fetch the uploaded dataframe for this session, or raise a clean error."""
    if not session_id or session_id not in SESSION_STORE:
        raise HTTPException(status_code=400, detail="No data uploaded yet. Please upload a CSV first.")
    return SESSION_STORE[session_id]


# --------------------------------------------------------------------------
# ROUTES
# --------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """Simple health check endpoint — useful for Render's uptime checks."""
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_csv(response: Response, file: UploadFile = File(...), session_id: str | None = Cookie(None)):
    """
    Accepts a CSV with columns: date, description, amount
    Stores the cleaned dataframe in memory under a session cookie.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    contents = await file.read()
    try:
        raw_df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        df = load_and_prepare(raw_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    # create a new session if one doesn't exist yet
    session_id = session_id or str(uuid.uuid4())
    SESSION_STORE[session_id] = df
    response.set_cookie(key="session_id", value=session_id, httponly=True)

    return {"message": "File processed successfully", "rows": len(df)}


@app.get("/api/summary")
def get_summary(session_id: str | None = Cookie(None)):
    df = get_df(session_id)
    summary = compute_summary(df)
    summary["ai_insights"] = generate_insights(summary)
    return summary


@app.get("/api/anomalies")
def get_anomalies(session_id: str | None = Cookie(None)):
    df = get_df(session_id)
    return {"anomalies": detect_anomalies(df)}


@app.get("/api/forecast")
def get_forecast(session_id: str | None = Cookie(None)):
    df = get_df(session_id)
    return forecast_next_month(df)


@app.post("/api/ask")
def ask_question(payload: Question, session_id: str | None = Cookie(None)):
    df = get_df(session_id)
    summary = compute_summary(df)
    answer = answer_question(df, summary, payload.question)
    return {"answer": answer}


# --------------------------------------------------------------------------
# Serve the frontend (single-service deployment: FastAPI serves its own UI)
# --------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")
