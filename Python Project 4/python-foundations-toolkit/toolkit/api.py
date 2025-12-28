# toolkit/api.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .core.csv_tools import summarize_csv
from .core.text_tools import summarize_text

app = FastAPI(
    title="Python Foundations Toolkit API",
    version="0.1.0",
    description=(
        "Tiny FastAPI service that reuses the toolkit's core utilities for "
        "CSV and text summaries."
    ),
)


class CSVSummaryRequest(BaseModel):
    path: str = Field(
        ...,
        description="Path to the CSV file (relative to server working directory).",
        examples=["data/sample.csv"],
    )
    delimiter: str = Field(
        ",",
        description="CSV delimiter (default: ',').",
    )
    max_preview_rows: int = Field(
        5,
        ge=0,
        le=100,
        description="Maximum number of preview rows to return (0–100).",
    )


class TextSummaryRequest(BaseModel):
    path: str = Field(
        ...,
        description="Path to the text file (relative to server working directory).",
        examples=["data/sample.txt"],
    )
    top_n: int = Field(
        5,
        ge=1,
        le=50,
        description="How many top words to include (1–50).",
    )


@app.get("/health", summary="Health check")
def health() -> Dict[str, Any]:
    """
    Simple health-check endpoint.
    """
    return {"status": "ok"}


@app.post("/csv-summary", summary="Summarize a CSV file")
def csv_summary(payload: CSVSummaryRequest) -> Dict[str, Any]:
    """
    Return column names, row count, and a small preview of a CSV file.

    Uses the same summarize_csv() function as the CLI.
    """
    path = Path(payload.path)

    try:
        summary = summarize_csv(
            path=path,
            delimiter=payload.delimiter,
            max_preview_rows=payload.max_preview_rows,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # pragma: no cover - generic fallback
        raise HTTPException(status_code=500, detail=f"Error summarizing CSV: {e}")

    return summary


@app.post("/text-summary", summary="Summarize a text file")
def text_summary(payload: TextSummaryRequest) -> Dict[str, Any]:
    """
    Return basic stats and top words for a text file.

    Uses the same summarize_text() function as the CLI.
    """
    path = Path(payload.path)

    try:
        summary = summarize_text(
            path=path,
            top_n=payload.top_n,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {e}")

    return summary