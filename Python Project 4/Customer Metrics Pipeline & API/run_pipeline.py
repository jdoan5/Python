# run_pipeline.py
"""
Convenience entry point so you can run:

    python run_pipeline.py

This generates synthetic customer data, trains the model,
and writes artifacts used by the FastAPI app.
"""
from customer_metrics.pipeline.train_model import run_pipeline


if __name__ == "__main__":
    run_pipeline()