import os
import pathlib
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# --- PATH ---
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SOURCE_DIR = DATA_DIR / "source"
INDEX_DIR = BASE_DIR / "index"

INDEX_DIR.mkdir(exist_ok=True)

def load_documents() -> List[Dict]:
    """
    Walk through data/source and load all .txt files.
    Each file can be split into chunks by blank lines.
    """

