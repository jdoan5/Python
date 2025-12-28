# Python Projects

This repository contains a collection of small-to-medium Python projects and practice examples.  
It started as a place to experiment with core Python concepts and has grown to include GUIs, data
pipelines, ML baselines, and FastAPI services.

## Repository layout (high level)

The projects are grouped roughly by theme under subfolders such as `Python Project 2`, `Python Project 3`, etc.  
Within each, you will find one or more self-contained projects with their own scripts and, in some
cases, a `requirements.txt`.

### 1. GUI utilities (Tkinter)

Small desktop apps built with Tkinter:

- **Simple Calculator**  
  `Python Project 2/Calculator/Simple Calculator.py`  
  Basic four-function calculator with clear / clear-entry and decimal support.

- **Scientific Calculator**  
  `Python Project 2/Calculator/Scientific Calculator.py`  
  Adds scientific operations (`x²`, `1/x`, `sin`, `cos`, `tan`, `log₁₀`, etc.), improved keyboard
  handling, and safer expression evaluation.

- **Unit Converter App**  
  `Python Project 2/Unit Converter App/`  
  Multiple small converters (e.g., BMI, temperature, and other unit conversions) in a single Tkinter UI.

### 2. Data & analytics projects

Projects that take CSV-style data, clean it, and prepare it for dashboards or analysis:

- **Popular Baby Names – Dashboard Prep**  
  `Python Project 2/Popular Baby Names with Looker Studio Dashboard/`  
  Cleans NYC baby name data, labels trends (New / Rising / Falling / Stable), and exports a tidy
  CSV for use in Looker Studio / Google Data Studio.

- **Customer Churn (Cars)**  
  `Python Project 2/Customer Churn (Cars)/`  
  Synthetic churn dataset for car owners. Uses TF-IDF features plus numeric/categorical variables and
  a Logistic Regression baseline. Includes CLI scripts `train.py` and `predict.py` for training and
  scoring CSV files.

### 3. Services & APIs (FastAPI)

Projects that expose Python logic via HTTP APIs:

- **Hello-Prod-FastAPI**  
  `Python Project 2/Hello-Prod-FastAPI/`  
  A small FastAPI template organized for “production style”: app-factory pattern, versioned routers,
  environment-based settings, and centralized error handling.

- **Customer Metrics Pipeline & API**  
  `Python Project 2/Customer Metrics Pipeline & API/`  
  Example churn pipeline that trains a simple model on customer metrics, saves artifacts, and exposes
  a `/score_customer` endpoint with interactive docs at `/docs`.

### 4. Foundations & practice

- **Misc practice scripts**  
  Small files exploring core syntax, control flow, functions/classes, and file I/O. Many of these are
  one-off utilities or experiments as new libraries and ideas are tried.

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/jdoan5/Python.git
cd Python