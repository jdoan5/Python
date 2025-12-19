import joblib

MODEL_PATH = "models/fake_news_tfidf_logreg.joblib"  # adjust if in /models/

model = joblib.load(MODEL_PATH)

print("Loaded type:", type(model))
print("\nModel repr:\n", model)

# If it's a Pipeline, list steps
if hasattr(model, "named_steps"):
    print("\nPipeline steps:", list(model.named_steps.keys()))
    for name, step in model.named_steps.items():
        print(f"\n--- Step: {name} ---")
        print("Type:", type(step))
        # Show key params (can be long)
        try:
            print("Params:", step.get_params())
        except Exception as e:
            print("Could not get params:", e)

# If it ends with a classifier, show classes + coefficients shape
clf = None
if hasattr(model, "named_steps"):
    for step in model.named_steps.values():
        if hasattr(step, "predict") and hasattr(step, "coef_"):
            clf = step
            break

if clf is not None:
    print("\nClassifier classes_:", getattr(clf, "classes_", None))
    print("coef_ shape:", clf.coef_.shape)