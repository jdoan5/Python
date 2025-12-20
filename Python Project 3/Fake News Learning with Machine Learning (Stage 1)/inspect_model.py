import argparse
import joblib
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Inspect model top TF-IDF features.")
    p.add_argument("--model", default="artifacts/fake_news_tfidf_logreg.joblib")
    p.add_argument("--topk", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    pipe = joblib.load(args.model)

    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]

    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]  # binary classifier

    # LogisticRegression: positive weights push toward class 1 (Fake)
    top_fake_idx = coefs.argsort()[-args.topk:][::-1]
    top_real_idx = coefs.argsort()[:args.topk]

    fake_df = pd.DataFrame({
        "feature": feature_names[top_fake_idx],
        "weight": coefs[top_fake_idx]
    })
    real_df = pd.DataFrame({
        "feature": feature_names[top_real_idx],
        "weight": coefs[top_real_idx]
    })

    print("\nMost indicative of FAKE (label=1):")
    print(fake_df.to_string(index=False))

    print("\nMost indicative of REAL (label=0):")
    print(real_df.to_string(index=False))


if __name__ == "__main__":
    main()