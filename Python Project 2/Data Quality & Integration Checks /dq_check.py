import json, re, sqlite3, datetime as dt
from pathlib import Path
import pandas as pd, yaml

def coerce(s: pd.Series, t: str) -> pd.Series:
    t = t.lower()
    if t == "int":      return pd.to_numeric(s, errors="coerce").astype("Int64")
    if t == "float":    return pd.to_numeric(s, errors="coerce")
    if t == "bool":     return s.astype(str).str.lower().map({"true":True,"false":False})
    if t == "date":     return pd.to_datetime(s, errors="coerce").dt.date
    if t == "datetime": return pd.to_datetime(s, errors="coerce")
    return s.astype(str)

def load(cfg):
    src = cfg["source"]
    if src["type"] == "csv":
        return pd.read_csv(src["path"])
    if src["type"] == "sqlite":
        con = sqlite3.connect(src["db_path"])
        q = src.get("query") or f"select * from {src['table']}"
        df = pd.read_sql_query(q, con); con.close(); return df
    raise SystemExit("Unsupported source")

def fk_parent_series(p):
    if p["type"] == "csv":
        return pd.read_csv(p["path"])[p["column"]]
    if p["type"] == "sqlite":
        con = sqlite3.connect(p["db_path"])
        q = p.get("query") or f"select {p['column']} from {p['table']}"
        s = pd.read_sql_query(q, con)[p["column"]]; con.close(); return s
    raise SystemExit("Unsupported FK parent source")

def main(cfg_path="dq_config.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    df = load(cfg)
    out = []

    # table checks
    t = cfg.get("checks", {}).get("table", {})
    if "min_rows" in t:
        out.append({"check":"min_rows","pass":len(df)>=t["min_rows"],"rows":len(df)})
    if t.get("no_duplicate_rows"):
        dups = df.duplicated().sum()
        out.append({"check":"no_duplicate_rows","pass":dups==0,"duplicate_rows":int(dups)})
    if t.get("unique_keys"):
        dups = df.duplicated(subset=t["unique_keys"]).sum()
        out.append({"check":"unique_keys","pass":dups==0,"duplicate_key_rows":int(dups),"keys":t["unique_keys"]})

    # column checks
    for col, rules in cfg.get("checks",{}).get("columns",{}).items():
        if col not in df.columns:
            out.append({"check":"column_exists","column":col,"pass":False}); continue
        s = df[col]
        if "dtype" in rules:
            coerced = coerce(s, rules["dtype"])
            bad = (coerced.isna() & s.notna()).sum() if rules["dtype"] in ("int","float","date","datetime","bool") else 0
            out.append({"check":"dtype","column":col,"target":rules["dtype"],"pass":bad==0,"invalid_dtype":int(bad)})
            s = coerced
        if rules.get("nullable") is False:
            nulls = s.isna().sum()
            out.append({"check":"not_null","column":col,"pass":nulls==0,"nulls":int(nulls)})
        if rules.get("unique"):
            dups = s.duplicated(keep=False).sum()
            out.append({"check":"unique","column":col,"pass":dups==0,"non_unique":int(dups)})
        if "allowed_values" in rules:
            bad = (~s.isin(rules["allowed_values"])).sum()
            out.append({"check":"allowed_values","column":col,"pass":bad==0,"invalid":int(bad)})
        if "regex" in rules and rules["regex"]:
            pat = re.compile(rules["regex"])
            invalid = (~s.fillna("").astype(str).map(lambda x: bool(pat.match(x)))).sum()
            out.append({"check":"regex","column":col,"pass":invalid==0,"invalid":int(invalid)})
        if "min" in rules:
            v = rules["min"];
            if isinstance(s.iloc[0], dt.date) and isinstance(v, str): v = dt.date.fromisoformat(v)
            bad = (s < v).sum()
            out.append({"check":"min","column":col,"pass":bad==0,"violations":int(bad),"min":str(v)})
        if "max" in rules:
            v = rules["max"];
            if v is not None:
                if isinstance(s.iloc[0], dt.date) and isinstance(v, str): v = dt.date.fromisoformat(v)
                bad = (s > v).sum()
                out.append({"check":"max","column":col,"pass":bad==0,"violations":int(bad),"max":str(v)})

    # integration checks (FK)
    for fk in cfg.get("integrations",{}).get("foreign_keys", []):
        child = fk["child"]["column"]
        parent_vals = set(fk_parent_series(fk["parent"]).dropna().astype(str).unique())
        missing = df[~df[child].astype(str).isin(parent_vals)]
        out.append({"check":"foreign_key","child":child,"pass":missing.empty,"missing_count":int(len(missing))})

    # summarize + write reports
    passed = sum(1 for r in out if r["pass"])
    report = {"summary":{"passed":passed,"failed":len(out)-passed,"total":len(out)},
              "results":out}
    Path("dq_report.json").write_text(json.dumps(report, indent=2, default=str))
    # quick markdown
    lines = [f"## Data Quality Summary",
             f"- Passed: **{passed}**  Failed: **{len(out)-passed}**  Total: **{len(out)}**",
             "", "### Checks"]
    for r in out:
        status = "✅" if r["pass"] else "❌"
        lines.append(f"- {status} `{r.get('check')}` {r}")
    Path("dq_report.md").write_text("\n".join(lines))
    print(json.dumps(report["summary"], indent=2))
    print("Wrote dq_report.json and dq_report.md")

if __name__ == "__main__":
    main()