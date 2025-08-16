
# Graph Experiment (Streamlit) — Ready for GitHub

Includes:
- `app.py` — Streamlit app (simple & precise) with Preflight + Debug
- `Colors in charts.csv` — your input file
- `images/` — all referenced images from the CSV (generated placeholders if missing)
- `requirements.txt`
- `README.md`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push the folder to a public GitHub repo.
2. On https://streamlit.io/cloud → New app → choose your repo → file path = `app.py` → Deploy.
3. The app reads the bundled `Colors in charts.csv` and image files.
