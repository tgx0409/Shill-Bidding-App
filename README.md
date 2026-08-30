# Shill Bidding Risk Dashboard (Streamlit)

A 4-page Streamlit app built from the assignment notebook: **Overview →
Explore the Data → Risk Predictor → Model Evaluation**. All 4 trained
models (Logistic Regression, SVM, Random Forest, Gradient Boosting) are
pre-trained and shipped as artifacts — the app does not retrain anything,
so it starts instantly.

## Folder contents

```
shill_bidding_app/
├── app.py                  # the Streamlit app (single entry point)
├── train_export.py         # run once to (re)generate the artifacts below
├── requirements.txt
├── .streamlit/config.toml  # theme
└── artifacts/               # pre-trained models + data, generated from train_export.py
    ├── model_logreg.joblib
    ├── model_svm.joblib
    ├── model_rf.joblib
    ├── model_gb.joblib
    ├── scaler.joblib
    ├── feature_ranges.json
    ├── clean_data.csv
    ├── X_test.csv
    ├── X_test_scaled.csv
    ├── y_test.csv
    └── results_table.csv
```

## 1. Run locally first (recommended before deploying)

```
cd shill_bidding_app
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501`. Click through all four pages once to
confirm everything loads before you deploy.

## 2. Deploy to Streamlit Community Cloud (free)

1. **Push this folder to a public (or private) GitHub repo.** The whole
   `shill_bidding_app/` folder — including `artifacts/` — must be committed.
   The `.csv`/`.joblib` files together are a few MB, well within GitHub's
   normal file-size limits.

   ```
   cd shill_bidding_app
   git init
   git add .
   git commit -m "Shill bidding risk dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, pick your repo/branch, and set **Main file path** to
   `app.py` (or `shill_bidding_app/app.py` if you pushed it inside a larger
   repo — adjust the path accordingly).
4. Click **Deploy**. First build takes a couple of minutes while it installs
   `requirements.txt`; after that the app is live at a public
   `https://<your-app-name>.streamlit.app` URL you can put in your
   assignment submission.

## 3. Common gotchas

- **"File not found: artifacts/..."** — the app expects to be run with
  `artifacts/` as a sibling folder of `app.py`. Don't rename or move it
  without updating the `ART = "artifacts"` path at the top of `app.py`.
- **Model version warnings on load** — `requirements.txt` pins
  `scikit-learn==1.8.0` to match the version the models were trained with.
  If you change this pin, re-run `train_export.py` first to re-export
  matching model files.
- **Slow first load** — the first page view re-trains nothing, but Streamlit
  Cloud's free tier can take a few seconds to "wake up" a sleeping app after
  inactivity. This is normal.
- **Want to retrain instead of using the shipped artifacts?** Put
  `Shill_Bidding_Dataset.csv` next to `train_export.py` and re-run it, then
  the `artifacts/` folder will be regenerated with fresh model files.

## Which model to feature

`Random Forest` and `Gradient Boosting` post the strongest PR-AUC (0.9988 /
0.9992) on the held-out test set, but `Successive_Outbidding` alone drives
most of that separation. see the "Limitations" section on the Model
Evaluation page for the full discussion (this is intentionally kept in the
app rather than duplicated in the written report).
