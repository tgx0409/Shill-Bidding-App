"""
Shill Bidding Risk Dashboard
Streamlit app: Overview, Explore the Data, Risk Predictor, Model Evaluation.
Built on the assignment notebook's pipeline; models are pre-trained
artifacts, so the app starts instantly and never retrains anything.
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

sns.set_style("whitegrid")

st.set_page_config(
    page_title="Shill Bidding Risk Dashboard",
    page_icon="🪙",
    layout="wide",
)

ART = "artifacts"

# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    return {
        "Logistic Regression (baseline)": joblib.load(f"{ART}/model_logreg.joblib"),
        "SVM (Poly kernel)": joblib.load(f"{ART}/model_svm.joblib"),
        "Random Forest": joblib.load(f"{ART}/model_rf.joblib"),
        "Gradient Boosting": joblib.load(f"{ART}/model_gb.joblib"),
    }

@st.cache_resource
def load_scaler():
    return joblib.load(f"{ART}/scaler.joblib")

@st.cache_data
def load_feature_ranges():
    with open(f"{ART}/feature_ranges.json") as f:
        return json.load(f)

@st.cache_data
def load_clean_data():
    return pd.read_csv(f"{ART}/clean_data.csv")

@st.cache_data
def load_test_set():
    X_test = pd.read_csv(f"{ART}/X_test.csv")
    X_test_scaled = pd.read_csv(f"{ART}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{ART}/y_test.csv").iloc[:, 0]
    return X_test, X_test_scaled, y_test

@st.cache_data
def load_results_table():
    return pd.read_csv(f"{ART}/results_table.csv", index_col=0)

MODELS = load_models()
SCALER = load_scaler()
RANGE_DATA = load_feature_ranges()
FEATURES = RANGE_DATA["features"]
RANGES = RANGE_DATA["ranges"]

# LR and SVM were trained on scaled features; RF and GB were trained on raw features
SCALED_MODELS = {"Logistic Regression (baseline)", "SVM (Poly kernel)"}

FEATURE_HELP = {
    "Bidder_Tendency": "Normalised tendency of a bidder to participate in many auctions run by the same seller.",
    "Bidding_Ratio": "Ratio of a bidder's bids to the total bids placed in the auction.",
    "Successive_Outbidding": "Whether the bidder successively outbid themself / a partner (0, 0.5, or 1).",
    "Last_Bidding": "Normalised time of the bidder's last bid before auction close.",
    "Auction_Bids": "Normalised number of bids placed in the auction.",
    "Starting_Price_Average": "Normalised ratio of the starting price to the average starting price.",
    "Early_Bidding": "Normalised time of the bidder's first bid.",
    "Winning_Ratio": "Ratio of auctions the bidder has won historically.",
    "Auction_Duration": "Duration of the auction in days.",
}

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
with st.sidebar:
    page = option_menu(
        menu_title=None,
        options=["Overview", "Explore the Data", "Risk Predictor", "Model Evaluation"],
        menu_icon=None,
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"display": "none"},
            "nav-link": {
                "font-size": "16px",
                "font-weight": "600",
                "text-align": "left",
                "margin": "2px 0px",
                "padding": "12px 16px",
                "border-radius": "4px",
                "color": "#1f1f1f",
                "transition": "background-color 0.25s ease, color 0.25s ease",
            },
            "nav-link-selected": {
                "background-color": "#FFE066",
                "color": "#1f1f1f",
                "font-weight": "700",
            },
        },
    )
st.write("")
st.write("")
st.write("")
st.sidebar.divider()
st.sidebar.markdown(
    "**Source:** UCI Shill Bidding Dataset with 6,321 bids across 807 online "
    "auctions, 9 behavioural features per bid."
)

# ----------------------------------------------------------------------------
# PAGE: Overview
# ----------------------------------------------------------------------------
if page == "Overview":
    st.title("Shill Bidding Risk: Project Overview")

    df = load_clean_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total bids", f"{len(df):,}")
    c2.metric("Normal bids", f"{(df.Class == 0).sum():,}")
    c3.metric("Shill bids", f"{(df.Class == 1).sum():,}")
    c4.metric("Shill bidding rate", f"{df.Class.mean()*100:.1f}%")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Business problem")
        st.markdown(
            """
            A shill bidder is someone artificially inflating the price on the seller's behalf. 
            Shill bidding erodes trust in online auction platforms by artificially
            raising prices for genuine bidders. Flagging suspicious bids lets a
            platform investigate or intervene before an auction closes.

            The dataset's 9 features capture bidding **behaviour** such as
            timing, repetition, ratios of bids to auction activity, and
            historical win rate, for a shill bidder's *pattern* is more
            detectable than any single bid.
            """
        )
    with col2:
        st.subheader("Models compared")
        st.markdown(
            """
            | Model | Family | Role |
            |---|---|---|
            | Logistic Regression | Linear | **Baseline** |
            | SVM (poly kernel) | Max-margin | Comparison |
            | Random Forest | Tree ensemble (bagging) | Comparison |
            | Gradient Boosting | Tree ensemble (boosting) | Comparison |

            Random Forest and Gradient Boosting were hyperparameter-tuned with
            `RandomizedSearchCV` (25 candidates, 5-fold stratified CV, scored on
            PR-AUC because the target is imbalanced with only 10.7% shill bids).
            """
        )

# ----------------------------------------------------------------------------
# PAGE: Explore the Data
# ----------------------------------------------------------------------------
elif page == "Explore the Data":
    st.title("Explore the Data")
    df = load_clean_data()

    tab1, tab2, tab3 = st.tabs(["Target Balance", "Feature Distributions", "Correlations"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Class distribution**")
            fig, ax = plt.subplots(figsize=(4, 4))
            df["Class"].value_counts().sort_index().plot(
                kind="bar", ax=ax, color=["#FBDA0C", "#0057AD"]
            )
            ax.set_xticklabels(["Normal (0)", "Shill (1)"], rotation=0)
            ax.set_ylabel("Count")
            st.pyplot(fig)
        with col2:
            st.markdown("**Why this matters**")
            st.markdown(
                """
                Only 10.7% of bids are shill bids, which means that accuracy alone can be misleading,
                for a model that rarely catches the minority class can still score high.
                That's why every model below was trained with `class_weight="balanced"`
                (or equivalent sample weighting for Gradient Boosting), and why
                **PR-AUC**, not plain accuracy, was used to select hyperparameters.
                """
            )

    with tab2:
        feature = st.selectbox(
            "Choose a feature", FEATURES,
            format_func=lambda x: x.replace("_", " ")
        )
        st.caption(FEATURE_HELP.get(feature, ""))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.histplot(data=df, x=feature, hue="Class", kde=True, ax=axes[0],
             palette=["#FBDA0C", "#0057AD"], multiple="layer", alpha=0.5)
        axes[0].set_title(f"{feature.replace('_', ' ')} distribution by class")
        axes[0].set_xlabel(feature.replace("_", " "))
        sns.boxplot(data=df, x="Class", y=feature, hue="Class", ax=axes[1],
             palette=["#FBDA0C", "#0057AD"], legend=False)
        axes[1].set_xticklabels(["Normal", "Shill"])
        axes[1].set_title(f"{feature.replace('_', ' ')} by class")
        axes[1].set_ylabel(feature.replace("_", " "))
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.markdown("**Correlation matrix (all features + target)**")
        corr = df.corr()
        corr_display = corr.rename(
            index=lambda x: x.replace("_", " "),
            columns=lambda x: x.replace("_", " "),
        )
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr_display, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    annot_kws={"size": 8}, ax=ax)
        st.pyplot(fig)
        st.caption(
            "`Successive_Outbidding` shows the strongest correlation with `Class` "
            "by a wide margin. This is confirmed by every model's feature "
            "importance ranking on the Model Evaluation page."
        )

# ----------------------------------------------------------------------------
# PAGE: Risk Predictor
# ----------------------------------------------------------------------------
elif page == "Risk Predictor":
    st.title("Live Shill Bidding Risk Predictor")

    model_name = st.selectbox("Model to use", list(MODELS.keys()))

    st.subheader("Bid features")
    c1, c2, c3 = st.columns(3)
    cols_per_col = [FEATURES[0:3], FEATURES[3:6], FEATURES[6:9]]
    values = {}

    for col_container, feats in zip([c1, c2, c3], cols_per_col):
        with col_container:
            for feat in feats:
                r = RANGES[feat]
                if feat == "Auction_Duration":
                    values[feat] = st.slider(
                        feat.replace("_", " "), int(r["min"]), int(r["max"]), int(round(r["median"])),
                        help=FEATURE_HELP.get(feat),
                    )
                elif feat == "Successive_Outbidding":
                    values[feat] = st.select_slider(
                        feat.replace("_", " "), options=[0.0, 0.5, 1.0], value=0.0,
                        help=FEATURE_HELP.get(feat),
                    )
                else:
                    values[feat] = st.slider(
                        feat.replace("_", " "), float(r["min"]), float(r["max"]), float(r["median"]),
                        help=FEATURE_HELP.get(feat),
                    )

    if st.button("Predict risk", type="primary"):
        X_input = pd.DataFrame([values])[FEATURES]

        model = MODELS[model_name]
        if model_name in SCALED_MODELS:
            X_input = pd.DataFrame(SCALER.transform(X_input), columns=FEATURES)

        prob = model.predict_proba(X_input)[0, 1]

        st.divider()
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Predicted shill-bidding probability", f"{prob*100:.1f}%")
            label = "Likely shill bidder" if prob >= 0.5 else "Likely normal bidder"
            st.markdown(f"**Model prediction:** {label}")
        with r2:
            fig, ax = plt.subplots(figsize=(6, 1.2))
            ax.barh([0], [1], color="#EDF3F2")
            ax.barh([0], [prob], color="#0057AD" if prob >= 0.5 else "#FBDA0C")
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Predicted probability of shill bidding")
            st.pyplot(fig)

        st.info(
            "This estimate reflects patterns learned from historical auction data "
            "and the chosen model's assumptions. It is a screening signal, not "
            "proof of wrongdoing. Flagged bids should be reviewed by a human "
            "before any action is taken."
        )

# ----------------------------------------------------------------------------
# PAGE: Model Evaluation
# ----------------------------------------------------------------------------
elif page == "Model Evaluation":
    st.title("Model Evaluation")

    X_test, X_test_scaled, y_test = load_test_set()
    results_precomputed = load_results_table()

    st.subheader("Metrics on the held-out test set (1,265 bids)")
    st.dataframe(results_precomputed.style.format("{:.4f}").highlight_max(axis=0, color="#FBDA0C"))

    st.divider()
    st.subheader("Metric comparison & confusion matrix")
    col1, col2 = st.columns(2)

    with col1:
        metric_choice = st.selectbox(
            "Compare models on a metric",
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"],
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        results_precomputed[metric_choice].sort_values().plot(kind="barh", ax=ax, color="#FBDA0C")
        ax.set_xlabel(metric_choice)
        st.pyplot(fig)

    with col2:
        cm_model_name = st.selectbox("Confusion matrix", list(MODELS.keys()), key="cm_model")
        model = MODELS[cm_model_name]
        X_te = X_test_scaled if cm_model_name in SCALED_MODELS else X_test
        y_pred = model.predict(X_te)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Shill"], yticklabels=["Normal", "Shill"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.divider()
    st.subheader("Feature importance (tree-based models)")
    fi_col1, fi_col2 = st.columns(2)
    with fi_col1:
        rf_imp = pd.Series(
            MODELS["Random Forest"].feature_importances_,
            index=[f.replace("_", " ") for f in FEATURES]
        ).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        rf_imp.plot(kind="barh", ax=ax, color="#55A868")
        ax.set_title("Random Forest")
        st.pyplot(fig)
    with fi_col2:
        gb_imp = pd.Series(
            MODELS["Gradient Boosting"].feature_importances_,
            index=[f.replace("_", " ") for f in FEATURES]
        ).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        gb_imp.plot(kind="barh", ax=ax, color="#0057AD")
        ax.set_title("Gradient Boosting")
        st.pyplot(fig)

    st.divider()
    st.subheader("Limitations")
    st.markdown(
        """
        - **`Successive_Outbidding` dominates every model.** It alone accounts for
          roughly 55-68% of impurity-based importance and 70%+ of permutation
          importance in both tree models. The dataset is close to linearly
          separable on this one feature, so the ensembles' large accuracy gains
          over the baseline are modest in absolute terms even though they look
          large in relative terms.
        - No external validation set from a separate auction platform was
          available. All evaluation is on a held-out split of the same source
          dataset, so generalisation to a different platform's bidding patterns
          is untested.
        - Class imbalance (10.7% positive) means small changes in the
          classification threshold noticeably shift precision/recall trade-offs;
          the notebook explores this via precision-recall threshold scanning.
        - This demo predicts one bid at a time; a production system would need
          batch scoring and a monitoring pipeline for concept drift as bidding
          behaviour evolves.
        """
    )
