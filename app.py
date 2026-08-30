"""
Shill Bidding Risk Dashboard
Streamlit app: Overview, Explore the Data, Risk Predictor, Model Evaluation.
Built on the assignment notebook's pipeline; models are pre-trained
artifacts, so the app starts instantly and never retrains anything.
"""

import json
import math
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
# Palette (sampled from the concept mock-ups)
# ----------------------------------------------------------------------------
PAGE_BG = "#D9D9D9"
BANNER_BG = "#B5ABA1"
CARD_BG = "#F6F7F1"
STAT_BG = "#F1EBDF"
TEXT_DARK = "#1F1F1F"
ACCENT_GREEN = "#6EBE44"
ACCENT_ORANGE = "#F2A93B"
ACCENT_RED = "#E8495C"

# ----------------------------------------------------------------------------
# Global CSS — page background, top nav, banners, cards, stat squares
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: {PAGE_BG};
    }}
    [data-testid="stHeader"] {{
        background-color: #FFFFFF;
    }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }}

    /* ---- banner (page title) ---- */
    div[class*="st-key-banner_"] {{
        background-color: {BANNER_BG};
        border-radius: 26px;
        padding: 1.6rem 2.2rem 1.8rem 2.2rem;
        margin-bottom: 1.4rem;
    }}
    .banner-eyebrow {{
        font-size: 0.95rem;
        color: {TEXT_DARK};
        margin-bottom: 0.1rem;
        opacity: 0.85;
    }}
    .banner-title {{
        font-size: 2.4rem;
        font-weight: 600;
        color: {TEXT_DARK};
        line-height: 1.15;
    }}

    /* ---- generic cream card ---- */
    div[class*="st-key-card_"] {{
        background-color: {CARD_BG};
        border-radius: 22px;
        padding: 1.5rem 1.7rem;
        margin-bottom: 1.2rem;
    }}
    div[class*="st-key-card_"] h3, div[class*="st-key-card_"] h4 {{
        margin-top: 0;
    }}

    /* ---- outer tinted panel (wraps Problem / Remedies) ---- */
    div[class*="st-key-outer_"] {{
        background-color: {STAT_BG};
        border-radius: 22px;
        padding: 1.5rem 1.6rem 1.7rem 1.6rem;
        margin-bottom: 0rem;
    }}
    div[class*="st-key-outer_"] h3, div[class*="st-key-outer_"] h4, div[class*="st-key-outer_"] h5 {{
        margin-top: 0;
    }}

    /* ---- white inner box inside an outer panel ---- */
    div[class*="st-key-inner_"] {{
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        height: 100%;
    }}

    /* ---- small peach stat squares (Overview) ---- */
    div[class*="st-key-stat_"] {{
        background-color: {STAT_BG};
        border-radius: 22px;
        padding: 0.7rem 0.6rem;
        margin-bottom: 0.6rem;
        min-height: 128px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    div[class*="st-key-stat_"] [data-testid="stMarkdownContainer"] {{
        text-align: center !important;
        width: 100%;
    }}
    .stat-number {{
        font-size: 2.1rem;
        font-weight: 700;
        color: {TEXT_DARK};
        line-height: 1.05;
        text-align: center !important;
    }}
    .stat-label {{
        font-size: 0.95rem;
        color: {TEXT_DARK};
        opacity: 0.75;
        margin-top: 0.15rem;
        text-align: center !important;
    }}

    /* Scoped (NOT global) — tightens the gap between the two squares in
       each stat row, only inside the "stats_wrap" container. Everywhere
       else in the app, Streamlit's default column spacing is untouched. */
    div[class*="st-key-stats_wrap"] div[data-testid="stColumn"] {{
        padding: 0 0.3rem;
    }}

    /* top nav underline colour tweak */
    .nav-link-selected {{
        border-bottom: 3px solid {BANNER_BG} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def banner(key: str, eyebrow: str, title: str):
    """Solid taupe title banner, matching the concept mock-up."""
    with st.container(key=key):
        st.markdown(f'<div class="banner-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="banner-title">{title}</div>', unsafe_allow_html=True)


def card(key: str):
    """Returns a cream rounded container to use as `with card("x"):`."""
    return st.container(key=key)


# ----------------------------------------------------------------------------
# Animated risk gauge (custom SVG + CSS transition, embedded as an HTML
# component so the needle actually sweeps into place on every prediction)
# ----------------------------------------------------------------------------
def _polar(cx, cy, r, angle_deg):
    a = math.radians(angle_deg)
    return cx + r * math.cos(a), cy - r * math.sin(a)


def _band_path(cx, cy, r_outer, r_inner, a_start, a_end):
    x1, y1 = _polar(cx, cy, r_outer, a_start)
    x2, y2 = _polar(cx, cy, r_outer, a_end)
    x3, y3 = _polar(cx, cy, r_inner, a_end)
    x4, y4 = _polar(cx, cy, r_inner, a_start)
    return (
        f"M {x1:.2f} {y1:.2f} "
        f"A {r_outer} {r_outer} 0 0 1 {x2:.2f} {y2:.2f} "
        f"L {x3:.2f} {y3:.2f} "
        f"A {r_inner} {r_inner} 0 0 0 {x4:.2f} {y4:.2f} Z"
    )


def render_gauge(prob: float, height: int = 260):
    """Semicircular risk gauge: green (low) -> orange (mid) -> red (high),
    with a needle that animates from the middle into position."""
    cx, cy, r_out, r_in = 150, 145, 120, 78
    green = _band_path(cx, cy, r_out, r_in, 180, 120)
    orange = _band_path(cx, cy, r_out, r_in, 120, 60)
    red = _band_path(cx, cy, r_out, r_in, 60, 0)

    rotation = (max(0.0, min(1.0, prob)) - 0.5) * 180
    pct = prob * 100

    html = f"""
    <div style="display:flex; justify-content:center; align-items:center; width:100%;">
      <svg viewBox="0 0 300 190" width="340" height="{height}">
        <path d="{green}" fill="{ACCENT_GREEN}"></path>
        <path d="{orange}" fill="{ACCENT_ORANGE}"></path>
        <path d="{red}" fill="{ACCENT_RED}"></path>
        <g id="needle" style="transform-origin: {cx}px {cy}px; transform: rotate(0deg);
                               transition: transform 1.1s cubic-bezier(0.22, 1, 0.36, 1);">
          <line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - r_out + 22}"
                stroke="#1f1f1f" stroke-width="6" stroke-linecap="round"></line>
        </g>
        <circle cx="{cx}" cy="{cy}" r="10" fill="#1f1f1f"></circle>
        <text x="{cx}" y="{cy + 45}" text-anchor="middle"
              font-size="26" font-weight="700" fill="#1f1f1f">{pct:.1f}%</text>
      </svg>
    </div>
    <script>
      const needle = document.getElementById('needle');
      requestAnimationFrame(() => {{
        setTimeout(() => {{ needle.style.transform = 'rotate({rotation:.2f}deg)'; }}, 120);
      }});
    </script>
    """
    st.iframe(html, height=height + 20)


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

# Plain-language summary shown *inside* the Risk Predictor cards, so a
# non-technical reader can tell what they're dragging and why it matters.
FEATURE_SUMMARY = {
    "Bidder_Tendency": "How often this bidder shows up across many auctions from the *same* seller. "
                        "0 = spread out across different sellers. 1 = keeps returning to one seller — "
                        "a classic shill pattern.",
    "Bidding_Ratio": "Share of all the bids in this auction that came from this one bidder. "
                      "0 = barely bid. 1 = placed almost every bid in the auction (suspicious).",
    "Successive_Outbidding": "Did this bidder outbid themself or a partner back-to-back? "
                              "0 = never, 0.5 = once, 1 = repeatedly. The single strongest signal in the whole model.",
    "Last_Bidding": "How close to the auction's close this bidder's *last* bid landed. "
                     "Near 1 = bid right at the very end, which can be used to push the price up late.",
    "Auction_Bids": "How many bids the auction attracted overall, normalised. "
                     "Higher can mean genuine interest, or repeated shill activity inflating the count.",
    "Starting_Price_Average": "How the auction's starting price compares to the average starting price "
                               "for similar auctions. Far from 1 can signal manipulation of the opening price.",
    "Early_Bidding": "How close to the auction's *opening* this bidder's first bid landed. "
                      "Near 0 = jumped in immediately, which shill accounts often do to set a floor.",
    "Winning_Ratio": "Share of past auctions this bidder has actually won. "
                      "Shill accounts often bid a lot but rarely win (low ratio) since they aren't buying.",
    "Auction_Duration": "How many days the auction ran for. Shorter or unusually long auctions can change "
                         "how much room there is for shill behaviour.",
}

# ----------------------------------------------------------------------------
# Top navigation
# ----------------------------------------------------------------------------
page = option_menu(
    menu_title=None,
    options=["Overview", "Explore the data", "Risk predictor", "Model Evaluation"],
    icons=[""] * 4,
    menu_icon=None,
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#FFFFFF"},
        "icon": {"display": "none"},
        "nav-link": {
            "font-size": "17px",
            "font-weight": "500",
            "text-align": "center",
            "margin": "0px 18px",
            "padding": "18px 4px 14px 4px",
            "border-radius": "0px",
            "color": "#1f1f1f",
            "background-color": "transparent",
        },
        "nav-link-selected": {
            "background-color": "transparent",
            "color": "#1f1f1f",
            "font-weight": "600",
            "border-bottom": f"3px solid {BANNER_BG}",
        },
    },
)

# ----------------------------------------------------------------------------
# PAGE: Overview
# ----------------------------------------------------------------------------
if page == "Overview":
    banner("banner_overview", "Shill Bidding Risk", "Project Overview")

    df = load_clean_data()
    total_bids = len(df)
    normal_bids = int((df.Class == 0).sum())
    shill_bids = int((df.Class == 1).sum())
    shill_rate = df.Class.mean() * 100

    left, mid, right = st.columns([1.1, 1.6, 1.6])

    with left:
        with card("stats_wrap"):
            s1, s2 = st.columns(2)
            with s1:
                with card("stat_total"):
                    st.markdown(f'<div class="stat-number">{total_bids:,}</div>'
                                f'<div class="stat-label">Total bids</div>', unsafe_allow_html=True)
            with s2:
                with card("stat_normal"):
                    st.markdown(f'<div class="stat-number">{normal_bids:,}</div>'
                                f'<div class="stat-label">Normal bids</div>', unsafe_allow_html=True)
            s3, s4 = st.columns(2)
            with s3:
                with card("stat_shill"):
                    st.markdown(f'<div class="stat-number">{shill_bids:,}</div>'
                                f'<div class="stat-label">Shill bids</div>', unsafe_allow_html=True)
            with s4:
                with card("stat_rate"):
                    st.markdown(f'<div class="stat-number">{shill_rate:.1f}%</div>'
                                f'<div class="stat-label">Shill bidding rate</div>', unsafe_allow_html=True)

        with card("card_source"):
            st.markdown("**Data source**")
            st.markdown(
                "UCI Shill Bidding Dataset with 6,321 bids across 807 online "
                "auctions, 9 behavioural features per bid."
            )

    with mid:
        with card("card_business_problem"):
            st.subheader("Business problem")
            st.markdown(
                """
                A shill bidder is someone who places bids on their
                own listing with no intent to buy, purely to push genuine buyers into
                paying more. It's one of the hardest forms of auction fraud to catch,
                because a shill bidder tries to behave like an ordinary one.

                However, there are certain patterns that still give it away, such as
                bidding repeatedly on the same seller's auctions, jumping in right after 
                being outbid, bidding early to draw attention to a listing, and rarely 
                winning the auctions they enter, since winning would mean actually paying.

                Here, we turn those patterns into a screening tool by flagging
                bids that are likely to be shill bids so a platform's trust & safety team can
                prioritise which ones to review, rather than relying on buyer complaints
                or manual checks after the fact.
                """
            )

    with right:
        with card("card_models_compared"):
            st.subheader("Models compared")
            st.markdown(
                """
                | Model | Family | Role |
                |---|---|---|
                | Logistic Regression | Linear | **Baseline** |
                | SVM (poly kernel) | Max-margin | Comparison |
                | Random Forest | Tree ensemble (bagging) | Comparison |
                | Gradient Boosting | Tree ensemble (boosting) | Comparison |
                """
            )

# ----------------------------------------------------------------------------
# PAGE: Explore the data
# ----------------------------------------------------------------------------
elif page == "Explore the data":
    banner("banner_explore", "Shill Bidding Risk", "Explore the Data")
    df = load_clean_data()

    st.markdown("##### Target Imbalance")
    col1, col2 = st.columns([1, 2.6])
    with col1:
        with card("card_class_balance"):
            st.markdown("**Class distribution**")
            fig, ax = plt.subplots(figsize=(4, 4))
            df["Class"].value_counts().sort_index().plot(
                kind="bar", ax=ax, color=["#FBDA0C", "#0057AD"]
            )
            ax.set_xticklabels(["Normal (0)", "Shill (1)"], rotation=0)
            ax.set_ylabel("Count")
            fig.patch.set_alpha(0)
            st.pyplot(fig)

    with col2:
        with st.container(key="outer_problem"):
            st.markdown("**Problem**")
            with st.container(key="inner_problem"):
                st.markdown(
                    """
                    Only ~10.7% of bids are shill bids. A model could just guess "normal" 
                    every time and still look ~89% accurate without learning anything.
                    """
                )

        with st.container(key="outer_remedies"):
            st.markdown("**Remedies**")
            r1, r2, r3 = st.columns(3)
            with r1:
                with st.container(key="inner_remedy1"):
                    st.markdown("**1. Class weighting**")
                    st.markdown(
                        "Every model was trained with `class_weight=\"balanced\"` "
                    )
            with r2:
                with st.container(key="inner_remedy2"):
                    st.markdown("**2. Stratified split**")
                    st.markdown(
                        "Train/test split kept the same 89.32% / 10.68% ratio in both sets."
                    )
            with r3:
                with st.container(key="inner_remedy3"):
                    st.markdown("**3. Metric tuning**")
                    st.markdown(
                        "Used PR-AUC instead of accuracy or ROC-AUC to tune hyperparameters."
                    )

    st.markdown("##### Feature Distributions")
    with card("card_feature_select"):
        feature = st.selectbox(
            "Choose a feature", FEATURES,
            format_func=lambda x: x.replace("_", " ")
        )
        st.caption(FEATURE_HELP.get(feature, ""))

    col1, col2 = st.columns(2)
    with col1:
        with card("card_feature_hist"):
            fig, ax = plt.subplots(figsize=(5.2, 4))
            sns.histplot(data=df, x=feature, hue="Class", kde=True, ax=ax,
                         palette=["#FBDA0C", "#0057AD"], multiple="layer", alpha=0.5)
            ax.set_title(f"{feature.replace('_', ' ')} distribution by class")
            ax.set_xlabel(feature.replace("_", " "))
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)
    with col2:
        with card("card_feature_box"):
            fig, ax = plt.subplots(figsize=(5.2, 4))
            sns.boxplot(data=df, x="Class", y=feature, hue="Class", ax=ax,
                        palette=["#FBDA0C", "#0057AD"], legend=False)
            ax.set_xticklabels(["Normal", "Shill"])
            ax.set_title(f"{feature.replace('_', ' ')} by class")
            ax.set_ylabel(feature.replace("_", " "))
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("##### Correlations")
    with card("card_correlations"):
        st.markdown("**Correlation matrix (all features + target)**")
        corr = df.corr()
        corr_display = corr.rename(
            index=lambda x: x.replace("_", " "),
            columns=lambda x: x.replace("_", " "),
        )
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr_display, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    annot_kws={"size": 8}, ax=ax)
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        st.caption(
            "`Successive_Outbidding` shows the strongest correlation with `Class` "
            "by a wide margin. This is confirmed by every model's feature "
            "importance ranking on the Model Evaluation page."
        )

# ----------------------------------------------------------------------------
# PAGE: Risk predictor
# ----------------------------------------------------------------------------
elif page == "Risk predictor":
    banner("banner_predictor", "Shill Bidding Risk", "Risk Predictor")

    values = {}
    left, right = st.columns([1.5, 1])

    with left:
        with card("card_model_choice"):
            model_name = st.selectbox("Model to use", list(MODELS.keys()))

        with card("card_sliders"):
            st.markdown("##### Bid features")
            fc1, fc2 = st.columns(2)
            cols_per_col = [FEATURES[0:5], FEATURES[5:9]]

            for col_container, feats in zip([fc1, fc2], cols_per_col):
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

            predict_clicked = st.button("Predict risk", type="primary")

    with right:
        with card("card_gauge"):
            if predict_clicked:
                X_input = pd.DataFrame([values])[FEATURES]
                model = MODELS[model_name]
                if model_name in SCALED_MODELS:
                    X_input = pd.DataFrame(SCALER.transform(X_input), columns=FEATURES)
                prob = float(model.predict_proba(X_input)[0, 1])
                st.session_state["last_prob"] = prob

            prob = st.session_state.get("last_prob", 0.0)
            label = "Likely shill bidder" if prob >= 0.5 else "Likely normal bidder"

            render_gauge(prob)
            st.metric("Predicted shill-bidding probability", f"{prob * 100:.1f}%")
            st.markdown(f"**Model prediction:** {label}")
            st.caption(
                "Gauge: green = low risk, orange = medium, red = high risk of shill bidding, "
                "based on the selected model's predicted probability."
            )

# ----------------------------------------------------------------------------
# PAGE: Model Evaluation
# ----------------------------------------------------------------------------
elif page == "Model Evaluation":
    banner("banner_eval", "Shill Bidding Risk", "Model Evaluation")

    X_test, X_test_scaled, y_test = load_test_set()
    results_precomputed = load_results_table()

    row1a, row1b = st.columns(2)

    with row1a:
        with card("card_metrics_table"):
            st.markdown("**Metrics on the held-out test set**")
            st.dataframe(results_precomputed.style.format("{:.4f}").highlight_max(axis=0, color="#FCEF9A"),
                         width="stretch")

    with row1b:
        with card("card_metric_bar"):
            st.markdown("**Compare models on a metric**")
            metric_choice = st.selectbox(
                "metric_choice",
                ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"],
                label_visibility="collapsed",
            )
            fig, ax = plt.subplots(figsize=(5, 4))
            results_precomputed[metric_choice].sort_values().plot(kind="barh", ax=ax, color="#FBDA0C")
            ax.set_xlabel(metric_choice)
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)

    row2a, row2b = st.columns(2)

    with row2a:
        with card("card_roc"):
            st.markdown("**ROC curves**")
            fig, ax = plt.subplots(figsize=(5, 4))
            for name, model in MODELS.items():
                X_te = X_test_scaled if name in SCALED_MODELS else X_test
                prob = model.predict_proba(X_te)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, prob)
                auc_val = roc_auc_score(y_test, prob)
                ax.plot(fpr, tpr, label=f"{name.split(' ')[0]} ({auc_val:.2f})")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(fontsize=6)
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig)

    with row2b:
        with card("card_confusion"):
            st.subheader("Confusion matrix")
            cm_model_name = st.selectbox("Model", list(MODELS.keys()), key="cm_model")
            model = MODELS[cm_model_name]
            X_te = X_test_scaled if cm_model_name in SCALED_MODELS else X_test
            y_pred = model.predict(X_te)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Normal", "Shill"], yticklabels=["Normal", "Shill"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            fig.patch.set_alpha(0)
            st.pyplot(fig)

    fi1, fi2 = st.columns(2)
    with fi1:
        with card("card_fi_rf"):
            st.markdown("**Feature importance — Random Forest**")
            rf_imp = pd.Series(
                MODELS["Random Forest"].feature_importances_,
                index=[f.replace("_", " ") for f in FEATURES]
            ).sort_values()
            fig, ax = plt.subplots(figsize=(5, 4))
            rf_imp.plot(kind="barh", ax=ax, color="#55A868")
            fig.patch.set_alpha(0)
            st.pyplot(fig)
    with fi2:
        with card("card_fi_gb"):
            st.markdown("**Feature importance — Gradient Boosting**")
            gb_imp = pd.Series(
                MODELS["Gradient Boosting"].feature_importances_,
                index=[f.replace("_", " ") for f in FEATURES]
            ).sort_values()
            fig, ax = plt.subplots(figsize=(5, 4))
            gb_imp.plot(kind="barh", ax=ax, color="#0057AD")
            fig.patch.set_alpha(0)
            st.pyplot(fig)

    with card("card_limitations"):
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
            """
        )
