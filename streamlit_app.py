import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Player Selection Predictor",
    page_icon="⚽",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────
@st.cache_resource
def load_resources():
    df = pd.read_csv("final_data.csv", low_memory=False)
    df["player_name"] = df["player_name"].astype(str).str.lower().str.strip()
    model = load_model("best_transformer.keras", compile=False)
    scaler = joblib.load("scaler.pkl")
    col_means = joblib.load("col_means.pkl")   # ← training wale means
    return df, model, scaler, col_means

try:
    df, model, scaler, col_means = load_resources()
except Exception as e:
    st.error(f"❌ Failed to load resources: {e}")
    st.stop()

# ─────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────
train_columns = [
    "performance_score",
    "offensive_skill",
    "defensive_skill",
    "efficiency",
    "aggression",
    "mistake_penalty"
]

# ─────────────────────────────────────────
# HELPER: PREPARE FEATURES (same as training)
# ─────────────────────────────────────────
def prepare_features(player_row):
    features = player_row[train_columns].copy()
    features = features.apply(pd.to_numeric, errors="coerce")
    # fillna with training means — exactly like training
    features = features.fillna(col_means)
    features = features.fillna(0)  # fallback agar mean bhi na ho
    return features

# ─────────────────────────────────────────
# HELPER: GET SCORES FOR ALL PLAYERS
# ─────────────────────────────────────────
@st.cache_data
def get_all_scores():
    features = df[train_columns].apply(pd.to_numeric, errors="coerce")
    features = features.fillna(col_means).fillna(0)
    X = scaler.transform(features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    scores = model.predict(X, verbose=0).flatten()
    result = df.copy()
    result["Confidence"] = scores
    result["Prediction"] = result["Confidence"].apply(
        lambda x: "Selected ✅" if x >= 0.5 else "Not Selected ❌"
    )
    return result

# ─────────────────────────────────────────
# HELPER: PREDICT SINGLE PLAYER
# ─────────────────────────────────────────
def predict_player(name):
    player = df[df["player_name"] == name.strip().lower()]
    if player.empty:
        return None, None, None
    features = prepare_features(player.head(1))
    X = scaler.transform(features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    prob = float(model.predict(X, verbose=0)[0][0])
    result = "Selected ✅" if prob >= 0.5 else "Not Selected ❌"
    return result, prob, features

# ─────────────────────────────────────────
# SIDEBAR NAV
# ─────────────────────────────────────────
st.sidebar.title("⚽ Player Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "🔮 Predict Player",
    "📊 Prediction Probability",
    "⚖️ Compare Players",
    "🏆 Top Players",
    "💀 Worst Players",
    "👥 Best Team Selector",
    "📈 Selection Statistics",
    "🔍 Search Player",
    "🥇 Player Rank",
    "🧠 Why Selected",
])
st.sidebar.markdown("---")
st.sidebar.success("✅ Model loaded")
st.sidebar.caption(f"Total players: {df['player_name'].nunique()}")


# ═════════════════════════════════════════
# PAGE 1 — PREDICT PLAYER
# ═════════════════════════════════════════
if page == "🔮 Predict Player":

    st.title("🔮 Player Prediction")
    st.markdown("Predict using the AI ​​model whether the player will be selected or not.")
    st.divider()

    player_name = st.text_input("Player Name", placeholder="e.g. Player_0")

    if st.button("Predict", use_container_width=True, type="primary"):
        if not player_name.strip():
            st.warning("Player name enter karo.")
        else:
            result, prob, features = predict_player(player_name)
            if result is None:
                st.error(f"❌ Player '{player_name}' not found.")
            else:
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Result: {result}")
                    st.metric("Confidence Score", f"{prob:.4f}")
                    st.metric("Confidence %", f"{prob*100:.2f}%")
                    st.progress(float(prob))
                with col2:
                    st.subheader("📊 Player Stats")
                    st.dataframe(features.reset_index(drop=True), use_container_width=True)


# ═════════════════════════════════════════
# PAGE 2 — PREDICTION PROBABILITY
# ═════════════════════════════════════════
elif page == "📊 Prediction Probability":

    st.title("📊 Prediction Probability")
    st.markdown("Predict using the AI ​​model")
    st.divider()

    player_name = st.text_input("Player Name", placeholder="e.g. Player_0")

    if st.button("Get Probability", use_container_width=True, type="primary"):
        if not player_name.strip():
            st.warning("Player name enter karo.")
        else:
            result, prob, features = predict_player(player_name)
            if result is None:
                st.error(f"❌ Player '{player_name}' not found.")
                st.info("Available players (sample):")
                st.write(df["player_name"].head(10).tolist())
            else:
                percentage = prob * 100
                if prob >= 0.90:
                    level = "⭐ Elite Player"
                elif prob >= 0.75:
                    level = "🔥 Excellent Player"
                elif prob >= 0.60:
                    level = "✅ Good Player"
                elif prob >= 0.40:
                    level = "⚠️ Average Player"
                else:
                    level = "❌ Weak Player"

                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("Selection Probability", f"{prob:.4f}")
                col2.metric("Selection %", f"{percentage:.2f}%")
                col3.metric("Performance Level", level)
                st.progress(float(prob))


# ═════════════════════════════════════════
# PAGE 3 — COMPARE PLAYERS
# ═════════════════════════════════════════
elif page == "⚖️ Compare Players":

    st.title("⚖️ Compare Two Players")
    st.markdown("Perform a side-by-side comparison of the two players.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        p1 = st.text_input("Player 1", placeholder="e.g. Player_0")
    with col2:
        p2 = st.text_input("Player 2", placeholder="e.g. Player_1")

    if st.button("Compare", use_container_width=True, type="primary"):
        if not p1.strip() or not p2.strip():
            st.warning("Dono players ka naam enter karo.")
        else:
            r1, prob1, f1 = predict_player(p1)
            r2, prob2, f2 = predict_player(p2)

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"👤 {p1.title()}")
                if r1 is None:
                    st.error("Player not found")
                else:
                    st.metric("Prediction", r1)
                    st.metric("Confidence", f"{prob1:.4f}")
                    st.progress(float(prob1))

            with col2:
                st.subheader(f"👤 {p2.title()}")
                if r2 is None:
                    st.error("Player not found")
                else:
                    st.metric("Prediction", r2)
                    st.metric("Confidence", f"{prob2:.4f}")
                    st.progress(float(prob2))

            if r1 is not None and r2 is not None:
                st.divider()
                diff = abs(prob1 - prob2)
                if prob1 > prob2:
                    st.success(f"🏆 Better Player: **{p1.title()}** (Difference: {diff:.4f})")
                elif prob2 > prob1:
                    st.success(f"🏆 Better Player: **{p2.title()}** (Difference: {diff:.4f})")
                else:
                    st.info("🤝 Both players are equal!")

                st.subheader("📊 Stats Comparison")
                compare_df = pd.DataFrame({
                    p1.title(): f1[train_columns].values[0],
                    p2.title(): f2[train_columns].values[0]
                }, index=train_columns)
                st.dataframe(compare_df, use_container_width=True)
                st.bar_chart(compare_df.T)


# ═════════════════════════════════════════
# PAGE 4 — TOP PLAYERS
# ═════════════════════════════════════════
elif page == "🏆 Top Players":

    st.title("🏆 Top Players")
    st.markdown("Top players based on confidence scores.")
    st.divider()

    n = st.slider("Top N Players", min_value=5, max_value=50, value=10)

    if st.button("Show Top Players", use_container_width=True, type="primary"):
        with st.spinner("Calculating scores..."):
            all_scores = get_all_scores()
            top = all_scores.sort_values("Confidence", ascending=False).head(n)
            top = top[["player_name", "Confidence", "Prediction"]].reset_index(drop=True)
            top.index += 1

        st.dataframe(top, use_container_width=True)
        st.subheader(f"📊 Top {n} Players — Confidence Chart")
        st.bar_chart(top.set_index("player_name")["Confidence"])


# ═════════════════════════════════════════
# PAGE 5 — WORST PLAYERS
# ═════════════════════════════════════════
elif page == "💀 Worst Players":

    st.title("💀 Worst Players")
    st.markdown("Players with low confidence scores.")
    st.divider()

    n = st.slider("Bottom N Players", min_value=5, max_value=50, value=10)

    if st.button("Show Worst Players", use_container_width=True, type="primary"):
        with st.spinner("Calculating scores..."):
            all_scores = get_all_scores()
            worst = all_scores.sort_values("Confidence", ascending=True).head(n)
            worst = worst[["player_name", "Confidence", "Prediction"]].reset_index(drop=True)
            worst.index += 1

        st.dataframe(worst, use_container_width=True)
        st.subheader(f"📊 Bottom {n} Players — Confidence Chart")
        st.bar_chart(worst.set_index("player_name")["Confidence"])


# ═════════════════════════════════════════
# PAGE 6 — BEST TEAM SELECTOR
# ═════════════════════════════════════════
elif page == "👥 Best Team Selector":

    st.title("👥 Best Team Selector")
    st.markdown("Select the 'Best XI' or a custom-sized team.")
    st.divider()

    team_size = st.slider("Team Size", min_value=5, max_value=22, value=11)

    if st.button("Select Best Team", use_container_width=True, type="primary"):
        with st.spinner("Selecting best team..."):
            all_scores = get_all_scores()
            team = all_scores.sort_values("Confidence", ascending=False).head(team_size)
            team = team[["player_name", "Confidence", "Prediction"]].reset_index(drop=True)
            team.index += 1

        st.success(f"✅ Best {team_size} players selected!")
        st.dataframe(team, use_container_width=True)
        st.subheader("📊 Team Confidence Chart")
        st.bar_chart(team.set_index("player_name")["Confidence"])


# ═════════════════════════════════════════
# PAGE 7 — SELECTION STATISTICS
# ═════════════════════════════════════════
elif page == "📈 Selection Statistics":

    st.title("📈 Selection Statistics")
    st.markdown("Overall selection rate and breakdowns for the entire database.")
    st.divider()

    if st.button("Calculate Statistics", use_container_width=True, type="primary"):
        with st.spinner("Calculating..."):
            all_scores = get_all_scores()
            probs = all_scores["Confidence"].values
            selected = int(np.sum(probs >= 0.5))
            rejected = int(np.sum(probs < 0.5))
            total = len(df)
            rate = round(selected / total * 100, 2)

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Selected", selected)
        col2.metric("❌ Not Selected", rejected)
        col3.metric("📊 Selection Rate", f"{rate}%")

        st.divider()
        st.subheader("Selected vs Not Selected")
        chart_df = pd.DataFrame({"Count": [selected, rejected]}, index=["Selected", "Not Selected"])
        st.bar_chart(chart_df)

        st.subheader("Confidence Score Distribution")
        hist_df = pd.DataFrame({"Confidence": probs})
        st.bar_chart(hist_df["Confidence"].value_counts(bins=10).sort_index())


# ═════════════════════════════════════════
# PAGE 8 — SEARCH PLAYER
# ═════════════════════════════════════════
elif page == "🔍 Search Player":

    st.title("🔍 Search Player")
    st.markdown("Search by player name or alias.")
    st.divider()

    search = st.text_input("Search", placeholder="e.g. Player_1")

    if st.button("Search", use_container_width=True, type="primary"):
        if not search.strip():
            st.warning("Kuch toh likho search mein.")
        else:
            result = df[df["player_name"].str.contains(search.strip().lower(), na=False)]
            if result.empty:
                st.error("❌ No players found.")
            else:
                st.success(f"✅ {len(result)} player(s) found")
                st.dataframe(
                    result[["player_name"]].drop_duplicates().reset_index(drop=True),
                    use_container_width=True
                )


# ═════════════════════════════════════════
# PAGE 9 — PLAYER RANK
# ═════════════════════════════════════════
elif page == "🥇 Player Rank":

    st.title("🥇 Player Rank")
    st.markdown("Find out the player's overall rank in the database.")
    st.divider()

    player_name = st.text_input("Player Name", placeholder="e.g. Player_0")

    if st.button("Get Rank", use_container_width=True, type="primary"):
        if not player_name.strip():
            st.warning("Player name enter karo.")
        else:
            with st.spinner("Calculating rank..."):
                all_scores = get_all_scores()
                ranking = all_scores.sort_values("Confidence", ascending=False).reset_index(drop=True)
                ranking["Rank"] = ranking.index + 1
                player = ranking[ranking["player_name"] == player_name.strip().lower()]

            if player.empty:
                st.error(f"❌ Player '{player_name}' not found.")
            else:
                row = player.iloc[0]
                col1, col2, col3 = st.columns(3)
                col1.metric("🥇 Rank", f"#{int(row['Rank'])}")
                col2.metric("Confidence", f"{row['Confidence']:.4f}")
                col3.metric("Prediction", row["Prediction"])
                total = len(ranking)
                percentile = round((1 - row["Rank"] / total) * 100, 1)
                st.info(f"📊 Player is in top **{percentile}%** of all {total} players.")


# ═════════════════════════════════════════
# PAGE 10 — WHY SELECTED
# ═════════════════════════════════════════
elif page == "🧠 Why Selected":

    st.title("🧠 Why Selected / Not Selected")
    st.markdown("A detailed analysis — why a player was selected or rejected.")
    st.divider()

    player_name = st.text_input("Player Name", placeholder="e.g. Player_0")

    if st.button("Analyze", use_container_width=True, type="primary"):
        if not player_name.strip():
            st.warning("Player name enter karo.")
        else:
            result, prob, features = predict_player(player_name)
            if result is None:
                st.error(f"❌ Player '{player_name}' not found.")
            else:
                st.divider()
                col1, col2 = st.columns(2)
                col1.metric("Prediction", result)
                col2.metric("Confidence", f"{prob:.4f}")
                st.progress(float(prob))

                st.subheader("🔍 Top Reasons (Highest Feature Values)")
                values = features[train_columns].iloc[0]
                values = pd.to_numeric(values, errors="coerce")
                top_features = values.sort_values(ascending=False).head(5)
                for feature, value in top_features.items():
                    st.write(f"✔ **{feature}**: {value:.4f}")

                st.subheader("🤖 Model Decision")
                if prob >= 0.80:
                    st.success("⭐ Excellent candidate for selection")
                elif prob >= 0.60:
                    st.success("✅ Good chance of selection")
                elif prob >= 0.50:
                    st.warning("⚠️ Borderline selection")
                else:
                    st.error("❌ Low chance of selection")

                st.subheader("📊 All Feature Values")
                feature_df = pd.DataFrame({
                    "Feature": train_columns,
                    "Value": [float(pd.to_numeric(features[c].values[0], errors="coerce")) for c in train_columns]
                })
                st.dataframe(feature_df, use_container_width=True)
                st.bar_chart(feature_df.set_index("Feature")["Value"])