from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from sklearn.metrics.pairwise import cosine_distances


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "analysis" / "output"
VIZ_DIR = OUTPUT_DIR / "visualizations"


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "memos": pd.read_csv(OUTPUT_DIR / "memos.csv"),
        "themes": pd.read_csv(OUTPUT_DIR / "themes.csv"),
        "references": pd.read_csv(OUTPUT_DIR / "references.csv"),
        "stance": pd.read_csv(OUTPUT_DIR / "stance.csv"),
        "prediction": pd.read_csv(OUTPUT_DIR / "prediction_style.csv"),
        "clusters": pd.read_csv(OUTPUT_DIR / "clusters.csv"),
        "important": pd.read_csv(OUTPUT_DIR / "important_memos.csv"),
        "change": pd.read_csv(OUTPUT_DIR / "change_report.csv"),
    }


def prepare_memo_frame(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    memos = data["memos"].copy()
    stance = data["stance"].copy()
    prediction = data["prediction"].copy()
    clusters = data["clusters"][["memo_id", "cluster_id", "cluster_label"]].copy()
    important = data["important"][["memo_id", "importance_score", "incoming_refs", "outgoing_refs"]].copy()
    memos["date"] = pd.to_datetime(memos["date"])
    stance["date"] = pd.to_datetime(stance["date"])
    prediction["date"] = pd.to_datetime(prediction["date"])
    df = (
        memos.merge(stance, on=["memo_id", "title", "date", "period"], how="left")
        .merge(prediction, on=["memo_id", "title", "date", "period"], how="left")
        .merge(clusters, on="memo_id", how="left")
        .merge(important, on="memo_id", how="left")
    )
    stance_total = (
        df["constructive_score"].fillna(0)
        + df["opportunistic_score"].fillna(0)
        + df["alarmed_score"].fillna(0)
        + df["cautious_score"].fillna(0)
        + df["humble_score"].fillna(0)
    )
    df["bullish_bearish_score"] = (
        df["constructive_score"].fillna(0)
        + df["opportunistic_score"].fillna(0)
        - df["alarmed_score"].fillna(0)
        - df["cautious_score"].fillna(0)
        - 0.5 * df["humble_score"].fillna(0)
    ) / stance_total.replace(0, np.nan)
    df["bullish_bearish_score"] = df["bullish_bearish_score"].fillna(0.0)
    return df.sort_values("date").reset_index(drop=True)


def save_figure(fig: go.Figure, filename: str) -> None:
    fig.update_layout(template="plotly_white")
    fig.write_html(VIZ_DIR / filename, include_plotlyjs="cdn")


def build_network_graph(df: pd.DataFrame, references: pd.DataFrame) -> None:
    ordered = df.sort_values("date").reset_index(drop=True).copy()
    ordered["y"] = 0.0
    ordered["year"] = ordered["date"].dt.year
    ref = references.copy()
    ref["source_date"] = pd.to_datetime(ref["source_date"])
    ref["target_date"] = pd.to_datetime(ref["target_date"])
    valid = ref[ref["source_date"] >= ref["target_date"]].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ordered["date"],
            y=ordered["y"],
            mode="lines",
            line=dict(color="rgba(70,70,70,0.35)", width=2),
            hoverinfo="none",
            showlegend=False,
        )
    )

    max_edges = min(len(valid), 400)
    if max_edges > 0:
        valid = valid.sort_values(["source_date", "target_date"]).tail(max_edges)
        meta = ordered.set_index("memo_id")[["date"]].to_dict("index")
        for row in valid.itertuples():
            if row.source_id not in meta or row.target_id not in meta:
                continue
            x0 = meta[row.source_id]["date"]
            x1 = meta[row.target_id]["date"]
            span_days = max((x0 - x1).days, 1)
            arc_height = min(0.18 + 0.0018 * span_days, 0.95)
            fig.add_annotation(
                x=x1,
                y=arc_height,
                ax=x0,
                ay=arc_height,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="rgba(139,58,43,0.28)",
                standoff=2,
                startstandoff=2,
                opacity=0.9,
                arrowside="end",
            )

    hover = [
        "<br>".join(
            [
                row.title,
                f"Date: {row.date.strftime('%Y-%m-%d')}",
                f"Incoming refs: {int(row.incoming_refs)}",
                f"Outgoing refs: {int(row.outgoing_refs)}",
                f"Cluster: {row.cluster_label}",
            ]
        )
        for row in ordered.itertuples()
    ]
    fig.add_trace(
        go.Scatter(
            x=ordered["date"],
            y=ordered["y"],
            mode="markers",
            hovertext=hover,
            hoverinfo="text",
            marker=dict(
                size=8 + ordered["incoming_refs"].fillna(0) * 1.6 + ordered["importance_score"].fillna(0) * 0.02,
                color=ordered["year"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Year"),
                line=dict(width=0.8, color="rgba(40,40,40,0.55)"),
                opacity=0.9,
            ),
            name="memos",
        )
    )
    fig.update_layout(
        title="Howard Marks Memos Timeline With Back-References",
        xaxis=dict(title="Date"),
        yaxis=dict(visible=False, range=[-0.25, 1.1]),
        hovermode="closest",
        margin=dict(l=20, r=20, t=60, b=40),
    )
    save_figure(fig, "memo_network.html")


def build_theme_visuals(themes: pd.DataFrame, change: pd.DataFrame) -> None:
    themes = themes.copy()
    themes["date"] = pd.to_datetime(themes["date"])
    themes["year"] = themes["date"].dt.year

    top_themes = (
        themes.groupby("theme", as_index=False)["hits"]
        .sum()
        .sort_values("hits", ascending=False)
        .head(8)["theme"]
        .tolist()
    )
    yearly = (
        themes[themes["theme"].isin(top_themes)]
        .groupby(["year", "theme"], as_index=False)["hits_per_1k_words"]
        .mean()
    )
    fig_trends = px.line(
        yearly,
        x="year",
        y="hits_per_1k_words",
        color="theme",
        markers=True,
        title="Theme Intensity Over Time",
        labels={"hits_per_1k_words": "Average hits per 1k words"},
    )
    save_figure(fig_trends, "theme_trends.html")

    change_sorted = change.sort_values("delta_1990s_to_2020s")
    fig_change = px.bar(
        change_sorted,
        x="delta_1990s_to_2020s",
        y="theme",
        orientation="h",
        color="delta_1990s_to_2020s",
        color_continuous_scale="RdBu",
        title="Biggest Thematic Increases / Decreases: 1990s to 2020s",
        labels={"delta_1990s_to_2020s": "Delta in avg hits per 1k words"},
    )
    fig_change.update_layout(coloraxis_showscale=False)
    save_figure(fig_change, "theme_change_bars.html")


def fetch_sp500_history(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    history = yf.download("^GSPC", start=start.date().isoformat(), end=(end + pd.Timedelta(days=10)).date().isoformat(), progress=False)
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = [col[0].lower() for col in history.columns]
    else:
        history.columns = [str(col).lower() for col in history.columns]
    history = history.reset_index()
    history = history.rename(columns={"Date": "date", "date": "date", "close": "sp500_close"})
    history["date"] = pd.to_datetime(history["date"])
    history = history[["date", "sp500_close"]].dropna()
    return history


def build_sentiment_vs_sp500(df: pd.DataFrame) -> None:
    memo_monthly = (
        df.set_index("date")["bullish_bearish_score"]
        .resample("MS")
        .mean()
        .rolling(3, min_periods=1)
        .mean()
        .reset_index()
    )
    sp500 = fetch_sp500_history(df["date"].min(), df["date"].max())
    sp500_monthly = (
        sp500.set_index("date")["sp500_close"]
        .resample("MS")
        .last()
        .ffill()
        .reset_index()
    )
    base = float(sp500_monthly["sp500_close"].iloc[0])
    sp500_monthly["sp500_indexed"] = sp500_monthly["sp500_close"] / base * 100.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sp500_monthly["date"],
            y=sp500_monthly["sp500_indexed"],
            mode="lines",
            name="S&P 500 (indexed to 100)",
            line=dict(color="#1f77b4", width=2.2),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=memo_monthly["date"],
            y=memo_monthly["bullish_bearish_score"],
            mode="lines+markers",
            name="Memo bullish/bearish score",
            line=dict(color="#d62728", width=2.2),
            marker=dict(size=5),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Memo Tone vs. S&P 500",
        xaxis=dict(title="Date"),
        yaxis=dict(title="S&P 500 (indexed)", side="left"),
        yaxis2=dict(title="Bullish/Bearish score", overlaying="y", side="right", zeroline=True),
        hovermode="x unified",
    )
    save_figure(fig, "sentiment_vs_sp500.html")


def build_feature_frame(df: pd.DataFrame, themes: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    theme_pivot = (
        themes.pivot_table(index="memo_id", columns="theme", values="hits_per_1k_words", fill_value=0.0)
        .reset_index()
    )
    merged = df.merge(theme_pivot, on="memo_id", how="left").sort_values("date").reset_index(drop=True)
    tone_cols = [
        "cautious_score",
        "opportunistic_score",
        "alarmed_score",
        "constructive_score",
        "humble_score",
        "forecasting_score",
        "warning_score",
        "preparation_score",
        "anti_prediction_score",
    ]
    theme_cols = [col for col in theme_pivot.columns if col != "memo_id"]
    return merged, tone_cols + theme_cols, theme_cols


def summarize_theme_delta(earlier: pd.Series, later: pd.Series, theme_cols: list[str]) -> tuple[str, str]:
    deltas = {theme: float(later[theme] - earlier[theme]) for theme in theme_cols}
    up = [item for item in sorted(deltas.items(), key=lambda x: x[1], reverse=True) if item[1] > 0.15][:3]
    down = [item for item in sorted(deltas.items(), key=lambda x: x[1]) if item[1] < -0.15][:3]
    up_text = ", ".join(f"{theme} (+{value:.2f})" for theme, value in up) if up else "none"
    down_text = ", ".join(f"{theme} ({value:.2f})" for theme, value in down) if down else "none"
    return up_text, down_text


def explain_shift(earlier: pd.Series, later: pd.Series, theme_cols: list[str]) -> str:
    parts = []
    if earlier["dominant_stance"] != later["dominant_stance"]:
        parts.append(f"stance moves from {earlier['dominant_stance']} to {later['dominant_stance']}")
    if earlier["dominant_prediction_style"] != later["dominant_prediction_style"]:
        parts.append(
            f"prediction posture moves from {earlier['dominant_prediction_style']} to {later['dominant_prediction_style']}"
        )
    sentiment_delta = later["bullish_bearish_score"] - earlier["bullish_bearish_score"]
    if sentiment_delta > 0.2:
        parts.append("later memo is materially more constructive / bullish in tone")
    elif sentiment_delta < -0.2:
        parts.append("later memo is materially more defensive / bearish in tone")
    up_text, down_text = summarize_theme_delta(earlier, later, theme_cols)
    if up_text != "none":
        parts.append(f"greater emphasis on {up_text}")
    if down_text != "none":
        parts.append(f"less emphasis on {down_text}")
    return "; ".join(parts) if parts else "shift is driven by a broader change in feature mix rather than one dominant dimension"


def build_tone_shift_report(df: pd.DataFrame, themes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged, feature_cols, theme_cols = build_feature_frame(df, themes)
    vectors = merged[feature_cols].fillna(0.0).to_numpy()
    distances = cosine_distances(vectors[:-1], vectors[1:]).diagonal()
    rows = []
    for idx, distance in enumerate(distances, start=1):
        prev_row = merged.iloc[idx - 1]
        next_row = merged.iloc[idx]
        sentiment_delta = float(next_row["bullish_bearish_score"] - prev_row["bullish_bearish_score"])
        increased_themes, decreased_themes = summarize_theme_delta(prev_row, next_row, theme_cols)
        rows.append(
            {
                "from_title": prev_row["title"],
                "from_date": prev_row["date"].strftime("%Y-%m-%d"),
                "to_title": next_row["title"],
                "to_date": next_row["date"].strftime("%Y-%m-%d"),
                "days_between": int((next_row["date"] - prev_row["date"]).days),
                "tone_shift_score": round(float(distance), 4),
                "sentiment_delta": round(sentiment_delta, 4),
                "from_stance": prev_row["dominant_stance"],
                "to_stance": next_row["dominant_stance"],
                "from_prediction_style": prev_row["dominant_prediction_style"],
                "to_prediction_style": next_row["dominant_prediction_style"],
                "increased_themes": increased_themes,
                "decreased_themes": decreased_themes,
                "explanation": explain_shift(prev_row, next_row, theme_cols),
            }
        )
    shifts = pd.DataFrame(rows).sort_values("tone_shift_score", ascending=False)
    shifts.to_csv(OUTPUT_DIR / "tone_shifts.csv", index=False)

    long_rows = []
    for i in range(len(merged)):
        earlier = merged.iloc[i]
        for j in range(i + 1, len(merged)):
            later = merged.iloc[j]
            years_apart = (later["date"] - earlier["date"]).days / 365.25
            if years_apart < 5:
                continue
            sentiment_delta = float(later["bullish_bearish_score"] - earlier["bullish_bearish_score"])
            distance = float(cosine_distances([vectors[i]], [vectors[j]])[0][0])
            score = distance + 0.25 * abs(sentiment_delta)
            if earlier["dominant_stance"] != later["dominant_stance"]:
                score += 0.08
            if earlier["dominant_prediction_style"] != later["dominant_prediction_style"]:
                score += 0.06
            increased_themes, decreased_themes = summarize_theme_delta(earlier, later, theme_cols)
            long_rows.append(
                {
                    "from_title": earlier["title"],
                    "from_date": earlier["date"].strftime("%Y-%m-%d"),
                    "to_title": later["title"],
                    "to_date": later["date"].strftime("%Y-%m-%d"),
                    "years_between": round(years_apart, 1),
                    "long_shift_score": round(score, 4),
                    "sentiment_delta": round(sentiment_delta, 4),
                    "from_stance": earlier["dominant_stance"],
                    "to_stance": later["dominant_stance"],
                    "from_prediction_style": earlier["dominant_prediction_style"],
                    "to_prediction_style": later["dominant_prediction_style"],
                    "increased_themes": increased_themes,
                    "decreased_themes": decreased_themes,
                    "explanation": explain_shift(earlier, later, theme_cols),
                }
            )
    long_shifts = pd.DataFrame(long_rows).sort_values("long_shift_score", ascending=False)
    long_shifts.to_csv(OUTPUT_DIR / "long_term_tone_shifts.csv", index=False)

    lines = [
        "# Biggest Opinion / Tone Changes Between Consecutive Memos",
        "",
        "This report ranks adjacent memos by change in stance, prediction posture, and theme mix.",
        "",
        "## Biggest Adjacent Shifts",
        "",
    ]
    for row in shifts.head(20).itertuples():
        lines.extend(
            [
                f"## {row.from_title} -> {row.to_title}",
                f"- Dates: {row.from_date} to {row.to_date} ({row.days_between} days)",
                f"- Tone shift score: {row.tone_shift_score}",
                f"- Sentiment delta: {row.sentiment_delta:+.4f}",
                f"- Stance: {row.from_stance} -> {row.to_stance}",
                f"- Prediction posture: {row.from_prediction_style} -> {row.to_prediction_style}",
                f"- Themes up: {row.increased_themes}",
                f"- Themes down: {row.decreased_themes}",
                f"- Explanation: {row.explanation}",
                "",
            ]
        )
    recent = shifts[pd.to_datetime(shifts["to_date"]) >= pd.Timestamp("2023-01-01")].head(12)
    lines.extend(["## Recent Examples", ""])
    for row in recent.itertuples():
        lines.extend(
            [
                f"### {row.from_title} -> {row.to_title}",
                f"- Dates: {row.from_date} to {row.to_date}",
                f"- Shift score: {row.tone_shift_score}",
                f"- Explanation: {row.explanation}",
                "",
            ]
        )
    recent_long = long_shifts[pd.to_datetime(long_shifts["to_date"]) >= pd.Timestamp("2024-01-01")].head(15)
    lines.extend(["## Long-Horizon Reversals Or Oppositions", "", "These pairs are at least five years apart and rank highly for difference in tone, posture, and theme mix.", ""])
    for row in recent_long.itertuples():
        lines.extend(
            [
                f"### {row.from_title} ({row.from_date}) -> {row.to_title} ({row.to_date})",
                f"- Years apart: {row.years_between}",
                f"- Long-shift score: {row.long_shift_score}",
                f"- Sentiment delta: {row.sentiment_delta:+.4f}",
                f"- Stance: {row.from_stance} -> {row.to_stance}",
                f"- Prediction posture: {row.from_prediction_style} -> {row.to_prediction_style}",
                f"- Themes up: {row.increased_themes}",
                f"- Themes down: {row.decreased_themes}",
                f"- Explanation: {row.explanation}",
                "",
            ]
        )
    (OUTPUT_DIR / "tone_shift_report.md").write_text("\n".join(lines), encoding="utf-8")
    return shifts, long_shifts


def build_tone_shift_visual(shifts: pd.DataFrame) -> None:
    shifts = shifts.copy()
    shifts["to_date"] = pd.to_datetime(shifts["to_date"])
    fig = px.scatter(
        shifts,
        x="to_date",
        y="tone_shift_score",
        size=shifts["sentiment_delta"].abs() + 0.02,
        color="sentiment_delta",
        hover_data=["from_title", "to_title", "from_stance", "to_stance", "from_prediction_style", "to_prediction_style"],
        color_continuous_scale="RdBu",
        title="Tone Shift Intensity Across Consecutive Memos",
        labels={"to_date": "Later memo date", "tone_shift_score": "Shift score"},
    )
    save_figure(fig, "tone_shift_scatter.html")


def write_index() -> None:
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Howard Marks Memo Visualizations</title>
  <style>
    body { font-family: Georgia, serif; margin: 32px; background: #f7f4ed; color: #1f1f1f; }
    h1 { margin-bottom: 8px; }
    ul { line-height: 1.8; }
    a { color: #8b3a2b; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <h1>Howard Marks Memo Visualizations</h1>
  <p>Interactive outputs generated from the memo corpus analysis.</p>
  <ul>
    <li><a href="memo_network.html">Interactive memo network</a></li>
    <li><a href="theme_trends.html">Theme trends over time</a></li>
    <li><a href="theme_change_bars.html">Biggest thematic increases/decreases</a></li>
    <li><a href="sentiment_vs_sp500.html">Bullish/bearish tone vs. S&amp;P 500</a></li>
    <li><a href="tone_shift_scatter.html">Tone-shift intensity across time</a></li>
  </ul>
</body>
</html>
"""
    (VIZ_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    memo_df = prepare_memo_frame(data)
    build_network_graph(memo_df, data["references"])
    build_theme_visuals(data["themes"], data["change"])
    build_sentiment_vs_sp500(memo_df)
    shifts, _ = build_tone_shift_report(memo_df, data["themes"])
    build_tone_shift_visual(shifts)
    write_index()
    print(f"Wrote visualizations to {VIZ_DIR}")


if __name__ == "__main__":
    main()
