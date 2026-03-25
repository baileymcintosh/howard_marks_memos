from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "analysis" / "output"
REQUEST_TIMEOUT = 20
USER_AGENT = "oaktree-memo-analysis/1.0"


THEME_KEYWORDS = {
    "risk": ["risk", "risky", "safety", "loss", "losses", "downside", "preservation of capital", "margin of safety", "default"],
    "cycles": ["cycle", "cycles", "cyclical", "pendulum", "swing", "swings", "upswing", "downswing", "boom", "bust"],
    "psychology": ["psychology", "emotion", "greed", "fear", "euphoria", "pessimism", "optimism", "investor psychology", "sentiment"],
    "valuation": ["value", "valuation", "price", "priced", "cheap", "expensive", "intrinsic value", "multiple", "return"],
    "credit": ["credit", "debt", "bond", "bonds", "loan", "loans", "yield", "spread", "default", "distressed"],
    "macro": ["economy", "economic", "macro", "gdp", "recession", "inflation", "deflation", "growth", "fiscal", "monetary"],
    "rates": ["interest rate", "interest rates", "rates", "fed", "federal reserve", "yield curve", "cost of capital"],
    "uncertainty": ["uncertain", "uncertainty", "unknown", "unknowable", "nobody knows", "surprise", "unpredictable"],
    "prediction": ["predict", "prediction", "forecast", "outlook", "expect", "expected", "probability", "prepare"],
    "politics": ["election", "politics", "political", "government", "regulation", "policy", "congress", "president"],
    "technology_ai": ["technology", "internet", "dot-com", "bubble.com", "ai", "artificial intelligence", "silicon valley", "tech"],
}


STANCE_KEYWORDS = {
    "cautious": ["caution", "cautious", "prudence", "prudent", "careful", "defensive", "skeptical", "skepticism"],
    "opportunistic": ["opportunity", "opportunities", "attractive", "aggressive", "buy", "bargain", "cheap", "favorable"],
    "alarmed": ["danger", "warning", "alarm", "bubble", "excess", "frothy", "panic", "crisis", "grave"],
    "constructive": ["improve", "benefit", "good", "better", "healthy", "optimistic", "resilient", "encouraging"],
    "humble": ["i don't know", "nobody knows", "uncertain", "uncertainty", "unknowable", "humility", "modest"],
}


PREDICTION_STYLE_KEYWORDS = {
    "forecasting": ["forecast", "predict", "prediction", "outlook", "will", "likely"],
    "warning": ["warning", "bubble", "danger", "risk", "frothy", "excess", "too high"],
    "preparation": ["prepare", "prepared", "positioning", "defensive", "offense", "prudence"],
    "anti_prediction": ["i don't know", "nobody knows", "can't predict", "cannot predict", "unknowable"],
}


CORE_BELIEF_PATTERNS = {
    "risk_control": ["risk", "avoid losses", "preservation of capital", "defensive", "margin of safety"],
    "market_cycles": ["cycle", "pendulum", "upswing", "downswing", "boom", "bust"],
    "investor_psychology": ["greed", "fear", "psychology", "euphoria", "pessimism", "optimism"],
    "humility_uncertainty": ["nobody knows", "uncertainty", "i don't know", "unknowable", "surprise"],
    "price_vs_value": ["value", "price", "intrinsic", "cheap", "expensive", "valuation"],
    "preparation_over_prediction": ["prepare", "predict", "forecast", "positioning", "probability"],
}


PERIODS = [("1990s", 1990, 1999), ("2000s", 2000, 2009), ("2010s", 2010, 2019), ("2020s", 2020, 2029)]


STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "howard", "marks", "oaktree", "memo", "memos", "investor", "investors", "investment", "investments", "market", "markets"
}

AMBIGUOUS_REFERENCE_TITLES = {
    "on the other hand",
    "everyone knows",
    "not enough",
    "you bet",
    "yet again",
    "now what",
    "who knew",
    "ditto",
    "mysterious",
    "economic reality",
    "latest update",
    "latest thinking",
    "this time it s different",
    "go figure",
}


@dataclass
class Memo:
    memo_id: str
    directory: str
    title: str
    url: str
    downloaded: str | None
    text: str
    word_count: int
    date: str | None
    date_source: str


def fix_mojibake(text: str) -> str:
    bad_markers = text.count("â") + text.count("ï") + text.count("Ã")
    if bad_markers < 5:
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
        if repaired.count("â") + repaired.count("ï") + repaired.count("Ã") < bad_markers:
            return repaired
    except Exception:
        pass
    return text


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    frontmatter, body = parts[1], parts[2]
    meta: dict[str, str] = {}
    for line in frontmatter.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"')
    return meta, body.strip()


def normalize_title(text: str) -> str:
    text = fix_mojibake(text).lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str) -> str:
    return normalize_title(text).replace(" ", "_")


def extract_date_from_url(url: str) -> str | None:
    match = re.search(r"/(\d{4}-\d{2}-\d{2})[-.]", url)
    return match.group(1) if match else None


def fetch_date_from_url(url: str, title: str, session: requests.Session) -> tuple[str | None, str]:
    if not url or "oaktreecapital.com" not in url:
        return None, "missing"
    candidates = [url]
    title_slug = slugify(title).replace("_", "-").replace("s-the", "s-the")
    insight_candidate = f"https://www.oaktreecapital.com/insights/memo/{title_slug}"
    if insight_candidate not in candidates:
        candidates.append(insight_candidate)
    patterns = [
        r"<!-- Date -->\s*<p>([A-Z][a-z]{2,8} \d{1,2}, \d{4})</p>",
        r"<p>([A-Z][a-z]{2,8} \d{1,2}, \d{4})</p>\s*<br />\s*<p>P\.s\.:",
        r"([A-Z][a-z]{2,8} \d{1,2}, \d{4})",
    ]
    for candidate in candidates:
        try:
            response = session.get(candidate, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
        except Exception:
            continue
        html = response.text
        for pattern in patterns:
            match = re.search(pattern, html, re.DOTALL)
            if not match:
                continue
            try:
                dt = pd.to_datetime(match.group(1))
                source = "fetched_insight" if candidate != url else "fetched"
                return dt.strftime("%Y-%m-%d"), source
            except Exception:
                continue
    return None, "fetch_no_date"


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [piece.strip() for piece in pieces if len(piece.strip()) > 40]


def count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    lowered = text.lower()
    hits = 0
    for keyword in keywords:
        if " " in keyword:
            hits += lowered.count(keyword.lower())
        else:
            hits += len(re.findall(rf"\b{re.escape(keyword.lower())}\b", lowered))
    return hits


def build_reference_patterns(title: str) -> tuple[list[re.Pattern[str]], re.Pattern[str], bool]:
    clean_title = fix_mojibake(title).strip()
    escaped = re.escape(clean_title)
    cue_patterns = [
        re.compile(rf'(?i)\b(?:memo|memos)\s+(?:called|entitled|titled)?\s*[“"\'`]?\s*{escaped}\s*[”"\'`]?\b'),
        re.compile(rf'(?i)\b(?:in|from)\s+(?:my\s+)?memo\s*[“"\'`]?\s*{escaped}\s*[”"\'`]?\b'),
        re.compile(rf'(?i)\b(?:called|entitled|titled)\s*[“"\'`]?\s*{escaped}\s*[”"\'`]?\b'),
        re.compile(rf'[“"\'`]\s*{escaped}\s*[”"\'`]'),
    ]
    bare_pattern = re.compile(rf'(?<![A-Za-z]){escaped}(?![A-Za-z])')
    ambiguous = normalize_title(clean_title) in AMBIGUOUS_REFERENCE_TITLES
    return cue_patterns, bare_pattern, ambiguous


def period_label(date_value: str | None) -> str:
    if not date_value:
        return "undated"
    year = int(date_value[:4])
    for label, start, end in PERIODS:
        if start <= year <= end:
            return label
    return "other"


def load_memos() -> list[Memo]:
    session = requests.Session()
    memos: list[Memo] = []
    for article_path in sorted((ROOT / "memos").glob("2026-03_*/article.md")):
        raw = fix_mojibake(article_path.read_text(encoding="utf-8", errors="ignore"))
        meta, body = parse_frontmatter(raw)
        title = meta.get("title", article_path.parent.name)
        url = meta.get("url", "")
        downloaded = meta.get("downloaded")
        date = extract_date_from_url(url)
        date_source = "url" if date else "none"
        if not date:
            date, date_source = fetch_date_from_url(url, title, session)
        body = re.sub(r"\s+", " ", fix_mojibake(body)).strip()
        memos.append(
            Memo(
                memo_id=slugify(title),
                directory=article_path.parent.name,
                title=title,
                url=url,
                downloaded=downloaded,
                text=body,
                word_count=len(re.findall(r"\b[\w'-]+\b", body)),
                date=date,
                date_source=date_source,
            )
        )
    return memos


def build_metadata_df(memos: list[Memo]) -> pd.DataFrame:
    rows = []
    for memo in memos:
        rows.append(
            {
                "memo_id": memo.memo_id,
                "directory": memo.directory,
                "title": memo.title,
                "date": memo.date,
                "year": int(memo.date[:4]) if memo.date else np.nan,
                "period": period_label(memo.date),
                "date_source": memo.date_source,
                "url": memo.url,
                "downloaded": memo.downloaded,
                "word_count": memo.word_count,
                "text": memo.text,
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "title"], na_position="last").reset_index(drop=True)


def build_theme_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        memo_text = row["text"]
        for theme, keywords in THEME_KEYWORDS.items():
            hits = count_keyword_hits(memo_text, keywords)
            rows.append(
                {
                    "memo_id": row["memo_id"],
                    "title": row["title"],
                    "date": row["date"],
                    "year": row["year"],
                    "period": row["period"],
                    "theme": theme,
                    "hits": hits,
                    "hits_per_1k_words": (hits / max(row["word_count"], 1)) * 1000.0,
                }
            )
    return pd.DataFrame(rows)


def build_stance_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        memo_text = row["text"]
        total = 0
        scores = {}
        for stance, keywords in STANCE_KEYWORDS.items():
            score = count_keyword_hits(memo_text, keywords)
            scores[stance] = score
            total += score
        dominant = max(scores, key=scores.get) if total else "neutral"
        entry = {
            "memo_id": row["memo_id"],
            "title": row["title"],
            "date": row["date"],
            "period": row["period"],
            "dominant_stance": dominant,
        }
        for stance, score in scores.items():
            entry[f"{stance}_score"] = score
        rows.append(entry)
    return pd.DataFrame(rows)


def build_prediction_style_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        memo_text = row["text"].lower()
        entry = {
            "memo_id": row["memo_id"],
            "title": row["title"],
            "date": row["date"],
            "period": row["period"],
        }
        for style, keywords in PREDICTION_STYLE_KEYWORDS.items():
            entry[f"{style}_score"] = count_keyword_hits(memo_text, keywords)
        style_columns = [f"{style}_score" for style in PREDICTION_STYLE_KEYWORDS]
        entry["dominant_prediction_style"] = max(style_columns, key=lambda key: entry[key]).replace("_score", "")
        rows.append(entry)
    return pd.DataFrame(rows)


def find_references(df: pd.DataFrame) -> pd.DataFrame:
    title_map = {
        row["memo_id"]: {
            "title": row["title"],
            "normalized": normalize_title(row["title"]),
            "date": row["date"] or "",
            "reference_meta": build_reference_patterns(row["title"]),
        }
        for _, row in df.iterrows()
    }
    rows = []
    ordered_titles = sorted(title_map.items(), key=lambda item: len(item[1]["normalized"]), reverse=True)
    for _, source_row in df.iterrows():
        source_date = pd.to_datetime(source_row["date"], errors="coerce")
        source_text = fix_mojibake(source_row["text"])
        for target_id, info in ordered_titles:
            if target_id == source_row["memo_id"]:
                continue
            normalized_title = info["normalized"]
            target_date = pd.to_datetime(info["date"], errors="coerce")
            if len(normalized_title.split()) < 2:
                continue
            if pd.notna(source_date) and pd.notna(target_date) and source_date < target_date:
                continue
            cue_patterns, bare_pattern, ambiguous = info["reference_meta"]
            cue_hit = any(pattern.search(source_text) for pattern in cue_patterns)
            bare_hit = bool(bare_pattern.search(source_text))
            if cue_hit or (bare_hit and not ambiguous):
                rows.append(
                    {
                        "source_id": source_row["memo_id"],
                        "source_title": source_row["title"],
                        "source_date": source_row["date"],
                        "target_id": target_id,
                        "target_title": info["title"],
                        "target_date": info["date"],
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["source_id", "source_title", "source_date", "target_id", "target_title", "target_date"])
    return pd.DataFrame(rows).drop_duplicates().sort_values(["source_date", "source_title", "target_date", "target_title"])


def build_tfidf(df: pd.DataFrame) -> tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS), ngram_range=(1, 2), min_df=2, max_df=0.75)
    matrix = vectorizer.fit_transform(df["text"])
    return vectorizer, matrix


def cluster_memos(df: pd.DataFrame, matrix: np.ndarray, vectorizer: TfidfVectorizer) -> pd.DataFrame:
    model = KMeans(n_clusters=6, random_state=42, n_init=20)
    labels = model.fit_predict(matrix)
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_labels = {}
    for idx, center in enumerate(model.cluster_centers_):
        top_terms = feature_names[np.argsort(center)[-5:]][::-1]
        cluster_labels[idx] = ", ".join(top_terms[:3])
    output = df[["memo_id", "title", "date", "period", "word_count"]].copy()
    output["cluster_id"] = labels
    output["cluster_label"] = output["cluster_id"].map(cluster_labels)
    return output.sort_values(["cluster_id", "date", "title"])


def build_language_fingerprint(df: pd.DataFrame, vectorizer: TfidfVectorizer, matrix: np.ndarray) -> pd.DataFrame:
    feature_names = np.array(vectorizer.get_feature_names_out())
    global_scores = np.asarray(matrix.sum(axis=0)).ravel()
    mask = np.array([(" " in term) for term in feature_names])
    ranked = sorted(zip(feature_names[mask], global_scores[mask]), key=lambda item: item[1], reverse=True)
    return pd.DataFrame([{"phrase": phrase, "score": round(float(score), 3)} for phrase, score in ranked[:75]])


def build_core_beliefs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for belief, keywords in CORE_BELIEF_PATTERNS.items():
        hit_rows = []
        for _, row in df.iterrows():
            hits = count_keyword_hits(row["text"], keywords)
            if hits:
                hit_rows.append((hits, row))
        hit_rows.sort(key=lambda item: item[0], reverse=True)
        example_sentence = ""
        supporting_titles = []
        for _, row in hit_rows[:5]:
            supporting_titles.append(row["title"])
            if not example_sentence:
                for sentence in split_sentences(row["text"]):
                    if any(keyword.lower() in sentence.lower() for keyword in keywords):
                        example_sentence = sentence
                        break
        rows.append(
            {
                "belief": belief,
                "supporting_memos": supporting_titles[:5],
                "example_sentence": example_sentence,
                "memo_count": len(hit_rows),
            }
        )
    return pd.DataFrame(rows).sort_values("memo_count", ascending=False)


def build_change_report(theme_df: pd.DataFrame) -> pd.DataFrame:
    dated = theme_df.dropna(subset=["year"]).copy()
    grouped = dated.groupby(["period", "theme"], as_index=False)["hits_per_1k_words"].mean()
    pivot = grouped.pivot(index="theme", columns="period", values="hits_per_1k_words").fillna(0.0)
    rows = []
    for theme, values in pivot.iterrows():
        early = values.get("1990s", 0.0)
        late = values.get("2020s", 0.0)
        rows.append(
            {
                "theme": theme,
                "avg_1990s": round(float(early), 3),
                "avg_2020s": round(float(late), 3),
                "delta_1990s_to_2020s": round(float(late - early), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_1990s_to_2020s", ascending=False)


def rank_memos(df: pd.DataFrame, references: pd.DataFrame, theme_df: pd.DataFrame) -> pd.DataFrame:
    incoming = references.groupby("target_id").size().to_dict()
    outgoing = references.groupby("source_id").size().to_dict()
    breadth = theme_df.groupby("memo_id")["theme"].apply(lambda s: s.nunique()).to_dict()
    theme_hits = theme_df.groupby("memo_id")["hits"].sum().to_dict()
    rows = []
    for _, row in df.iterrows():
        score = (
            4.0 * incoming.get(row["memo_id"], 0)
            + 1.5 * outgoing.get(row["memo_id"], 0)
            + 0.003 * row["word_count"]
            + 0.08 * theme_hits.get(row["memo_id"], 0)
            + 0.5 * breadth.get(row["memo_id"], 0)
        )
        rows.append(
            {
                "memo_id": row["memo_id"],
                "title": row["title"],
                "date": row["date"],
                "incoming_refs": incoming.get(row["memo_id"], 0),
                "outgoing_refs": outgoing.get(row["memo_id"], 0),
                "word_count": row["word_count"],
                "importance_score": round(score, 3),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_score", ascending=False)


def build_similarity_table(df: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    sim = cosine_similarity(matrix)
    rows = []
    for idx, row in df.iterrows():
        order = np.argsort(sim[idx])[::-1]
        related = []
        for other_idx in order:
            if other_idx == idx:
                continue
            related.append(df.iloc[other_idx]["title"])
            if len(related) == 3:
                break
        rows.append({"memo_id": row["memo_id"], "title": row["title"], "date": row["date"], "nearest_neighbors": " | ".join(related)})
    return pd.DataFrame(rows)


def build_meeting_questions(core_beliefs: pd.DataFrame, change_df: pd.DataFrame, ranked_df: pd.DataFrame, references: pd.DataFrame) -> list[str]:
    top_beliefs = core_beliefs.head(3)["belief"].tolist()
    top_shift = change_df.iloc[0]["theme"] if not change_df.empty else "markets"
    top_memos = ranked_df.head(3)["title"].tolist()
    top_ref = ""
    if not references.empty:
        top_ref = references.groupby("target_title").size().sort_values(ascending=False).index[0]
    return [
        f"Which of your recurring ideas around {top_beliefs[0].replace('_', ' ')} do you think readers still underappreciate?",
        f"Looking back across the memos, where do you think your thinking changed most on {top_shift}?",
        f"If someone only read {top_memos[0]} and {top_memos[1]}, what part of your framework would they still miss?",
        f"Why do you think {top_ref or top_memos[0]} became such a durable reference point in your later memos?",
        "When you write, are you aiming more to improve investor behavior or to improve portfolio decisions directly?",
        "How do you decide when a market environment deserves a memo versus a brief update or no comment at all?",
        "Which recent memo best reflects your current thinking, and which older memo still feels fully intact to you?",
        "Where do you think readers most often confuse caution with pessimism in your writing?",
    ]


def write_markdown_summary(
    df: pd.DataFrame,
    theme_df: pd.DataFrame,
    core_beliefs: pd.DataFrame,
    references: pd.DataFrame,
    stance_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    fingerprint_df: pd.DataFrame,
    change_df: pd.DataFrame,
) -> None:
    dated = df.dropna(subset=["date"]).copy()
    top_theme_period = (
        theme_df.groupby(["period", "theme"], as_index=False)["hits_per_1k_words"].mean()
        .sort_values(["period", "hits_per_1k_words"], ascending=[True, False])
    )
    top_refs = references.groupby("target_title", as_index=False).size().sort_values("size", ascending=False).head(10)
    dominant_stances = stance_df.groupby("dominant_stance", as_index=False).size().sort_values("size", ascending=False)
    dominant_prediction = prediction_df.groupby("dominant_prediction_style", as_index=False).size().sort_values("size", ascending=False)
    lines = [
        "# Howard Marks Memo Analysis",
        "",
        "## Corpus",
        f"- Memos loaded: {len(df)}",
        f"- Dated memos: {int(df['date'].notna().sum())}",
        f"- Undated memos after enrichment: {int(df['date'].isna().sum())}",
        f"- Earliest dated memo: {dated.iloc[0]['title']} ({dated.iloc[0]['date']})" if not dated.empty else "- Earliest dated memo: n/a",
        f"- Latest dated memo: {dated.iloc[-1]['title']} ({dated.iloc[-1]['date']})" if not dated.empty else "- Latest dated memo: n/a",
        "",
        "## 1. Topic Evolution",
        "Top themes by period:",
    ]
    for period in ["1990s", "2000s", "2010s", "2020s", "undated"]:
        subset = top_theme_period[top_theme_period["period"] == period].head(4)
        if subset.empty:
            continue
        lines.append("- " + period + ": " + ", ".join(f"{row.theme} ({row.hits_per_1k_words:.2f})" for row in subset.itertuples()))
    lines.extend(["", "## 2. Core Beliefs"])
    for row in core_beliefs.itertuples():
        lines.append(f"- {row.belief}: appears in {row.memo_count} memos; example: {row.example_sentence[:240].strip()}")
    lines.extend(["", "## 3. Change Detection", "Largest 1990s to 2020s theme shifts:"])
    for row in change_df.head(6).itertuples():
        lines.append(f"- {row.theme}: {row.avg_1990s:.2f} to {row.avg_2020s:.2f} hits per 1k words (delta {row.delta_1990s_to_2020s:+.2f})")
    lines.extend(["", "## 4. Memo Network", "Most referenced memos within the corpus:"])
    for row in top_refs.itertuples():
        lines.append(f"- {row.target_title}: cited {row.size} times")
    lines.extend(["", "## 5. Stance Over Time", "Dominant stance distribution:"])
    for row in dominant_stances.itertuples():
        lines.append(f"- {row.dominant_stance}: {row.size}")
    lines.extend(["", "## 6. Prediction Style", "Dominant prediction posture:"])
    for row in dominant_prediction.itertuples():
        lines.append(f"- {row.dominant_prediction_style}: {row.size}")
    lines.extend(["", "## 7. Language Fingerprint", "Representative repeated phrases:"])
    for row in fingerprint_df.head(12).itertuples():
        lines.append(f"- {row.phrase}")
    lines.extend(["", "## 8. Memo Clusters"])
    for cluster_id, subset in clusters_df.groupby("cluster_id"):
        lines.append(f"- Cluster {cluster_id} [{subset['cluster_label'].iloc[0]}]: " + ", ".join(subset['title'].head(4)))
    lines.extend(["", "## 9. Most Important Memos"])
    for row in ranked_df.head(10).itertuples():
        lines.append(f"- {row.title} ({row.date or 'undated'}): score {row.importance_score}, incoming refs {row.incoming_refs}")
    lines.extend(["", "## 10. Candidate Meeting Questions"])
    for question in build_meeting_questions(core_beliefs, change_df, ranked_df, references):
        lines.append(f"- {question}")
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_questions_file(questions: list[str]) -> None:
    lines = ["# Candidate Questions", ""] + [f"{idx}. {question}" for idx, question in enumerate(questions, start=1)]
    (OUTPUT_DIR / "meeting_questions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_briefing_file(df: pd.DataFrame, ranked_df: pd.DataFrame, core_beliefs: pd.DataFrame, change_df: pd.DataFrame, stance_df: pd.DataFrame) -> None:
    latest = df.dropna(subset=["date"]).sort_values("date").tail(5)
    stance_counts = stance_df.groupby("dominant_stance").size().sort_values(ascending=False)
    lines = ["# Meeting Brief", "", "## High-Signal Read List"]
    for row in ranked_df.head(8).itertuples():
        lines.append(f"- {row.title} ({row.date or 'undated'})")
    lines.extend(["", "## Durable Principles"])
    for row in core_beliefs.head(5).itertuples():
        lines.append(f"- {row.belief.replace('_', ' ')}")
    lines.extend(["", "## Recent Context"])
    for row in latest.itertuples():
        lines.append(f"- {row.title} ({row.date})")
    lines.extend(["", "## Major Long-Run Shifts"])
    for row in change_df.head(5).itertuples():
        lines.append(f"- {row.theme}: delta {row.delta_1990s_to_2020s:+.2f} hits per 1k words")
    lines.extend(["", "## Tone Profile"])
    for stance, count in stance_counts.items():
        lines.append(f"- {stance}: {count}")
    (OUTPUT_DIR / "meeting_brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_llm_seed_file(core_beliefs: pd.DataFrame, ranked_df: pd.DataFrame, references: pd.DataFrame) -> None:
    lines = [
        "# Howard Marks Memo Assistant Seed Notes",
        "",
        "## Intended Use",
        "- Retrieval and synthesis over the memo corpus with citations.",
        "- Avoid imitating personality unless the prompt explicitly asks for tone analysis.",
        "",
        "## Canonical Memos",
    ]
    for title in ranked_df.head(12)["title"].tolist():
        lines.append(f"- {title}")
    lines.extend(["", "## Core Concepts"])
    for item in core_beliefs.head(6)["belief"].tolist():
        lines.append(f"- {item.replace('_', ' ')}")
    if not references.empty:
        lines.extend(["", "## Useful Retrieval Hint", "- Prioritize memos that are frequently cited by later memos before surfacing adjacent or recent ones."])
    (OUTPUT_DIR / "llm_seed_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json_bundle(df: pd.DataFrame, theme_df: pd.DataFrame, references: pd.DataFrame, stance_df: pd.DataFrame, prediction_df: pd.DataFrame, clusters_df: pd.DataFrame, ranked_df: pd.DataFrame) -> None:
    bundle = {
        "memos": df.drop(columns=["text"]).to_dict(orient="records"),
        "top_memos": ranked_df.head(20).to_dict(orient="records"),
        "themes": theme_df.to_dict(orient="records"),
        "references": references.to_dict(orient="records"),
        "stance": stance_df.to_dict(orient="records"),
        "prediction_style": prediction_df.to_dict(orient="records"),
        "clusters": clusters_df.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "analysis_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    memos = load_memos()
    df = build_metadata_df(memos)
    theme_df = build_theme_df(df)
    stance_df = build_stance_df(df)
    prediction_df = build_prediction_style_df(df)
    references = find_references(df)
    vectorizer, matrix = build_tfidf(df)
    clusters_df = cluster_memos(df, matrix, vectorizer)
    fingerprint_df = build_language_fingerprint(df, vectorizer, matrix)
    core_beliefs = build_core_beliefs(df)
    change_df = build_change_report(theme_df)
    ranked_df = rank_memos(df, references, theme_df)
    similarity_df = build_similarity_table(df, matrix)

    df.drop(columns=["text"]).to_csv(OUTPUT_DIR / "memos.csv", index=False)
    theme_df.to_csv(OUTPUT_DIR / "themes.csv", index=False)
    stance_df.to_csv(OUTPUT_DIR / "stance.csv", index=False)
    prediction_df.to_csv(OUTPUT_DIR / "prediction_style.csv", index=False)
    references.to_csv(OUTPUT_DIR / "references.csv", index=False)
    clusters_df.to_csv(OUTPUT_DIR / "clusters.csv", index=False)
    fingerprint_df.to_csv(OUTPUT_DIR / "language_fingerprint.csv", index=False)
    core_beliefs.to_csv(OUTPUT_DIR / "core_beliefs.csv", index=False)
    change_df.to_csv(OUTPUT_DIR / "change_report.csv", index=False)
    ranked_df.to_csv(OUTPUT_DIR / "important_memos.csv", index=False)
    similarity_df.to_csv(OUTPUT_DIR / "similarity.csv", index=False)

    write_markdown_summary(df, theme_df, core_beliefs, references, stance_df, prediction_df, clusters_df, ranked_df, fingerprint_df, change_df)
    questions = build_meeting_questions(core_beliefs, change_df, ranked_df, references)
    write_questions_file(questions)
    write_briefing_file(df, ranked_df, core_beliefs, change_df, stance_df)
    write_llm_seed_file(core_beliefs, ranked_df, references)
    write_json_bundle(df, theme_df, references, stance_df, prediction_df, clusters_df, ranked_df)
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
