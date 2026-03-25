# Memo Analysis

This directory contains a first-pass analysis pipeline over the Howard Marks memo corpus in the repository.

## What it produces

Running `python analysis/run_memo_analysis.py` writes reviewable outputs to `analysis/output/`, including:

- `memos.csv`: normalized memo metadata
- `themes.csv`: keyword-based topic evolution table
- `references.csv`: inferred memo-to-memo references
- `stance.csv`: tone / stance scoring
- `prediction_style.csv`: forecasting vs. preparation posture
- `clusters.csv`: unsupervised memo clusters
- `language_fingerprint.csv`: repeated phrases
- `core_beliefs.csv`: recurring principle summary
- `important_memos.csv`: ranked reading list
- `meeting_brief.md`: compact prep brief
- `meeting_questions.md`: candidate questions
- `summary.md`: top-level narrative summary
- `analysis_bundle.json`: bundled structured output for later LLM work

## Notes

- The pipeline uses the local markdown files as the corpus.
- The memo corpus lives under `memos/`.
- If a memo date is missing from the local frontmatter or URL, the script attempts to enrich it from the memo URL on Oaktree's site.
- The analysis is intended as a strong first pass, not a final scholarly classification. The outputs are meant to be refined as you review what looks most useful for your meeting.
