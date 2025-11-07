# Router Prompt

Decide the **intent** of the user message. Choose one:
- `search_only` — retrieve passages and answer.
- `analytics` — run tabular/statistical analysis (CSV/XLSX/JSON).
- `web` — use optional web search (only if policy enables).
- `clarify` — ask a brief clarification when the query is underspecified.

Return JSON:
{"intent":"search_only","reason":"..."}
