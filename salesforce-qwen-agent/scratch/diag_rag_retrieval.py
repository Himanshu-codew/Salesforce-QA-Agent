"""
Temporary diagnostic: Is the RAG system doing semantic vector retrieval,
or is it keyword/rule-based?

Runs 5 different queries through ToolRAGRetriever and reports raw
TF-similarity scores, the rule overrides, and the final selected tools.
No network calls are made — pure offline retrieval analysis.
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure RAG is enabled for the test
os.environ["ENABLE_RAG_TOOLS"] = "true"

from agent.rag import (  # noqa: E402
    ToolRAGRetriever,
    tokenize,
    compute_vector,
    cosine_similarity,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

retriever = ToolRAGRetriever(default_top_k=3, min_confidence=0.12)

QUERIES = {
    "1. user-info": "Who am I? Show my own profile and email from my Salesforce account.",
    "2. recent-records": "Show me my recent Accounts I have viewed.",
    "3. soql-custom-filter": "List all Opportunities with Amount greater than 50000 ordered by CloseDate.",
    "4. no-salesforce-tool": "What is the weather in London today?",
    "5. ambiguous": "Can you help me out with something?",
}

print("\n" + "=" * 80)
print("DIAGNOSTIC: Full RAG retrieval pipeline trace")
print("=" * 80)

for label, query in QUERIES.items():
    print("\n" + "=" * 80)
    print(f"[CASE {label}] QUERY: {query!r}")
    print("=" * 80)

    # 1. Tokenize raw
    q_tokens = tokenize(query)
    print(f"[ANALYSIS] Tokens: {q_tokens}")

    # 2. Build query TF vector and score against every tool doc
    q_vec = compute_vector(q_tokens, retriever.vocab)
    print(f"[ANALYSIS] TF vector dimension: {len(q_vec)}  (embedding model: NONE)")

    scores = []
    for name, doc_vec in retriever.vectors.items():
        sim = cosine_similarity(q_vec, doc_vec)
        scores.append((sim, name))
    scores.sort(key=lambda x: x[0], reverse=True)

    print("[ANALYSIS] Raw cosine-TF similarity (all tools, sorted desc):")
    for sim, name in scores:
        marker = " <-- top-3 candidate" if scores.index((sim, name)) < 3 else ""
        print(f"    {name:28s} sim={sim:.4f}{marker}")

    # 3. Now call the actual public method (with the rule overrides)
    selected = retriever.get_relevant_tools(query, top_k=3)
    print(f"[ANALYSIS] FINAL SELECTED (public method, incl. rules): {[t['function']['name'] for t in selected]}")

    # 4. Determine which selected tools came from pure vector top-3 vs rule-added
    vector_top3 = {name for _, name in scores[:3]}
    selected_names = {t["function"]["name"] for t in selected}
    rule_only = selected_names - vector_top3
    vector_kept = selected_names & vector_top3
    print(f"[ANALYSIS] Pure-vector top-3 names: {sorted(vector_top3)}")
    print(f"[ANALYSIS] Selected that came from pure vector top-3: {sorted(vector_kept)}")
    print(f"[ANALYSIS] Selected that came ONLY from hardcoded intent rules: {sorted(rule_only)}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
