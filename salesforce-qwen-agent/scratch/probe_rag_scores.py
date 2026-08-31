"""Probe: get real scores from new RAG for benchmark queries + unrelated + short."""
import sys, os, logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.rag import ToolRAGRetriever

retriever = ToolRAGRetriever(default_top_k=6, min_confidence=0.0)  # very low to see all scores

queries = [
    "What is my Salesforce user information?",
    "Show me my recent Accounts.",
    "Find Opportunities where Amount is greater than 50000.",
    "What is the weather in London?",
    "Help me with something.",
    "Show me all Accounts AND tell me how many Leads I have.",
    "Find ABC Technologies, show its Opportunities, AND count its Contacts.",
    "Show me Contacts at John Doe, update the phone on the first one to 555-1111, and then delete the oldest Lead.",
    "Find the newest Lead and delete it, then create a new Account for whatever company it was from.",
    "Show me Tasks and Events together, sorted by date, for the next 7 days.",
    "Show me Opportunities where the Account's Industry is Technology OR the Amount is over $50,000.",
    "Show me accounts with SOQL query",
    "hi",
]

for q in queries:
    print(f"\n{'='*70}")
    print(f"QUERY: {q}")
    print("="*70)
    tools = retriever.get_relevant_tools(q, top_k=6)
    tool_names = [t["function"]["name"] for t in tools]
    print(f"SELECTED TOOLS: {tool_names}")
