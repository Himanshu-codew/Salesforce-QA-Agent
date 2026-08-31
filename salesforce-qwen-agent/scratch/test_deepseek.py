import asyncio
import json
from llm.qwen import QwenLLM
import os

async def main():
    llm = QwenLLM(
        api_key=os.getenv("QWEN_API_KEY", "ollama"),
        base_url=os.getenv("QWEN_BASE_URL", "https://fiscally-coherent-gratified.ngrok-free.dev/v1"),
        model=os.getenv("QWEN_MODEL", "deepseek-coder-v2:16b")
    )
    
    print("Testing Planner...")
    res = await llm.chat(
        messages=[{"role": "user", "content": "Create an execution plan for: Show me all Accounts. Output STRICTLY a JSON array of tasks."}],
        temperature=0.0
    )
    print("Planner Output:")
    print(res)

    print("\nTesting DataAgent...")
    tools = [{
        "type": "function",
        "function": {
            "name": "soqlQuery",
            "description": "Execute a SOQL query",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}}
            }
        }
    }]
    res2 = await llm.chat_with_tools(
        messages=[{"role": "user", "content": "Fetch all accounts using SOQL."}],
        tools=tools,
        temperature=0.0
    )
    print("DataAgent Tool Calls:")
    print(json.dumps(res2["tool_calls"], indent=2))
    print("DataAgent Raw Content:")
    print(res2["content"])

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())
