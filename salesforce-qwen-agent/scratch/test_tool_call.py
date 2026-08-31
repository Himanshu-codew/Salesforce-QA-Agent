import asyncio, os, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from llm.qwen import QwenLLM

async def test():
    llm = QwenLLM(
        api_key=os.getenv('QWEN_API_KEY','ollama'),
        base_url=os.getenv('QWEN_BASE_URL','http://localhost:11434/v1'),
        model=os.getenv('QWEN_MODEL','qwen2.5:latest'),
    )
    
    tools = [{
        'type': 'function',
        'function': {
            'name': 'soqlQuery',
            'description': 'Run a SOQL query on Salesforce',
            'parameters': {
                'type': 'object',
                'properties': {'q': {'type': 'string', 'description': 'SOQL query string'}},
                'required': ['q']
            }
        }
    }]
    
    messages = [
        {'role': 'system', 'content': 'You are a Salesforce assistant. Use soqlQuery tool to fetch data. Never reply without calling a tool first.'},
        {'role': 'user', 'content': 'Show me all Accounts in Salesforce'}
    ]
    
    print(f"Model: {os.getenv('QWEN_MODEL')}")
    print(f"URL:   {os.getenv('QWEN_BASE_URL')}")
    print("Testing tool calling...")
    
    result = await llm.chat_with_tools(messages=messages, tools=tools)
    print(f"\nContent:     {repr(result.get('content','')[:200])}")
    print(f"Tool calls:  {result.get('tool_calls', [])}")
    print(f"Finish:      {result.get('finish_reason')}")
    
    if result.get('tool_calls'):
        print('\n✅ SUCCESS: Model called MCP tools correctly!')
    else:
        print('\n❌ ISSUE: Model did NOT call any tool — needs fix')

asyncio.run(test())
