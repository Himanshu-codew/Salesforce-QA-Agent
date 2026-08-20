# Salesforce Qwen Agent 🤖⚡

An AI-powered Salesforce assistant that lets you interact with your Salesforce org through natural language. Powered by **Qwen3 Instruct** for intelligent reasoning and **Salesforce MCP Server** for secure data operations.

![Architecture](https://img.shields.io/badge/Architecture-Agent_+_MCP-blue)
![LLM](https://img.shields.io/badge/LLM-Qwen3_Instruct-purple)
![Platform](https://img.shields.io/badge/Platform-Salesforce-00A1E0)

---

## ✨ Features

- **Natural Language Interface** — Ask questions, create records, run queries in plain English
- **11 Salesforce Tools** — Full CRUD, SOQL/SOSL queries, schema exploration, and more
- **Real-time Chat UI** — Premium dark-themed interface with live tool execution visualization
- **Safety Guardrails** — Destructive operations require explicit user confirmation
- **Multi-step Reasoning** — Agent can chain multiple tool calls to answer complex questions
- **Conversation Memory** — Maintains context across messages within a session
- **REST API Fallback** — Gracefully falls back to direct Salesforce REST API if MCP is unavailable

---

## 🏗️ Architecture

```
User
 │
 ▼
Web UI / Chat (WebSocket)
 │
 ▼
FastAPI Backend (app.py)
 │
 ├── Qwen3 Instruct (LLM reasoning + tool selection)
 │
 └── MCP Client → Salesforce MCP Server
                   │
                   ├── SOQL Queries
                   ├── SOSL Search
                   ├── CRUD Operations
                   ├── Schema / Metadata
                   ├── Related Records
                   └── User Info
```

---

## 📁 Project Structure

```
salesforce-qwen-agent/
├── app.py                # FastAPI + WebSocket backend
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── llm/
│   ├── base.py           # Abstract LLM interface
│   └── qwen.py           # Qwen3 Instruct via DashScope
├── agent/
│   ├── agent.py          # Core agent loop
│   ├── prompts.py        # System prompts & safety templates
│   ├── memory.py         # Conversation history
│   └── planner.py        # Task planning & safety checks
├── mcp/
│   ├── client.py         # Salesforce MCP/REST client
│   ├── registry.py       # Tool registry & format conversion
│   └── executor.py       # Tool execution & error handling
├── tools/
│   └── salesforce.py     # 11 tool definitions
├── static/
│   ├── index.html        # Chat UI
│   ├── style.css         # Premium dark theme
│   └── script.js         # WebSocket chat client
├── tests/
│   └── test_agent.py     # Unit tests
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Salesforce org with API access
- [DashScope API key](https://dashscope.console.aliyun.com/) (for Qwen3)

### 1. Install Dependencies

```bash
cd salesforce-qwen-agent
pip install -r requirements.txt
```

### 2. Configure Environment

Edit the `.env` file with your credentials:

```env
# Required: Your DashScope API key for Qwen3
QWEN_API_KEY=your_dashscope_api_key_here

# Salesforce credentials (pre-configured)
SALESFORCE_USERNAME=your_username
SALESFORCE_PASSWORD=your_password
SALESFORCE_SECURITY_TOKEN=your_token
# ... (see .env for all options)
```

### 3. Run the Server

```bash
python app.py
```

### 4. Open the Chat UI

Navigate to **http://localhost:8000** in your browser.

---

## 🔧 Available Tools

| # | Tool | Type | Description |
|---|------|------|-------------|
| 1 | `soqlQuery` | 🟢 Read | Execute SOQL queries |
| 2 | `find` | 🟢 Read | SOSL search across objects |
| 3 | `getRelatedRecords` | 🟢 Read | Traverse object relationships |
| 4 | `listRecentSobjectRecords` | 🟢 Read | View recently accessed records |
| 5 | `getUserInfo` | 🟢 Read | Current user's identity & role |
| 6 | `getObjectSchema` | 🟢 Read | Explore object schemas & fields |
| 7 | `createSobjectRecord` | 🟡 Write | Create new records |
| 8 | `updateSobjectRecord` | 🟡 Write | Update records by ID |
| 9 | `updateRelatedRecord` | 🟡 Write | Update child records via relationship |
| 10 | `deleteSobjectRecord` | 🔴 Delete | Delete records (requires confirmation) |
| 11 | `deleteRelatedRecord` | 🔴 Delete | Delete related records (requires confirmation) |

---

## 💬 Example Queries

```
"Show me the top 10 Accounts by AnnualRevenue"
"What fields does the Opportunity object have?"
"Search for contacts named John"
"Create a new Account named Acme Corp with Industry set to Technology"
"Update the phone number for Account ID 001xx000003ABC"
"Show me all Contacts related to Account 001xx000003ABC"
"Who am I? Show my Salesforce user info"
```

---

## 🔒 Safety

- **Read operations** execute immediately
- **Write operations** (create/update) proceed with a summary
- **Delete operations** always require explicit user confirmation ("yes" to proceed)
- All operations respect your Salesforce permissions and field-level security

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat UI |
| `/chat` | POST | HTTP chat (JSON request/response) |
| `/ws/{session_id}` | WebSocket | Real-time streaming chat |
| `/health` | GET | Health check |

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN_API_KEY` | — | DashScope API key (required) |
| `QWEN_MODEL` | `qwen3-235b-a22b` | Qwen3 model name |
| `APP_PORT` | `8000` | Server port |
| `MAX_CONVERSATION_HISTORY` | `20` | Messages to keep in memory |
| `MAX_TOOL_CALLS_PER_TURN` | `10` | Max tool calls per user message |

---

## 📝 License

MIT
