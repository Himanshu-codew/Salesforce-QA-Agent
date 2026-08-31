"""
Multi-Agent Orchestrator
Coordinates Planner, DataAgent, ActionAgent, and Synthesizer for robust execution.
"""
import json
import logging
from typing import Any, AsyncGenerator

from llm.base import BaseLLM
from sfmcp.executor import ToolExecutor
from .multi_agent_prompts import PLANNER_PROMPT, DATA_AGENT_PROMPT, ACTION_AGENT_PROMPT, SYNTHESIZER_PROMPT
from .memory import ConversationMemory
from .planner import TaskPlanner
from .rag import ToolRAGRetriever

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrates the multi-agent workflow:
    1. Planner breaks down the query.
    2. Workers (Data/Action) execute sub-tasks.
    3. Synthesizer formats final response.
    """

    def __init__(
        self,
        llm: BaseLLM,
        executor: ToolExecutor,
        max_iterations: int = 20,
        max_history: int = 4,
    ):
        self.llm = llm
        self.executor = executor
        self.safety_planner = TaskPlanner()
        self._memories: dict[str, ConversationMemory] = {}
        self._max_history = max_history
        self.rag_retriever = ToolRAGRetriever(default_top_k=6)

    def _get_memory(self, session_id: str) -> ConversationMemory:
        if session_id not in self._memories:
            self._memories[session_id] = ConversationMemory(max_messages=self._max_history)
        return self._memories[session_id]

    async def process_message(
        self,
        user_message: str,
        session_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        
        memory = self._get_memory(session_id)
        memory.max_messages = self._max_history
        
        # Check pending confirmations (destructive actions)
        if self.safety_planner.has_pending_confirmation(session_id):
            pending = self.safety_planner.process_confirmation(user_message, session_id)
            if pending:
                yield {"type": "thinking", "data": "Executing confirmed operation..."}
                tool_name = pending["tool_name"]
                arguments = pending["arguments"]
                yield {"type": "tool_call", "data": {"name": tool_name, "arguments": arguments}}
                result = await self.executor.execute(tool_name, arguments)
                yield {"type": "tool_result", "data": {"name": tool_name, "result": result}}
                
                # After destruction, synthesize the result
                synth_response = await self._synthesize_response(user_message, [{"tool": tool_name, "result": result}])
                memory.add_user_message(user_message)
                memory.add_assistant_message(synth_response)
                yield {"type": "response", "data": synth_response}
                return
            else:
                decline_msg = "✅ Operation cancelled. No records were deleted."
                memory.add_user_message(user_message)
                memory.add_assistant_message(decline_msg)
                yield {"type": "response", "data": decline_msg}
                return
                
        logger.info(f"📩 [USER MESSAGE] ({session_id}): {user_message}")
        memory.add_user_message(user_message)
        
        # 1. PLANNER STAGE
        yield {"type": "thinking", "data": "[Planner] Decomposing your request..."}
        plan = await self._generate_plan(user_message, memory)
        
        if not plan:
            # Fallback if planning fails
            yield {"type": "response", "data": "I couldn't understand how to break down your request."}
            return
            
        yield {"type": "thinking", "data": f"[Planner] Generated {len(plan)} sub-tasks."}
        
        # 2. EXECUTION STAGE
        tools = self.rag_retriever.get_relevant_tools(user_message, top_k=6)
        all_results = []
        task_outputs = {}
        
        for task in plan:
            task_id = task.get("task_id", 0)
            desc = task.get("description", "")
            agent_type = task.get("agent", "DataAgent")
            
            yield {"type": "thinking", "data": f"[{agent_type}] Executing task: {desc}"}
            
            # Build context from previous dependent tasks
            context = ""
            deps = task.get("depends_on", [])
            for dep_id in deps:
                if dep_id in task_outputs:
                    context += f"\\nResult of Task {dep_id}: {task_outputs[dep_id]}"
                    
            task_prompt = f"Task: {desc}\\nContext: {context}\\nOutput JSON tool calls only."
            
            # Agent execution
            agent_msgs = [{"role": "system", "content": DATA_AGENT_PROMPT if agent_type == "DataAgent" else ACTION_AGENT_PROMPT}, {"role": "user", "content": task_prompt}]
            
            try:
                llm_result = await self.llm.chat_with_tools(messages=agent_msgs, tools=tools, temperature=0.0)
                
                tool_calls = llm_result.get("tool_calls", [])
                task_res = []
                for tc in tool_calls:
                    tc_name = tc.get("name")
                    tc_args = tc.get("arguments", {})
                    
                    # Safety check
                    safety = self.safety_planner.check_tool_safety(tc_name, tc_args, session_id)
                    if safety.get("requires_confirmation"):
                        yield {"type": "response", "data": safety["confirmation_message"]}
                        return # Pause for confirmation
                        
                    yield {"type": "tool_call", "data": {"name": tc_name, "arguments": tc_args}}
                    res = await self.executor.execute(tc_name, tc_args)
                    yield {"type": "tool_result", "data": {"name": tc_name, "result": res}}
                    
                    task_res.append({"tool": tc_name, "result": res})
                    all_results.append({"tool": tc_name, "result": res})
                    
                task_outputs[task_id] = json.dumps(task_res)
                
            except Exception as e:
                logger.error(f"Worker agent failed: {e}")
                yield {"type": "error", "data": f"Failed to execute task: {desc}"}
                
        # 3. SYNTHESIZER STAGE
        yield {"type": "thinking", "data": "[Synthesizer] Formatting final response..."}
        synth_response = await self._synthesize_response(user_message, all_results)
        
        memory.add_assistant_message(synth_response)
        yield {"type": "response", "data": synth_response}
        
    async def _generate_plan(self, user_query: str, memory: ConversationMemory) -> list[dict]:
        msgs = [{"role": "system", "content": PLANNER_PROMPT}]
        # Include a bit of history
        hist = memory.get_messages_for_llm("")
        msgs.extend([m for m in hist if m["role"] != "system"])
        msgs.append({"role": "user", "content": f"Create an execution plan for: {user_query}"})
        
        try:
            res = await self.llm.chat(messages=msgs, temperature=0.0)
            logger.info(f"🤔 [PLANNER RAW OUTPUT]:\n{res}")
            
            content = res.strip()
            # Robust JSON array extraction
            start_idx = content.find("[")
            end_idx = content.rfind("]")
            
            if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                try:
                    json_str = content[start_idx:end_idx+1]
                    plan = json.loads(json_str)
                    if isinstance(plan, list):
                        return plan
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse extracted JSON array: {e}")
                    
            logger.warning("⚠️ Planner did not return a valid JSON list.")
            return []
        except Exception as e:
            logger.error(f"❌ Planning failed! Exception: {e}")
            return []
            
    async def _synthesize_response(self, user_query: str, tool_results: list[dict]) -> str:
        msgs = [{"role": "system", "content": SYNTHESIZER_PROMPT}]
        
        data_context = json.dumps(tool_results, indent=2)
        user_msg = f"Original Query: {user_query}\\n\\nRaw Tool Results:\\n{data_context}\\n\\nPlease provide the final Markdown response."
        msgs.append({"role": "user", "content": user_msg})
        
        try:
            res = await self.llm.chat(messages=msgs, temperature=0.3)
            return res.strip()
        except Exception as e:
            logger.error(f"Synthesizer failed: {e}")
            return "An error occurred while formatting the response."
