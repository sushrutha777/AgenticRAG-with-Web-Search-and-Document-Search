"""LangGraph nodes for RAG workflow with custom ReAct agent."""

from typing import List, Optional
from src.state.rag_state import RAGState

from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


class RAGNodes:
    """Contains node functions for RAG workflow."""

    def __init__(self, retriever, llm):
        self.retriever = retriever      # VectorStoreRetriever
        self.llm = llm                  # Chat model
        self.tools = {}

    # 1. RETRIEVE DOCUMENTS NODE
    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieve relevant docs for the given question."""
        docs = self.retriever.invoke(state.question)   # <-- FIXED
        state.retrieved_docs = docs
        return state

    # 2. BUILD TOOLSET (retriever + wikipedia + websearch)
    def _build_tools(self):
        from langchain_community.utilities import WikipediaAPIWrapper
        from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
        from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
        from langchain_core.documents import Document

        # RETRIEVER TOOL
        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
    
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = getattr(d, "metadata", {})
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
    
            return "\n\n".join(merged)
    
        retriever_tool = Tool(
            name="retriever",
            description="Search indexed corpus for relevant text.",
            func=retriever_tool_fn,
        )
    
        # WIKIPEDIA TOOL
        wiki_api = WikipediaAPIWrapper(top_k_results=3, lang="en")
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge.",
            func=wiki_api.run,
        )
        # FREE DUCKDUCKGO WEB SEARCH TOOL
        ddg = DuckDuckGoSearchRun()
    
        websearch_tool = Tool(
            name="web_search",
            description="Free unlimited web search using DuckDuckGo.",
            func=ddg.run,
        )
        # REGISTER ALL TOOLS
        self.tools = {
            "retriever": retriever_tool,
            "wikipedia": wikipedia_tool,
            "web_search": websearch_tool,
        }
        
    # 3. EXECUTE A TOOL BY NAME
    def _run_tool(self, name: str, input: str) -> str:
        tool = self.tools.get(name)
        if not tool:
            return f"Tool '{name}' does not exist."
        try:
            return tool.func(input)
        except Exception as e:
            return f"Tool '{name}' failed: {e}"

    # 4. GENERATE ANSWER WITH CUSTOM REACT LOOP
    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using a manual ReAct loop."""
        question = state.question

        if not self.tools:
            self._build_tools()

        # Step 1: Ask LLM what to do
        think_prompt = (
            "You are a ReAct agent.\n"
            "Tools available: retriever, wikipedia.\n"
            "Decide whether to answer directly OR call a tool.\n"
            "Respond EXACTLY in this JSON format:\n\n"
            "{\n"
            '  "tool": "<tool name or none>",\n'
            '  "input": "<tool input or empty>"\n'
            "}\n\n"
            f"User question: {question}"
        )

        try:
            decision_msg = self.llm.invoke(think_prompt)
            decision_text = getattr(decision_msg, "content", str(decision_msg))
        except Exception as e:
            state.answer = f"LLM error: {e}"
            return state

        import json
        tool = None
        tool_input = question

        try:
            parsed = json.loads(decision_text)
            tool = parsed.get("tool")
            if parsed.get("input"):
                tool_input = parsed["input"]
        except:
            pass

        # Step 2: Run selected tool
        tool_result = ""
        if tool and tool.lower() in self.tools:
            tool_result = self._run_tool(tool.lower(), tool_input)

        # Step 3: Final Answer Synthesis
        final_prompt = (
            "You previously chose a tool. Here is its result:\n\n"
            f"{tool_result}\n\n"
            "Now answer the user's original question:\n"
            f"{question}\n\n"
            "Final Answer:"
        )

        try:
            final_msg = self.llm.invoke(final_prompt)
            answer = getattr(final_msg, "content", str(final_msg))
        except Exception as e:
            answer = f"LLM error during final answer: {e}"

        state.answer = answer
        return state
