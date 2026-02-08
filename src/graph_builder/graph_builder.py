"""Graph builder for LangGraph workflow"""

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.node.reactnode import RAGNodes

class GraphBuilder:
    """Builds and manages the LangGraph workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize graph builder
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None
    
    def build(self):
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """
        # Create state graph
        builder = StateGraph(RAGState)
        
        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("decision_engine", self.nodes.decision_engine)
        builder.add_node("responder", self.nodes.generate_answer)
        
        # Set entry point
        builder.set_entry_point("retriever")
        
        # Add edges
        builder.add_edge("retriever", "decision_engine")
        
        # Conditional routing from decision_engine
        def route_decision(state: RAGState):
            if state.decision == "out_of_scope":
                return "end"
            elif state.decision == "need_more_context":
                # We can either go to a web search node or just to responder 
                # and let responder handle it (it already has web search logic)
                return "responder"
            else: # answer_from_documents
                return "responder"
                
        builder.add_conditional_edges(
            "decision_engine",
            route_decision,
            {
                "end": END,
                "responder": "responder"
            }
        )
        
        builder.add_edge("responder", END)
        
        # Compile graph
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question: str) -> dict:
        """
        Run the RAG workflow
        
        Args:
            question: User question
            
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()
        
        initial_state = RAGState(question=question)
        result = self.graph.invoke(initial_state)
        
        # Ensure result is a dict for app.py
        if not isinstance(result, dict):
            # Try Pydantic v2
            if hasattr(result, "model_dump"):
                return result.model_dump()
            # Try Pydantic v1
            if hasattr(result, "dict"):
                return result.dict()
                
        return result