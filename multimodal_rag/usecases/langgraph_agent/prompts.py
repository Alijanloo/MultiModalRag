"""Prompts for the agentic RAG system."""


class AgenticRAGPrompts:
    """Collection of prompts used in the agentic RAG workflow."""

    @staticmethod
    def get_query_or_respond_prompt(
        conversation_history: str, current_message: str
    ) -> str:
        """Prompt for deciding whether to query documents or respond directly."""
        return f"""You are an intelligent assistant that can either respond directly to users or search for information when needed.

Conversation history:
{conversation_history}

Current user message: {current_message}

If the user's question requires specific information that you don't have or is about document content, use the retrieve_documents tool to search for relevant information.
If the user's question is a general greeting, conversation, or something you can answer without additional context, respond directly.

Available tools:
- retrieve_documents: Search for relevant documents and information"""

    @staticmethod
    def get_document_grading_prompt(context: str, question: str) -> str:
        """Prompt for grading document relevance."""
        return f"""You are a grader assessing relevance of retrieved document content to a user question.

Retrieved content: {context}

User question: {question}

If the content contains information related to the user question, respond with 'yes'.
If the content is not relevant or doesn't contain useful information, respond with 'no'.

Respond with only 'yes' or 'no'."""

    @staticmethod
    def get_query_rewrite_prompt(original_query: str) -> str:
        """Prompt for rewriting queries for better retrieval."""
        return f"""Look at the input and try to reason about the underlying semantic intent and meaning.

Original query: {original_query}

Formulate an improved query that would be better for searching and finding relevant information:"""

    @staticmethod
    def get_answer_generation_prompt(history: str, context: str) -> str:
        """Prompt for generating final answers."""
        return f"""You are a conversational chatbot for question-answering tasks. Use the following retrieved context to interact with user.

If you don't know the answer based on the context, just say that you don't know.
Keep the answer concise and informative.

Context: {context}

Conversation History: {history}
"""

    @staticmethod
    def get_retriever_tool_definition() -> dict:
        """Tool definition for the document retriever."""
        return {
            "name": "retrieve_documents",
            "description": "Search and return information from the document repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for retrieving relevant documents",
                    }
                },
                "required": ["query"],
            },
        }

    @staticmethod
    def get_answer_response_schema() -> dict:
        """Response schema for structured answer generation."""
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The main answer content",
                },
                "chunk_ids_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of chunk IDs used to generate this answer",
                },
            },
            "required": ["answer", "chunk_ids_used"],
        }
