"""Agentic RAG implementation using LangGraph."""

from typing import Dict, List, Optional, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from multimodal_rag.frameworks.logging_config import get_logger
from multimodal_rag.usecases.interfaces.document_repository import (
    IDocumentIndexRepository,
)
from multimodal_rag.usecases.interfaces.embedding_service import (
    EmbeddingServiceInterface,
)
from multimodal_rag.usecases.interfaces.llm_service import LLMServiceInterface
from multimodal_rag.entities.document import DocChunk, DocumentPicture
from .dtos import AgentResponse, ConversationState, ChatMessage
from .prompts import AgenticRAGPrompts

logger = get_logger(__name__)


class AgentState(TypedDict):
    """Extended state for the agentic RAG workflow."""

    messages: List[BaseMessage]
    current_chunks: Dict[str, DocChunk]


class AgenticRAGUseCase:
    """Agentic RAG implementation using LangGraph for intelligent document retrieval and response generation."""

    def __init__(
        self,
        document_repository: IDocumentIndexRepository,
        embedding_service: EmbeddingServiceInterface,
        llm_service: LLMServiceInterface,
        index_name: Optional[str] = None,
        retrieval_size: int = 10,
    ):
        """Initialize the agentic RAG use case.

        Args:
            document_repository: Repository for document operations
            embedding_service: Service for generating embeddings
            llm_service: Service for LLM operations
            index_name: Elasticsearch index name
            retrieval_size: Number of documents to retrieve
        """
        self._document_repository = document_repository
        self._embedding_service = embedding_service
        self._llm_service = llm_service
        self._index_name = index_name
        self._retrieval_size = retrieval_size
        self._graph = None
        self._retriever_tool = None

        self._initialize_tools()
        self._build_workflow()

    def _initialize_tools(self) -> None:
        """Initialize the retriever tool for the agent."""

        @tool
        async def retrieve_documents(query: str) -> str:
            """Search and return information from the document repository.

            Args:
                query: Search query for retrieving relevant documents

            Returns:
                Retrieved document content as formatted string with chunk IDs
            """
            try:
                vector = await self._embedding_service.embed_single(query)

                chunks = await self._document_repository.search_chunks(
                    query=query,
                    vector=vector,
                    size=self._retrieval_size,
                    index_name=self._index_name,
                )

                if not chunks:
                    return "No relevant documents found for the query."

                formatted_content = []
                for i, chunk in enumerate(chunks, 1):
                    chunk_id = f"chunk_{i}_{hash(chunk.text) % 10000}"
                    content = f"[CHUNK_ID: {chunk_id}]\nDocument {i}:\n{chunk.text}"
                    if chunk.meta and chunk.meta.headings:
                        content = f"[CHUNK_ID: {chunk_id}]\nDocument {i} (Headings: {chunk.meta.headings}):\n{chunk.text}"
                    formatted_content.append(content)

                return "\n\n".join(formatted_content)

            except Exception as e:
                logger.error(f"Error in retrieve_documents tool: {e}")
                return f"Error retrieving documents: {str(e)}"

        self._retriever_tool = retrieve_documents

    async def _generate_query_or_respond(
        self, state: AgentState
    ) -> Dict[str, List[BaseMessage]]:
        """Generate a response or decide to retrieve documents based on the current state."""
        try:
            conversation_history = []
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    conversation_history.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    conversation_history.append(f"Assistant: {msg.content}")

            current_message = state["messages"][-1].content if state["messages"] else ""
            conversation_context = chr(10).join(conversation_history[-3:])

            prompt = AgenticRAGPrompts.get_query_or_respond_prompt(
                conversation_context, current_message
            )

            tools = [AgenticRAGPrompts.get_retriever_tool_definition()]

            response_data = await self._llm_service.generate_content_with_tools(
                prompt, tools=tools
            )

            if response_data.get("has_function_call", False):
                function_calls = response_data.get("function_calls", [])
                tool_call = function_calls[0]
                tool_call_id = f"call_{hash(str(tool_call)) % 10000}"
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": tool_call_id,
                            "name": tool_call["name"],
                            "args": tool_call["args"],
                        }
                    ],
                )
            else:
                response = AIMessage(
                    content=response_data.get("text", "I'll help you with that.")
                )

            return {"messages": [response]}

        except Exception as e:
            logger.error(f"Error in _generate_query_or_respond: {e}")
            error_response = AIMessage(
                content="I apologize, but I encountered an error while processing your request."
            )
            return {"messages": [error_response]}

    async def _grade_documents(
        self, state: AgentState
    ) -> Literal["generate_answer", "rewrite_question"]:
        """Determine whether the retrieved documents are relevant to the question."""
        try:
            if len(state["messages"]) < 3:
                return "rewrite_question"

            question = state["messages"][0].content
            context = state["messages"][-1].content

            grade_prompt = AgenticRAGPrompts.get_document_grading_prompt(
                context, question
            )

            response = await self._llm_service.generate_content(grade_prompt)

            if "yes" in response.lower():
                return "generate_answer"
            else:
                return "rewrite_question"

        except Exception as e:
            logger.error(f"Error in _grade_documents: {e}")
            return "rewrite_question"

    async def _rewrite_question(
        self, state: AgentState
    ) -> Dict[str, List[BaseMessage]]:
        """Rewrite the original user question for better retrieval."""
        try:
            original_question = state["messages"][0].content

            rewrite_prompt = AgenticRAGPrompts.get_question_rewrite_prompt(
                original_question
            )

            rewritten_content = await self._llm_service.generate_content(rewrite_prompt)

            rewritten_message = HumanMessage(content=rewritten_content)
            return {"messages": [rewritten_message]}

        except Exception as e:
            logger.error(f"Error in _rewrite_question: {e}")
            return {"messages": [state["messages"][0]]}

    async def _generate_answer(self, state: AgentState) -> Dict[str, List[BaseMessage]]:
        """Generate final answer based on retrieved context."""
        try:
            question = state["messages"][0].content
            context = state["messages"][-1].content

            generate_prompt = AgenticRAGPrompts.get_answer_generation_prompt(
                question, context
            )

            response_schema = AgenticRAGPrompts.get_answer_response_schema()

            if hasattr(self._llm_service, "generate_structured_content"):
                structured_response = (
                    await self._llm_service.generate_structured_content(
                        generate_prompt, response_schema=response_schema
                    )
                )

                answer_content = structured_response.get(
                    "answer", "I couldn't generate an answer."
                )
                chunk_ids_used = structured_response.get("chunk_ids_used", [])

                metadata_info = []
                if chunk_ids_used:
                    metadata_info.append(f"Used chunks: {', '.join(chunk_ids_used)}")

                if metadata_info:
                    answer_content += f"\n\n[Metadata: {'; '.join(metadata_info)}]"

                response = AIMessage(content=answer_content)
            else:
                answer = await self._llm_service.generate_content(generate_prompt)
                response = AIMessage(content=answer)

            return {"messages": [response]}

        except Exception as e:
            logger.error(f"Error in _generate_answer: {e}")
            error_response = AIMessage(
                content="I apologize, but I couldn't generate an answer based on the retrieved information."
            )
            return {"messages": [error_response]}

    def _build_workflow(self) -> None:
        """Build the LangGraph workflow for agentic RAG."""
        try:
            workflow = StateGraph(AgentState)

            workflow.add_node(
                "generate_query_or_respond", self._generate_query_or_respond
            )
            workflow.add_node("retrieve", ToolNode([self._retriever_tool]))
            workflow.add_node("rewrite_question", self._rewrite_question)
            workflow.add_node("generate_answer", self._generate_answer)

            workflow.add_edge(START, "generate_query_or_respond")

            workflow.add_conditional_edges(
                "generate_query_or_respond",
                tools_condition,
                {
                    "tools": "retrieve",
                    END: END,
                },
            )

            workflow.add_conditional_edges(
                "retrieve",
                self._grade_documents,
            )

            workflow.add_edge("generate_answer", END)
            workflow.add_edge("rewrite_question", "generate_query_or_respond")

            self._graph = workflow.compile()

        except Exception as e:
            logger.error(f"Error building workflow: {e}")
            raise

    async def process_message(
        self,
        message: str,
        chat_id: Optional[str] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> AgentResponse:
        """Process a user message and return an agent response with retrieved context.

        Args:
            message: User message to process
            chat_id: Optional chat session identifier
            conversation_history: Optional conversation history

        Returns:
            AgentResponse with generated content and associated pictures
        """
        try:
            messages = []

            if conversation_history:
                for hist_msg in conversation_history[-5:]:
                    if hist_msg.role == "user":
                        messages.append(HumanMessage(content=hist_msg.content))
                    elif hist_msg.role == "assistant":
                        messages.append(AIMessage(content=hist_msg.content))

            messages.append(HumanMessage(content=message))

            initial_state = {"messages": messages, "current_chunks": {}}

            final_state = None
            async for chunk in self._graph.astream(initial_state):
                final_state = chunk

            if not final_state:
                raise ValueError("Workflow did not produce any output")

            final_messages = None
            for node_name, node_state in final_state.items():
                if "messages" in node_state:
                    final_messages = node_state["messages"]

            if not final_messages:
                raise ValueError("No messages found in final state")

            response_content = final_messages[-1].content

            chunks_used = []
            chunk_ids_used = []
            pictures = []

            if "[Metadata:" in response_content:
                import re

                chunk_id_pattern = r"Used chunks: ([^;]+)"
                match = re.search(chunk_id_pattern, response_content)
                if match:
                    chunk_ids_text = match.group(1)
                    chunk_ids_used = [cid.strip() for cid in chunk_ids_text.split(",")]

            await self._extract_chunks_and_pictures_from_response(
                final_messages, chunks_used, pictures, chunk_ids_used, chat_id
            )

            return AgentResponse(
                content=response_content,
                chunks_used=chunks_used,
                chunk_ids_used=chunk_ids_used,
                pictures=pictures,
                chat_id=chat_id,
                metadata={"workflow_steps": len(final_state)},
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return AgentResponse(
                content="I apologize, but I encountered an error while processing your message.",
                chat_id=chat_id,
                metadata={"error": str(e)},
            )

    async def _extract_chunks_and_pictures_from_response(
        self,
        messages: List[BaseMessage],
        chunks_used: List[DocChunk],
        pictures: List[DocumentPicture],
        chunk_ids_used: List[str],
        chat_id: Optional[str],
    ) -> None:
        """Extract chunks and pictures from the response based on chunk IDs used."""
        try:
            for message in messages:
                if (
                    isinstance(message, ToolMessage)
                    and message.name == "retrieve_documents"
                ):
                    query = None
                    for msg in messages:
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                if tool_call["name"] == "retrieve_documents":
                                    query = tool_call["args"].get("query")
                                    break

                    if query:
                        vector = await self._embedding_service.embed_single(query)
                        retrieved_chunks = (
                            await self._document_repository.search_chunks(
                                query=query,
                                vector=vector,
                                size=self._retrieval_size,
                                index_name=self._index_name,
                            )
                        )

                        chunk_id_to_chunk = {}
                        for i, chunk in enumerate(retrieved_chunks, 1):
                            chunk_id = f"chunk_{i}_{hash(chunk.text) % 10000}"
                            chunk_id_to_chunk[chunk_id] = chunk

                        for chunk_id in chunk_ids_used:
                            if chunk_id in chunk_id_to_chunk:
                                chunks_used.append(chunk_id_to_chunk[chunk_id])

                        if not chunk_ids_used:
                            chunks_used.extend(retrieved_chunks)

                        for chunk in chunks_used:
                            if chunk.meta and hasattr(chunk.meta, "doc_items"):
                                doc_items = chunk.meta.doc_items
                                if doc_items:
                                    for item in doc_items:
                                        if (
                                            hasattr(item, "label")
                                            and "picture" in item.label.lower()
                                        ):
                                            picture_id = getattr(
                                                item, "picture_id", None
                                            )
                                            document_id = chunk.document_id

                                            if picture_id and document_id:
                                                picture = await self._document_repository.get_picture(
                                                    document_id=document_id,
                                                    picture_id=picture_id,
                                                    index_name=self._index_name,
                                                )
                                                if picture:
                                                    pictures.append(picture)
                        break

        except Exception as e:
            logger.error(f"Error extracting chunks and pictures from response: {e}")

    async def get_conversation_state(self, chat_id: str) -> Optional[ConversationState]:
        """Get the current state of a conversation."""
        return None

    async def save_conversation_state(self, state: ConversationState) -> bool:
        """Save the current state of a conversation."""
        return True
