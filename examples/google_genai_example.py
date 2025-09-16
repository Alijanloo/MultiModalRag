"""
Example usage of Google GenAI services in the MultiModal RAG system.

This example demonstrates how to use the embedding and LLM services
integrated through the dependency injection container.
"""

import asyncio
from multimodal_rag import logger
from multimodal_rag.container import ApplicationContainer


async def main():
    """Example usage of Google GenAI services."""
    
    logger.info("Starting Google GenAI services example...")
    
    # Initialize the application container
    container = ApplicationContainer()
    container.config.google_genai.api_key.from_env("GEMINI_API_KEY")
    container.config.google_genai.default_llm_model.from_value("gemini-2.5-flash")
    container.config.google_genai.embedding_model.from_value("gemini-embedding-001")
    container.config.google_genai.embedding_dimensions.from_value(768)
    
    # Wire the container
    container.wire(modules=[__name__])
    
    try:
        # Get services from container
        embedding_service = await container.embedding_service()
        llm_service = await container.llm_service()
        
        logger.info("Testing embedding service...")
        
        # Test embedding service
        test_text = "What is the meaning of life?"
        embedding = await embedding_service.embed_single(test_text)
        logger.info(f"Generated embedding with {len(embedding)} dimensions")
        
        # Test batch embeddings
        texts = ["Hello world", "How are you?", "Good morning"]
        embeddings = await embedding_service.embed_content(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        logger.info("Testing LLM service...")
        
        # Test LLM service
        prompt = "Explain how AI works in a few words"
        response = await llm_service.generate_content(prompt)
        logger.info(f"LLM Response: {response}")
        
        # Test structured content generation
        structured_prompt = "Generate a simple profile for a fictional character"
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "occupation": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["name", "age", "occupation"]
        }
        
        structured_response = await llm_service.generate_structured_content(
            structured_prompt, 
            response_schema=schema
        )
        logger.info(f"Structured Response: {structured_response}")
        
        # Show available models
        models = llm_service.get_available_models()
        logger.info(f"Available models: {models}")
        
        # Show embedding dimensions
        dimensions = embedding_service.get_embedding_dimensions()
        logger.info(f"Embedding dimensions: {dimensions}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.info("Make sure you have set the GEMINI_API_KEY environment variable")
    
    finally:
        # Cleanup container resources
        await container.shutdown_resources()


if __name__ == "__main__":
    asyncio.run(main())
