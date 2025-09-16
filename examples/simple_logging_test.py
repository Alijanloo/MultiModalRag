import sys

from multimodal_rag.frameworks.logging_config import get_logger

def main():
    """Main function to test logging."""
    logger = get_logger(__name__)
    
    print("🧪 Simple Logging Test - Colored Headers")
    print("=" * 45)
    
    logger.debug("🔍 This is a DEBUG message (cyan)")
    logger.info("ℹ️  This is an INFO message (green)")
    logger.warning("⚠️  This is a WARNING message (yellow)")
    logger.error("❌ This is an ERROR message (red)")
    
    # Test without actual critical error
    logger.log(50, "🚨 This is a CRITICAL level message (magenta)")
    
    print("\n✅ Color test completed!")
    print(f"Python version: {sys.version}")
    print(f"Script location: {__file__}")
    
    logger.info("✨ Colored logging test completed successfully")

    print("\n💡 To disable colors, set: MULTIMODAL_RAG_USE_COLORS=false")
    print("💡 To test without colors: MULTIMODAL_RAG_USE_COLORS=false uv run examples/simple_logging_test.py")

if __name__ == "__main__":
    main()
