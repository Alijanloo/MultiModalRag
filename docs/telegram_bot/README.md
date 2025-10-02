# Telegram Bot Integration

The Telegram bot provides a conversational interface to the AgenticRAG system with intelligent document search, image support, and modular architecture.

## Quick Setup

### 1. Create Bot Token
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` and follow instructions
3. Copy the bot token

### 2. Configure
Add your bot token to `config.yaml`:

```yaml
telegram:
  bot_token: "your_telegram_bot_token_here"
```

### 3. Run
```bash
uv run python examples/telegram_bot_example.py
```

## Key Features

- **Intelligent Search**: Natural language queries with contextual answers
- **Image Support**: Automatic display of relevant images from documents  
- **Source Transparency**: Interactive buttons to view source document chunks
- **Conversation Context**: Maintains chat history for follow-up questions

## Commands

- `/start` - Initialize conversation
- `/help` - Show help information  
- `/clear` - Clear conversation history

## Architecture

Modular design with clean separation of concerns:

```
multimodal_rag/frameworks/telegram_bot/
├── telegram_bot_service.py (Main orchestrator)
├── conversation_manager.py (Chat history)
├── chunk_manager.py (Document chunks)
├── response_formatter.py (Message formatting)
└── message_handlers.py (Command handling)
```

**Key Components:**
- **TelegramBotService**: Application lifecycle and coordination
- **ConversationManager**: User chat history management
- **ChunkManager**: Document chunk storage and retrieval
- **ResponseFormatter**: Message formatting with image support
- **MessageHandlers**: Command processing and AgenticRAG integration

## Usage Example

```
User: "What is machine learning?"
Bot: [Intelligent response with images if available]
     [📄 chunk_1_2834] [📄 chunk_2_5791] ← Click to view sources

User: [Clicks chunk button]
Bot: 📄 Document Chunk: chunk_1_2834
     📋 Document ID: ml_textbook_chapter_1
     Content: [Full chunk text...]
```

## Development

### Component Testing
```python
# Test individual components
conversation_manager = ConversationManager()
chunk_manager = ChunkManager()
response_formatter = ResponseFormatter()
```

### Adding Features
- **New Commands**: Extend `MessageHandlers`
- **Response Types**: Modify `ResponseFormatter`  
- **Conversation Logic**: Update `ConversationManager`
