# Telegram Bot Integration

This document describes how to use the Telegram bot integration with the MultiModal RAG system.

## Overview

The Telegram bot provides a conversational interface to the AgenticRAG system, allowing users to:

- Ask questions about indexed documents
- Receive intelligent responses with relevant context
- View images and diagrams from documents when available
- Maintain conversation history for context-aware responses

## Setup

### 1. Create a Telegram Bot

1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` and follow the instructions
3. Copy the bot token you receive

### 2. Configure the Application

1. Copy `config.yaml.example` to `config.yaml`
2. Add your bot token to the configuration:

```yaml
telegram:
  bot_token: "your_telegram_bot_token_here"
  max_message_length: 4096
  max_caption_length: 1024
```

3. Ensure all other services (Elasticsearch, Google GenAI) are properly configured

### 3. Run the Bot

```bash
uv run python examples/telegram_bot_example.py
```

## Bot Commands

- `/start` - Initialize conversation and show welcome message
- `/help` - Show help information
- `/clear` - Clear conversation history

## Features

### Intelligent Document Search

The bot uses the AgenticRAG system to:
- Understand user queries in natural language
- Search through indexed documents
- Provide contextually relevant answers
- Maintain conversation history for follow-up questions

### Image Support

When relevant images or diagrams are found in documents:
- Images are sent as a media group
- The response text is included as a caption
- Multiple images (up to 10) can be sent together
- Long responses are split across multiple messages if needed

### Error Handling

The bot includes robust error handling for:
- Network timeouts
- Telegram API rate limits
- Large message handling
- Image processing errors
- Fallback to text-only responses when image sending fails

## Architecture

The Telegram bot follows the Clean Architecture principles:

```
Frameworks Layer:
├── TelegramBotService (telegram_bot_service.py)
└── Integration with telegram.ext.Application

Use Cases Layer:
├── AgenticRAGUseCase
└── Message processing and response generation

Container:
└── Dependency injection for all services
```

## Usage Examples

### Basic Question
```
User: "What is machine learning?"
Bot: [Searches documents and provides relevant answer with supporting images if available]
```

### Follow-up Question
```
User: "Can you explain more about neural networks?"
Bot: [Uses conversation context to provide more specific information]
```

### Technical Query
```
User: "Show me the architecture diagram for the system"
Bot: [Finds and returns relevant diagrams with explanatory text]
```

## Limitations

- Maximum message length: 4096 characters (Telegram limit)
- Maximum caption length: 1024 characters (Telegram limit)
- Maximum images per message: 10 (Telegram limit)
- Conversation history is kept in memory (limited to last 10 messages per user)

## Troubleshooting

### Bot doesn't respond
- Check bot token configuration
- Verify Elasticsearch is running and accessible
- Check logs for error messages

### Images not showing
- Verify document indexing included image data
- Check image processing in the logs
- Bot will fallback to text-only responses

### Rate limiting
- The bot includes automatic delays between messages
- Consider implementing more sophisticated rate limiting for high-traffic scenarios

## Security Considerations

- Store bot token securely (use environment variables in production)
- Implement user authentication if needed
- Consider message content filtering for sensitive applications
- Monitor and log bot usage for security auditing
