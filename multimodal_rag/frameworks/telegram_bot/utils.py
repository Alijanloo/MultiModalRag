def escape_markdown(text: str) -> str:
    """Escape Markdown special characters for Telegram.

    Args:
        text: Text to escape

    Returns:
        Escaped text safe for Markdown parsing
    """
    # Characters that need escaping in Telegram Markdown (v1)
    # For Markdown v1, we need to escape: *, _, `, [
    escape_chars = ["*", "`", "["]

    escaped_text = text.replace("_", "-")
    for char in escape_chars:
        escaped_text = escaped_text.replace(char, f"\\{char}")

    return escaped_text
