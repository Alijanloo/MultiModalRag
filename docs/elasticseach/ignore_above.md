`ignore_above` is an Elasticsearch mapping parameter that sets a character limit for string field values during indexing.

## What it does:

- **Ignores values longer than the specified limit** - If a string field value exceeds the `ignore_above` threshold, Elasticsearch will not index that value
- **The field becomes effectively empty** for search purposes when the limit is exceeded
- **Prevents indexing errors** that could occur with extremely long strings

## Common use cases:

```python
# Example mapping with ignore_above
mapping = {
    "properties": {
        "title": {
            "type": "text",
            "ignore_above": 256  # Ignore titles longer than 256 characters
        },
        "description": {
            "type": "keyword",
            "ignore_above": 1000  # Ignore descriptions longer than 1000 characters
        }
    }
}
```

## Key points:

- **Default value**: For `keyword` fields, the default is usually 256 characters
- **Field types**: Commonly used with `keyword` and `text` fields
- **Search impact**: Values exceeding the limit won't be searchable
- **Storage**: The original document is still stored, but the field won't be indexed

## Gotcha:
If you set `ignore_above` too low, you might accidentally exclude important data from search results. Always consider your data's typical length when setting this parameter.

This is particularly useful for preventing issues with unexpectedly long field values that could impact indexing performance or cause errors.