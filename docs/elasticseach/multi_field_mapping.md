The `fields` property under the `chunk.text` field in elasticsearch mapping is defining **multi-field mapping** in Elasticsearch. This is a powerful feature that allows you to index the same data in multiple ways for different use cases.

## What's happening here:

```python
"text": {
    "type": "text",                    # Main field: full-text search
    "analyzer": "standard",
    "fields": {
        "keyword": {                   # Sub-field: exact matching
            "type": "keyword", 
            "ignore_above": 256
        }
    }
}
```

## Why use multi-fields?

1. **Primary `text` field**: Uses the `standard` analyzer, which:
   - Tokenizes text into individual words
   - Converts to lowercase
   - Removes punctuation
   - Perfect for full-text search queries like "find documents containing 'machine learning'"

2. **Secondary `keyword` field**: Stores the exact, unanalyzed value:
   - No tokenization or analysis
   - Case-sensitive exact matching
   - Useful for aggregations, sorting, and exact phrase filtering

## Practical example:

If you store the text: `"Machine Learning Algorithms"`

- **text field** stores tokens: `["machine", "learning", "algorithms"]`
- **keyword field** stores: `"Machine Learning Algorithms"` (exact)

## Usage scenarios:

```python
# Full-text search (uses main text field)
query = {"match": {"chunk.text": "machine learning"}}

# Exact match (uses keyword sub-field)  
query = {"term": {"chunk.text.keyword": "Machine Learning Algorithms"}}

# Aggregation (uses keyword sub-field)
aggs = {"unique_chunks": {"terms": {"field": "chunk.text.keyword"}}}
```

## Gotcha:
The `ignore_above: 256` means keyword values longer than 256 characters won't be indexed, saving storage space while handling most typical text chunks.

This pattern is very common in RAG applications where you need both semantic search capabilities and exact matching for deduplication or filtering.