## Highlight in Elasticsearch

**Highlight** is an Elasticsearch feature that marks search terms within the retrieved documents, similar to how search engines highlight your query terms in search results.

```python
"highlight": {
    "fields": {
        "chunk.text": {"fragment_size": 150, "number_of_fragments": 3}
    }
}
```

## Key Components

### 1. **Fields to Highlight**
- `"chunk.text"` - The specific field where you want to highlight matching terms
- Elasticsearch will scan this field for your search query terms

### 2. **Fragment Size**
- `"fragment_size": 150` - Each highlighted snippet will be ~150 characters long
- Think of it as the "window size" around each match

### 3. **Number of Fragments**
- `"number_of_fragments": 3` - Return up to 3 highlighted snippets per document
- Useful when your query matches multiple parts of a long text

## Example Output

If you search for "machine learning" in a document, Elasticsearch might return:

```json
{
  "_source": { /* full document */ },
  "highlight": {
    "chunk.text": [
      "...introduction to <em>machine learning</em> algorithms that...",
      "...supervised <em>machine learning</em> techniques include...",
      "...deep <em>learning</em> is a subset of <em>machine</em>..."
    ]
  }
}
```

## Benefits for RAG Applications

- **Context Preview**: Users see relevant snippets before clicking
- **Relevance Verification**: Confirms the document contains your search terms
- **Better UX**: Highlights help users quickly scan results

This is particularly valuable in RAG systems where you want to show users *why* a document was retrieved and *where* the relevant information appears.