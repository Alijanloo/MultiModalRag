# MultiModalRag

A Multi-Modal RAG (Retrieval-Augmented Generation) application for intelligently processing and indexing unstructured medical PDF documents, including text, tables, and visual content for comprehensive information retrieval.

## Overview

This project leverages Docling to intelligently structure unstructured medical PDF files by recognizing hierarchical content organization, tables, and visual elements. Docling uses advanced OCR models to identify document structure and convert raw content into clean markdown format with proper hierarchical headings, sections, and structured tables. The system also generates detailed JSON representations with complete parent/child references for every document element.

The application integrates with Google's Gemini 2.5-Flash multi-modal model for intelligent picture description and analysis. Docling's intelligent chunker merges related elements to create contextually rich chunks within specified token length limitations, while preserving hierarchical information for enhanced retrieval accuracy.

## Technologies Used

- **Docling**: Advanced document processing with OCR-based structure recognition
- **Google Gemini 2.5-Flash**: Multi-modal AI for intelligent image and diagram analysis
- **FastAPI**: For building the web application and API endpoints
- **Elasticsearch**: For storing and retrieving indexed multi-modal data
- **LangChain**: For advanced text processing and chunking strategies
- **OCR Models**: For recognizing document structure, tables, and hierarchical content
- **PyMuPDF**: For PDF file processing and manipulation

## Features

- **Intelligent Document Structure Recognition**: Advanced OCR-based processing to identify hierarchical content organization in medical PDFs
- **Multi-Modal Content Processing**: 
  - Clean markdown conversion with proper hierarchical headings and sections
  - Structured table extraction and formatting
  - Visual content analysis using Gemini 2.5-Flash for picture and diagram description
- **Hierarchical JSON Representation**: Complete document structure with parent/child element references
- **Context-Aware Chunking**: Docling's intelligent chunker merges related elements while maintaining hierarchical context
- **Token-Optimized Processing**: Configurable chunk sizes within specified token length limitations
- **Comprehensive Retrieval**: Access to information encoded in text, tables, and visual diagrams
- **Medical Document Specialization**: Optimized for complex medical PDF structures and terminology
- **Elasticsearch Integration**: Efficient indexing and retrieval of multi-modal content
- **Clean Architecture**: Separation of concerns with modular design patterns

## Installation

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
```

# ElasticSearch Setup
Elasticsearch needs a higher `vm.max_map_count` (≥ 262144) for its memory-mapped files.

1. Edit the sysctl configuration file:

   ```bash
   sudo nano /etc/sysctl.conf
   ```

2. Add this line at the end (or edit if it already exists):

   ```
   vm.max_map_count=262144
   ```

3. Save and reload the settings without rebooting:

   ```bash
   sudo sysctl --system
   ```

4. Verify:

   ```bash
   sysctl vm.max_map_count
   # Should show: vm.max_map_count = 262144
   ```

Finally run the elastic container:
```
docker run --name es01 --net elastic -p 9200:9200 -d -m 1GB elasticsearch:9.1.2
```

reset and get the password for the user
```
docker exec -it es01 /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic
```
