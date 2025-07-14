# TableRag

A RAG (Retrieval-Augmented Generation) application for indexing tables and raw text content from PDF documents and inferring on them.

## Overview

This project uses Docling to convert pdf files along with their containing pdfs into a clean markdown file, chunking based on the detected headings, and indexing them for efficient retrieval. The application utilizes various technologies, including FastAPI for the web framework, Elasticsearch for indexing and searching, and LangChain for text chunking and processing.

## Technologies Used

- **FastAPI**: For building the web application.
- **Elasticsearch**: For storing and retrieving indexed data.
- **LangChain**: For text chunking and processing.
- **Table Transformer**: For recognizing table structures in PDF documents.
- **PyMuPDF**: For reading PDF files.

## Features

- PDF processing with page-by-page table detection
- Table structure recognition using Table Transformer
- Text chunking with LangChain's character splitter
- Elasticsearch indexing for structured retrieval
- Clean architecture with separation of concerns

## Installation

```bash
# Install the package in development mode
pip install -e .
```
