# 🧠 MultiModal Agentic RAG

**A Clean-Architecture, Multimodal, Agentic RAG System for Intelligent Medical Document Processing**

---

## 📘 Overview

The **MultiModal Agentic RAG** is designed to **intelligently process and retrieve information from complex medical documents**—particularly the *Oxford Textbook of Medicine (6th Edition)*—by leveraging a **Retrieval-Augmented Generation (RAG)** pipeline with **multimodal (text, table, and image) understanding**.

It integrates advanced models for **document structure analysis, semantic chunking, and agentic reasoning**, enabling context-aware, explainable question answering.

---

## 🎯 Project Goals

* **Convert** unstructured PDFs into structured and searchable formats (Markdown & JSON).
* **Preserve** document hierarchy, semantic relationships, and contextual information.
* **Analyze** multimodal content—text, tables, and diagnostic images—through specialized models.
* **Build** an intelligent retrieval system using RAG with hierarchical context preservation.
* **Enable** accurate medical Q&A using information retrieved from text, tables, and images.
* **Implement** modular, maintainable system architecture following Clean Architecture principles.

---

## 🏗️ System Architecture

This project follows **Clean Architecture** to ensure high scalability, maintainability, and testability.
Dependencies flow **inward**—from frameworks toward the core business logic.

### Layers

| Layer                    | Description                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------- |
| **Entities**             | Core data models and validation (via **Pydantic**) independent of frameworks.      |
| **Use Cases**            | Implements the application’s business logic and RAG workflows using **LangGraph**. |
| **Interface Adapters**   | Bridges entities and external interfaces (e.g., APIs, vector stores).              |
| **Frameworks & Drivers** | Outer layer containing frameworks (FastAPI, OpenAI, Cohere, etc.).                 |

### Key Design Principles

* **Dependency Rule**: Inner layers never depend on outer layers.
* **Dependency Inversion**: Interfaces are defined at the core, implementations in outer layers.
* **Dependency Injection**: Managed via the `dependency-injector` library for modular configuration.
* **Logging & Error Handling**: Unified multi-level logging with structured error tracing and Prometheus metrics.

---

## 🧩 Core Components

### 🧱 1. Docling-Based Document Processing

The system uses **[Docling](https://github.com/IBM/Docling)**, an open-source library developed by IBM Research, to parse and structure medical PDFs.

**Key Features:**

* Hierarchical structure recognition (titles, sections, paragraphs).
* Accurate table extraction via **TableFormer**.
* Layout analysis using **Heron (RT-DETR + ResNet-50)**.
* Optical Character Recognition (OCR) with **EasyOCR (CRAFT + CRNN)**.
* Image captioning with **Gemini 2.5-Flash** (for complex medical visuals).
* Intelligent contextual chunking that preserves semantic relations.
* Structured output in **Markdown** and **JSON** for downstream processing.

**Why Docling?**
Unlike traditional tools (e.g., PyPDF2, PDFMiner), Docling maintains hierarchical document structures, integrates multimodal vision-language models, and supports high accuracy in complex tables and diagrams.

---

### 🧠 2. Agentic RAG System

The RAG system uses **LangGraph** to create an **agentic reasoning workflow**, where each node represents a step in the retrieval and generation process.

**Key Capabilities:**

* Intelligent decision-making to determine when to retrieve or directly answer.
* Context-aware document retrieval with **Elasticsearch**.
* Query rewriting for unanswerable or ambiguous questions.
* Answer generation with detailed source attribution (`chunk_ids_used`).
* Visual grounding: retrieves and attaches related images and metadata to answers.
* Context preservation across dialogue sessions.

---

### 🖼️ 3. Multimodal Integration

The system supports multimodal inputs:

* **Text:** Paragraphs, summaries, and semantic relationships.
* **Tables:** Extracted with TableFormer and stored with relational metadata.
* **Images:** Described by Gemini 2.5-Flash and linked to related textual content.

Each chunk is semantically enriched with metadata (e.g., section, page number, caption, and hierarchy).

---

### 🤖 4. Telegram Bot Interface

A **Telegram bot interface** provides interactive access to the RAG system:

* Displays retrieved chunks with inline buttons for transparency.
* Allows users to inspect detailed source data for generated answers.
* Provides human-readable, structured explanations alongside responses.

---

## ⚙️ Technologies & Frameworks

| Category                  | Tools                                   |
| ------------------------- | --------------------------------------- |
| **Presenter Frameworks**  | PythonTelegramBot                       |
| **Data Stores**           | Elasticsearch                           |
| **Document Processing**   | Docling, EasyOCR, TableFormer, Heron    |
| **Dependency Management** | dependency-injector                     |
| **Data Models**           | Pydantic                                |
| **Workflow Management**   | LangGraph                               |

---


---

## 📚 References

* *Oxford Textbook of Medicine, 6th Edition*
* Docling: [https://github.com/IBM/Docling](https://github.com/IBM/Docling)
* LangGraph: [https://github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
* TableFormer, Heron, EasyOCR, Gemini 2.5-Flash documentation
* Comparison study: [Docling vs GPT-5](https://github.com/Alijanloo/MultiModalRag/tree/master/docs/compare%20docling%20with%20GPT-5)

---

## 🏁 Conclusion

**MultiModal Agentic RAG** demonstrates a scalable, modular, and multimodal approach to **retrieval-augmented generation** for complex domains like medicine.
By combining **Clean Architecture, Docling, LangGraph**, and **agentic reasoning**, it achieves interpretable, context-rich, and efficient knowledge retrieval—paving the way for **trustworthy AI assistants in medical research and education.**

---

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
