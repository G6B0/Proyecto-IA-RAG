# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Legal RAG (Retrieval-Augmented Generation) system specialized in Chilean Consumer Law (Ley 19.496). The system combines legal documents with judicial precedents to provide contextualized legal assistance through an interactive CLI interface.

## Common Commands

### Running the Application
```bash
python main.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Setting up Ollama Models (Required)
```bash
# Start Ollama service
ollama serve

# Download required models
ollama pull llama3:8b
ollama pull nomic-embed-text
```

### Testing Ollama Connection
```bash
# Test if models are available
ollama list

# Test model response
ollama run llama3:8b "Hello"
```

### Data Files Verification
The system requires these CSV files in the `data/` directory:
- `Ley_consumidor_limpio.csv` - Consumer law articles
- `Fallos_judiciales_ley_19.496.csv` - Judicial precedents

## Architecture

### Core Components

1. **LegalAgent** (`src/legal_agent.py`): Main conversational agent that orchestrates the RAG system using LangGraph for state management and conversation memory.

2. **RAGSystem** (`src/rag_system.py`): Core RAG implementation with dual vector stores (law articles and judicial cases) using Chroma for document retrieval.

3. **DataLoader** (`src/data_loader.py`): Handles loading and processing of CSV legal documents with appropriate chunking strategies.

4. **Config** (`src/config.py`): Centralized configuration including model settings, chunking parameters, and prompt templates.

### Key Architecture Details

- **Dual Vector Stores**: Separate Chroma collections for law articles and judicial cases to enable targeted retrieval
- **Memory Management**: Uses LangGraph's MemorySaver for persistent conversation threads with UUID-based thread IDs
- **Document Processing**: Different chunking strategies for law articles (1000 chars) vs judicial cases (2000 chars)
- **Context Preservation**: Maintains conversation history across queries with contextualization using LangChain BaseMessage system
- **State Management**: Uses TypedDict ConversationState for LangGraph workflow with messages, queries, and sources

### Models Used
- **LLM**: `llama3:8b` via Ollama for text generation
- **Embeddings**: `nomic-embed-text` via Ollama for document vectorization

### Data Flow
1. User query → LegalAgent (`legal_agent.py:77`)
2. Query contextualization using conversation history (`legal_agent.py:111`)
3. Parallel retrieval from law and case vector stores (`rag_system.py:103`, `rag_system.py:124`)
4. Context formatting and LLM generation (`rag_system.py:145`)
5. Response formatting with source citations (`legal_agent.py:177`)

### LangGraph Workflow
The system uses a compiled LangGraph workflow with:
- **State Schema**: ConversationState with typed message handling
- **Process Node**: Single node that handles query processing, retrieval, and response generation
- **Checkpointer**: MemorySaver for conversation persistence
- **Thread Management**: UUID-based thread identification for conversation isolation

## Configuration

All configuration is centralized in `src/config.py`:
- Model settings (LLM_MODEL, EMBEDDING_MODEL)
- Chunking parameters (CHUNK_SIZE_LAW, CHUNK_SIZE_CASES)
- Retrieval settings (RETRIEVAL_K)
- File paths and prompts

## Vector Database

The system uses Chroma vector database stored in `./chroma_db/`:
- Creates vector stores on first run if they don't exist
- Persists embeddings across sessions
- Separate collections for law articles and judicial cases

## Interactive Commands

The CLI interface supports these commands:
- `/ayuda` - Show help and usage examples
- `/historial` - View conversation history
- `/limpiar` - Clear conversation history
- `/guardar` - Save conversation to JSON file
- `/cargar` - Load conversation from JSON file
- `/estado` - Show system status and document counts
- `/thread` - View/change conversation thread ID
- `/salir` - Exit the application

## Error Handling

Common issues and solutions:
- **Ollama not running**: Start with `ollama serve`
- **Models not found**: Install with `ollama pull llama3:8b` and `ollama pull nomic-embed-text`
- **Data files missing**: Ensure CSV files exist in `data/` directory with correct names
- **Memory issues**: Reduce CHUNK_SIZE in config.py