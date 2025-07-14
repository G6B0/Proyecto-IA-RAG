import os
from typing import Dict, Any

class Config:
    """Configuración centralizada del sistema RAG Legal"""
    
    # Modelos Ollama
    LLM_MODEL = "llama3:8b"
    EMBEDDING_MODEL = "nomic-embed-text"
    
    # Configuración de chunking
    CHUNK_SIZE_LAW = 1000
    CHUNK_OVERLAP_LAW = 100
    CHUNK_SIZE_CASES = 2000
    CHUNK_OVERLAP_CASES = 200
    
    # Rutas de archivos
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    LAW_FILE = os.path.join(DATA_DIR, "Ley_consumidor_limpio.csv")
    CASES_FILE = os.path.join(DATA_DIR, "Fallos_judiciales_ley_19.496.csv")
    
    # Configuración de retrieval
    RETRIEVAL_K = 4  # Número de documentos a recuperar
    
    # Prompts
    SYSTEM_PROMPT = """
    Eres un asistente legal especializado en la Ley 19.496 sobre Protección de los Derechos de los Consumidores de Chile.
    
    Tu función es ayudar a redactar documentos legales basándote en:
    - Artículos específicos de la ley
    - Jurisprudencia relevante
    
    Siempre debes:
    - Usar lenguaje claro y técnico
    - Citar artículos específicos
    - Referenciar casos judiciales cuando sea relevante
    - Incluir plazos legales cuando corresponda
    """
    
    DOCUMENT_PROMPT = """
    A continuación tienes dos bloques de contexto que debes usar para elaborar la respuesta:

    **Contexto legal** – Fragmentos de la Ley 19.496 sobre Protección de los Derechos de los Consumidores:
    ----------------
    {context}
    ----------------

    **Jurisprudencia relacionada** – Fallos judiciales previos que abordan situaciones similares:
    ----------------
    {context_fallo}
    ----------------

    **Historial de conversación:**  # HISTORIAL: Campo agregado para contexto
    ----------------
    {chat_history}
    ----------------

    **Pregunta actual:**
    {question}

    **Instrucciones para tu respuesta:**
    - Usa un lenguaje claro, técnico pero entendible para una persona no experta.
    - Cita explícitamente los artículos aplicables, agrupándolos según su propósito.
    - Si hay un fallo judicial que respalde el caso, menciónalo indicando el Rol, la Corte y la fecha.
    - Considera el contexto de la conversación anterior si es relevante.

    **Respuesta:**
    """
    
    CONTEXTUALIZE_PROMPT = """  # Prompt para contextualizar preguntas
    Dada la siguiente conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento 
    para que sea una pregunta independiente, en su idioma original.

    Historial de conversación:
    {chat_history}

    Pregunta de seguimiento: {question}
    
    Pregunta reformulada:
    """
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Retorna todas las configuraciones como diccionario"""
        return {
            "llm_model": cls.LLM_MODEL,
            "embedding_model": cls.EMBEDDING_MODEL,
            "chunk_size_law": cls.CHUNK_SIZE_LAW,
            "chunk_overlap_law": cls.CHUNK_OVERLAP_LAW,
            "chunk_size_cases": cls.CHUNK_SIZE_CASES,
            "chunk_overlap_cases": cls.CHUNK_OVERLAP_CASES,
            "retrieval_k": cls.RETRIEVAL_K,
            "data_dir": cls.DATA_DIR,
            "law_file": cls.LAW_FILE,
            "cases_file": cls.CASES_FILE
        }
    
    @classmethod
    def validate_files(cls) -> bool:
        """Valida que existan los archivos de datos"""
        return os.path.exists(cls.LAW_FILE) and os.path.exists(cls.CASES_FILE)