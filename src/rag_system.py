from typing import List, Dict, Any
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from .config import Config
from .data_loader import DataLoader

class RAGSystem:
    """Sistema RAG para consultas legales con dual retrieval"""
    
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        
        # Inicializar modelos
        self.llm = ChatOllama(model=self.config.LLM_MODEL)
        self.embeddings = OllamaEmbeddings(model=self.config.EMBEDDING_MODEL)
        
        # Vector stores
        self.vector_store_law = None
        self.vector_store_cases = None
        
        # Estado de inicializaci�n
        self.initialized = False
    
    def initialize(self):
        print("Inicializando sistema RAG...")
        
        # Cargar documentos
        law_docs, case_docs = self.data_loader.load_all_documents()
        
        # Crear vector stores
        print("Creando vector stores...")
        self.vector_store_law = InMemoryVectorStore(self.embeddings)
        self.vector_store_cases = InMemoryVectorStore(self.embeddings)
        
        # Agregar documentos a vector stores
        if law_docs:
            self.vector_store_law.add_documents(law_docs)
            print(f"Vector store de leyes: {len(law_docs)} documentos")
        
        if case_docs:
            self.vector_store_cases.add_documents(case_docs)
            print(f"Vector store de casos: {len(case_docs)} documentos")
        
        self.initialized = True
        print("Sistema RAG inicializado correctamente")
    
    def retrieve_law_documents(self, query: str) -> List[Document]:
        """
        Recupera documentos legales relevantes
        
        Args:
            query: Consulta del usuario
            
        Returns:
            List[Document]: Documentos relevantes
        """
        if not self.initialized:
            raise RuntimeError("Sistema no inicializado. Llama a initialize() primero")
        
        if not self.vector_store_law:
            return []
        
        return self.vector_store_law.similarity_search(
            query, 
            k=self.config.RETRIEVAL_K
        )
    
    def retrieve_case_documents(self, query: str) -> List[Document]:
        """
        Recupera fallos judiciales relevantes
        
        Args:
            query: Consulta del usuario
            
        Returns:
            List[Document]: Fallos relevantes
        """
        if not self.initialized:
            raise RuntimeError("Sistema no inicializado. Llama a initialize() primero")
        
        if not self.vector_store_cases:
            return []
        
        return self.vector_store_cases.similarity_search(
            query,
            k=self.config.RETRIEVAL_K
        )
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """
        Elimina documentos duplicados por contenido
        
        Args:
            documents: Lista de documentos
            
        Returns:
            List[Document]: Documentos sin duplicados
        """
        unique_docs = []
        seen_contents = set()
        
        for doc in documents:
            if doc.page_content not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(doc.page_content)
        
        return unique_docs
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Formatea documentos en contexto para el prompt
        
        Args:
            documents: Lista de documentos
            
        Returns:
            str: Contexto formateado
        """
        if not documents:
            return "No se encontró información relevante."
        
        unique_docs = self._deduplicate_documents(documents)
        return "\n\n".join(doc.page_content for doc in unique_docs)
    
    def _extract_sources(self, documents: List[Document]) -> Dict[str, List[str]]:
        """
        Extrae fuentes de los documentos
        
        Args:
            documents: Lista de documentos
            
        Returns:
            Dict: Fuentes organizadas por tipo
        """
        sources = {
            "articulos": set(),
            "casos": set()
        }
        
        unique_docs = self._deduplicate_documents(documents)
        
        for doc in unique_docs:
            metadata = doc.metadata
            
            if metadata.get('tipo') == 'ley':
                articulo = metadata.get('Articulo')
                if articulo:
                    sources["articulos"].add(articulo)
            
            elif metadata.get('tipo') == 'fallo':
                rol = metadata.get('Rol', 'N/A')
                corte = metadata.get('Corte_origen', 'N/A')
                fecha = metadata.get('Fecha_Sentencia', 'N/A')
                caso = f"Rol: {rol}, Corte: {corte}, Fecha: {fecha}"
                sources["casos"].add(caso)
        
        return {
            "articulos": sorted(list(sources["articulos"])),
            "casos": sorted(list(sources["casos"]))
        }
    
    def generate_response(self, query: str, chat_history: str = "") -> Dict[str, Any]:
        """
        Genera respuesta usando RAG
        
        Args:
            query: Consulta del usuario
            chat_history: Historial de conversación
            
        Returns:
            Dict: Respuesta con contexto y fuentes
        """
        if not self.initialized:
            raise RuntimeError("Sistema no inicializado. Llama a initialize() primero")
        
        # Recuperar documentos
        law_docs = self.retrieve_law_documents(query)
        case_docs = self.retrieve_case_documents(query)
        
        # Formatear contextos
        law_context = self._format_context(law_docs)
        case_context = self._format_context(case_docs)
        
        # Extraer fuentes
        all_docs = law_docs + case_docs
        sources = self._extract_sources(all_docs)
        
        # Crear prompt
        prompt = ChatPromptTemplate.from_template(self.config.DOCUMENT_PROMPT)
        
        # Generar respuesta
        formatted_prompt = prompt.format(
            context=law_context,
            context_fallo=case_context,
            chat_history=chat_history,
            question=query
        )
        
        response = self.llm.invoke(formatted_prompt)
        
        return {
            "answer": response.content.strip(),
            "sources": sources,
            "law_context": law_context,
            "case_context": case_context
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado del sistema
        
        Returns:
            Dict: Estado del sistema
        """
        return {
            "initialized": self.initialized,
            "law_docs_count": len(self.vector_store_law.docstore._store) if self.vector_store_law else 0,
            "case_docs_count": len(self.vector_store_cases.docstore._store) if self.vector_store_cases else 0,
            "config": self.config.get_config()
        }