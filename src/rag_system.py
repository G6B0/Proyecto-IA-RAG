from typing import List, Dict, Any
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from .config import Config
from .data_loader import DataLoader
import os

class RAGSystem:
    """Sistema RAG para consultas legales con dual retrieval y soporte para historial"""
    
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader()
        
        # Inicializar modelos
        self.llm = ChatOllama(model=self.config.LLM_MODEL)
        self.embeddings = OllamaEmbeddings(model=self.config.EMBEDDING_MODEL)
        
        # Vector stores
        self.vector_store_law = None
        self.vector_store_cases = None
        
        # Estado de inicialización
        self.initialized = False
        
        # Prompt template para respuestas con historial
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Crea el template de prompt que incluye historial"""
        return ChatPromptTemplate.from_messages([
            ("system", """Eres un asistente legal especializado en derecho chileno. 
            Utiliza la información proporcionada para responder las consultas de manera precisa y profesional.
            
            INSTRUCCIONES:
            1. Analiza tanto el contexto legal como el historial de conversación
            2. Proporciona respuestas basadas en las fuentes proporcionadas
            3. Mantén coherencia con las respuestas anteriores
            4. Si la consulta actual se relaciona con temas anteriores, hazlo explícito
            5. Cita las fuentes específicas cuando sea relevante
            
            CONTEXTO LEGAL:
            {context}
            
            HISTORIAL DE CONVERSACIÓN:
            {chat_history}"""),
            ("human", "{question}")
        ])
    
    def initialize(self):
        """Inicializa el sistema cargando datos y creando vector stores"""
        print("Inicializando sistema RAG...")

        law_path = "./chroma_db"
        cases_path = "./chroma_db"

        os.makedirs(law_path, exist_ok=True)

        db_exists = os.path.exists(os.path.join(law_path, "chroma.sqlite3"))

        if db_exists:
            print("Bases vectoriales existentes detectadas. Cargando desde disco...")

            self.vector_store_law = Chroma(
                collection_name="leyes_collection",
                embedding_function=self.embeddings,
                persist_directory=law_path
            )
            self.vector_store_cases = Chroma(
                collection_name="fallos_collection",
                embedding_function=self.embeddings,
                persist_directory=cases_path
            )
        else:
            print("Bases vectoriales no encontradas. Procesando documentos y creando nuevas...")
            law_docs, case_docs = self.data_loader.load_all_documents(case_batch_size=100)

            self.vector_store_law = Chroma(
                collection_name="leyes_collection",
                embedding_function=self.embeddings,
                persist_directory=law_path
            )
            if law_docs:
                self.vector_store_law.add_documents(law_docs)
                print(f"Vector store de leyes: {len(law_docs)} documentos")

            self.vector_store_cases = Chroma(
                collection_name="fallos_collection",
                embedding_function=self.embeddings,
                persist_directory=cases_path
            )
            if case_docs:
                # Agregar documentos por lotes
                batch_size = 5000  # Menor que el límite de 5461
                total_docs = len(case_docs)
    
                print(f"Agregando {total_docs} documentos al vector store en lotes de {batch_size}")
    
                for i in range(0, total_docs, batch_size):
                    batch_end = min(i + batch_size, total_docs)
                    batch = case_docs[i:batch_end]
        
                    batch_num = (i // batch_size) + 1
                    total_batches = (total_docs + batch_size - 1) // batch_size
        
                    print(f"Procesando lote {batch_num}/{total_batches}: documentos {i} a {batch_end}")
        
                    try:
                        self.vector_store_cases.add_documents(batch)
                        print(f"Lote {batch_num} agregado exitosamente")
                    except Exception as e:
                        print(f"Error agregando lote {batch_num}: {e}")
                        # Si falla, intenta con un batch más pequeño
                        if batch_size > 1000:
                            smaller_batch_size = 1000
                            for j in range(0, len(batch), smaller_batch_size):
                                smaller_batch = batch[j:j+smaller_batch_size]
                                self.vector_store_cases.add_documents(smaller_batch)
                        else:
                            raise e
    
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
    
    def generate_response(self, query: str, chat_history: str = "") -> Dict[str, Any]:
        """
        Genera una respuesta basada en RAG considerando el historial
        
        Args:
            query: Consulta actual del usuario
            chat_history: Historial de conversación formateado
            
        Returns:
            Dict: Respuesta con contexto y fuentes
        """
        if not self.initialized:
            raise RuntimeError("Sistema no inicializado. Llama a initialize() primero")
        
        # Recuperar documentos relevantes
        law_docs = self.retrieve_law_documents(query)
        case_docs = self.retrieve_case_documents(query)
        
        # Combinar documentos
        all_docs = law_docs + case_docs
        
        # Formatear contexto
        context = self._format_context(all_docs)
        
        # Extraer fuentes
        sources = self._extract_sources(all_docs)
        
        # Generar respuesta usando el LLM
        try:
            # Crear el prompt completo
            messages = self.prompt_template.format_messages(
                context=context,
                chat_history=chat_history,
                question=query
            )
            
            # Invocar el modelo
            response = self.llm.invoke(messages)
            
            return {
                "answer": response.content,
                "sources": sources,
                "context": context,
                "retrieved_docs": len(all_docs)
            }
            
        except Exception as e:
            print(f"Error generando respuesta: {e}")
            return {
                "answer": "Lo siento, ocurrió un error al procesar tu consulta.",
                "sources": sources,
                "context": context,
                "retrieved_docs": len(all_docs)
            }
    
    def generate_response_with_messages(self, query: str, message_history: List[BaseMessage]) -> Dict[str, Any]:
        """
        Genera respuesta considerando el historial de mensajes de LangChain
        
        Args:
            query: Consulta actual
            message_history: Lista de mensajes BaseMessage
            
        Returns:
            Dict: Respuesta con contexto y fuentes
        """
        # Convertir mensajes a formato string
        formatted_history = self._format_message_history(message_history)
        
        # Usar el método principal
        return self.generate_response(query, formatted_history)
    
    def _format_message_history(self, messages: List[BaseMessage]) -> str:
        """
        Formatea el historial de mensajes para el prompt
        
        Args:
            messages: Lista de mensajes BaseMessage
            
        Returns:
            str: Historial formateado
        """
        if not messages:
            return "No hay historial de conversación anterior."
        
        formatted_history = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_history += f"Usuario: {message.content}\n"
            elif isinstance(message, AIMessage):
                formatted_history += f"Asistente: {message.content}\n"
        
        return formatted_history
    
    def contextualize_query(self, current_query: str, chat_history: str) -> str:
        """
        Contextualiza la consulta actual basándose en el historial
        
        Args:
            current_query: Consulta actual
            chat_history: Historial de conversación
            
        Returns:
            str: Consulta contextualizada
        """
        if not chat_history.strip():
            return current_query
        
        # Prompt para contextualización
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", """Dado el historial de conversación y la pregunta más reciente del usuario, 
            reformula la pregunta para que sea independiente y pueda entenderse sin el historial.
            
            REGLAS:
            1. NO respondas la pregunta, solo reformúlala
            2. Incluye el contexto necesario del historial en la nueva pregunta
            3. Mantén la intención original del usuario
            4. Si la pregunta ya es independiente, devuélvela tal como está
            
            HISTORIAL:
            {chat_history}"""),
            ("human", "Pregunta actual: {question}")
        ])
        
        try:
            messages = contextualize_prompt.format_messages(
                chat_history=chat_history,
                question=current_query
            )
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            print(f"Error contextualizando consulta: {e}")
            return current_query
    
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
        
        # Separar por tipo de documento
        law_docs = [doc for doc in unique_docs if doc.metadata.get('tipo') == 'ley']
        case_docs = [doc for doc in unique_docs if doc.metadata.get('tipo') == 'fallo']
        
        context = ""
        
        if law_docs:
            context += "=== LEGISLACIÓN RELEVANTE ===\n"
            for doc in law_docs:
                context += f"{doc.page_content}\n\n"
        
        if case_docs:
            context += "=== JURISPRUDENCIA RELEVANTE ===\n"
            for doc in case_docs:
                context += f"{doc.page_content}\n\n"
        
        return context
    
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
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado del sistema
        
        Returns:
            Dict: Estado del sistema
        """
        return {
            "initialized": self.initialized,
            "law_docs_count": self.vector_store_law._collection.count() if self.vector_store_law else 0,
            "case_docs_count": self.vector_store_cases._collection.count() if self.vector_store_cases else 0,
            "config": self.config.get_config()
        }