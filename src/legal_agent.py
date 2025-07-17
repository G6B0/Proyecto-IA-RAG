from typing import List, Dict, Any, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated, TypedDict
from .config import Config
from .rag_system import RAGSystem
import uuid

class ConversationState(TypedDict):
    """Estado de la conversación para LangGraph"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    contextualized_query: str
    original_query: str
    sources: Dict[str, List[str]]

class LegalAgent:
    def __init__(self):
        self.config = Config()
        self.rag_system = RAGSystem()
        self.app = None
        self.memory = MemorySaver()
        self.current_thread_id = None
        self.session_initialized = False

        self.phase = 1  # 1: recolectar datos, 3: ejecutar RAG

    def initialize(self):
        print("Inicializando agente legal...")
        self.rag_system.initialize()
        self._build_graph()
        self.session_initialized = True
        self.current_thread_id = str(uuid.uuid4())
        print("Agente legal listo para usar")

    def _build_graph(self):
        workflow = StateGraph(state_schema=ConversationState)

        def process_query(state: ConversationState) -> ConversationState:
            messages = state["messages"]
            current_query = messages[-1].content if messages else ""

            contextualized_query = self._contextualize_question(current_query, messages[:-1])
            rag_response = self.rag_system.generate_response(
                contextualized_query,
                self._format_message_history(messages)
            )
            formatted_answer = self._format_answer_with_sources(
                rag_response["answer"],
                rag_response["sources"]
            )
            ai_message = AIMessage(content=formatted_answer)

            return {
                "messages": [ai_message],
                "contextualized_query": contextualized_query,
                "original_query": current_query,
                "sources": rag_response["sources"]
            }

        workflow.add_edge(START, "process_query")
        workflow.add_node("process_query", process_query)

        self.app = workflow.compile(checkpointer=self.memory)

    def chat(self, query: str) -> Dict[str, Any]:
        if not self.session_initialized:
            raise RuntimeError("Agente no inicializado. Llama a initialize() primero")

        # Comando para ejecutar RAG
        if query.lower().strip() == "/finalizar":
            self.phase = 3
            return self._execute_phase_3()
        
        # Detectar automáticamente el tipo de consulta
        query_type = self._classify_query(query)
        
        if query_type == "direct":
            print(f"\nProcesando consulta directa: {query}")
            return self._process_direct_query(query)
        else:
            print(f"\nProcesando consulta compleja (fase {self.phase}): {query}")
        
        if self.phase == 1:
            # Guardar directamente el mensaje en el historial del grafo
            # Obtener estado actual
            config = {"configurable": {"thread_id": self.current_thread_id}}
            state = self.app.get_state(config)

            # Agregar el nuevo mensaje humano al historial actual
            current_messages = list(state.values.get("messages", []))
            current_messages.append(HumanMessage(content=query))

            # Actualizar estado con los mensajes nuevos
            new_state = {
                "messages": current_messages,
                "contextualized_query": "",
                "original_query": "",
                "sources": {"articulos": [], "casos": []}
            }
            self.app.update_state(config, new_state)

            return {
                "answer": "Gracias por la información. Sigue contándome o escribe '/finalizar' para que prepare la respuesta."
            }

    def _execute_phase_3(self):
        config = {"configurable": {"thread_id": self.current_thread_id}}
        state = self.app.get_state(config)
        messages = state.values.get("messages", [])

        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        concatenated_text = " ".join(m.content for m in human_messages)

        contextualized = self._contextualize_question(concatenated_text, human_messages)
        human_message = HumanMessage(content=contextualized)

        response = self.app.invoke({"messages": [human_message]}, config=config)

        self.app.update_state(config, {
            "messages": [],
            "contextualized_query": "",
            "original_query": "",
            "sources": {"articulos": [], "casos": []}
        })

        self.phase = 1

        return {
            "answer": response["messages"][-1].content,
            "sources": response["sources"],
            "contextualized_query": response["contextualized_query"],
            "original_query": response["original_query"],
        }





    def _contextualize_question(self, query: str, chat_history: List[BaseMessage]) -> str:
        """
        Contextualiza la pregunta actual basándose en el historial

        Args:
            query: Pregunta actual
            chat_history: Historial de mensajes

        Returns:
            str: Pregunta contextualizada
       """
        
        if not chat_history:
            return query

        try:
            from langchain_core.prompts import ChatPromptTemplate

            formatted_history = self._format_message_history(chat_history)

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

            messages = contextualize_prompt.format_messages(
                chat_history=formatted_history,
                question=query
            )

            response = self.rag_system.llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            print(f"Error contextualizando pregunta: {e}")
            return query

    
    def _format_message_history(self, messages: List[BaseMessage]) -> str:
        """
        Formatea el historial de mensajes para el sistema RAG
        
        Args:
            messages: Lista de mensajes
            
        Returns:
            str: Historial formateado
        """
        formatted_history = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_history += f"Usuario: {message.content}\n"
            elif isinstance(message, AIMessage):
                formatted_history += f"Asistente: {message.content}\n"
        return formatted_history
    
    def _format_answer_with_sources(self, answer: str, sources: Dict[str, List[str]]) -> str:
        """
        Formatea la respuesta incluyendo las fuentes
        
        Args:
            answer: Respuesta del modelo
            sources: Fuentes utilizadas
            
        Returns:
            str: Respuesta formateada con fuentes
        """
        formatted_answer = answer
        
        # Añadir fuentes de artículos
        if sources.get("articulos"):
            formatted_answer += "\n\n**Artículos utilizados como fuente:**\n"
            for articulo in sources["articulos"]:
                formatted_answer += f"- {articulo}\n"
        
        # Añadir fuentes de casos
        if sources.get("casos"):
            formatted_answer += "\n**Casos judiciales utilizados como fuente:**\n"
            for caso in sources["casos"]:
                formatted_answer += f"- {caso}\n"
        
        return formatted_answer
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Obtiene el historial de conversación
        
        Returns:
            List[Dict]: Historial de mensajes
        """
        if not self.session_initialized:
            return []
        
        try:
            config = {"configurable": {"thread_id": self.current_thread_id}}
            # Obtener el estado actual del grafo
            state = self.app.get_state(config)
            
            history = []
            for message in state.values.get("messages", []):
                if isinstance(message, HumanMessage):
                    history.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    history.append({"role": "assistant", "content": message.content})
            
            return history
        except Exception as e:
            print(f"Error obteniendo historial: {e}")
            return []
    
    def clear_history(self):
        """Limpia el historial de conversación"""
        if self.session_initialized:
            # Crear un nuevo thread_id para empezar de cero
            self.current_thread_id = str(uuid.uuid4())
            print("Historial limpiado")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado del agente
        
        Returns:
            Dict: Estado del agente
        """
        return {
            "initialized": self.session_initialized,
            "thread_id": self.current_thread_id,
            "rag_system_status": self.rag_system.get_status(),
            "messages_count": len(self.get_history())
        }
    
    def save_conversation(self, filename: str):
        """
        Guarda la conversación en un archivo
        
        Args:
            filename: Nombre del archivo
        """
        try:
            import json
            from datetime import datetime
            
            conversation_data = {
                "timestamp": datetime.now().isoformat(),
                "thread_id": self.current_thread_id,
                "conversation": self.get_history()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            print(f"Conversación guardada en {filename}")
            
        except Exception as e:
            print(f"Error guardando conversación: {e}")
    
    def load_conversation(self, filename: str):
        """
        Carga una conversación desde un archivo
        
        Args:
            filename: Nombre del archivo
        """
        try:
            import json
            
            with open(filename, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            # Crear nuevo thread_id o usar el guardado
            self.current_thread_id = conversation_data.get("thread_id", str(uuid.uuid4()))
            
            # Reconstruir historial en el grafo
            config = {"configurable": {"thread_id": self.current_thread_id}}
            
            messages = []
            for message in conversation_data.get("conversation", []):
                if message["role"] == "user":
                    messages.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    messages.append(AIMessage(content=message["content"]))
            
            # Restaurar estado en el grafo
            if messages:
                # Crear un estado inicial con todos los mensajes
                initial_state = {
                    "messages": messages,
                    "contextualized_query": "",
                    "original_query": "",
                    "sources": {"articulos": [], "casos": []}
                }
                
                # Actualizar el estado del grafo
                self.app.update_state(config, initial_state)
            
            print(f"Conversación cargada desde {filename}")
            print(f"Mensajes cargados: {len(messages)}")
            
        except Exception as e:
            print(f"Error cargando conversación: {e}")
    
    def set_thread_id(self, thread_id: str):
        """
        Establece un thread_id específico para la conversación
        
        Args:
            thread_id: ID del thread
        """
        self.current_thread_id = thread_id
        print(f"Thread ID establecido: {thread_id}")
    
    def get_thread_id(self) -> str:
        """
        Obtiene el thread_id actual
        
        Returns:
            str: Thread ID actual
        """
        return self.current_thread_id
    
    def _classify_query(self, query: str) -> str:
        """
        Clasifica automáticamente el tipo de consulta
        
        Args:
            query: Consulta del usuario
            
        Returns:
            str: "direct" para consultas directas, "complex" para consultas complejas
        """
        import unicodedata
        
        # Normalizar texto: quitar acentos y convertir a minúsculas
        def normalize_text(text):
            # Remover acentos
            text = unicodedata.normalize('NFD', text)
            text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
            return text.lower().strip()
        
        query_normalized = normalize_text(query)
        
        # Patrones para consultas directas (preguntas específicas sobre la ley)
        direct_patterns = [
            "que derechos tengo",      # Sin acento
            "cuanto tiempo tengo",     # Sin acento
            "que dice el articulo",    # Sin acento
            "puedo devolver",
            "puedo reclamar",
            "es legal",
            "esta permitido",          # Sin acento
            "cual es el plazo",        # Sin acento
            "que establece",           # Sin acento
            "como funciona",           # Sin acento
            "donde dice",              # Sin acento
            "que pasa si",             # Sin acento
            "tengo derecho a",
            "puede la empresa",
            "debe la empresa",
            "cuales son mis derechos", # Sin acento
            "que opciones tengo",      # Sin acento
            "es obligatorio",
            "derechos tengo",          # Patrón más amplio
            "tiempo tengo para",       # Patrón más amplio
            "puedo hacer",
            "debo hacer",
            "derecho a"
        ]
        
        # Patrones para consultas complejas (redacción de documentos/casos específicos)
        complex_patterns = [
            "quiero redactar",
            "necesito ayuda para",
            "me paso",              # Sin acento
            "me vendieron",
            "compre",               # Sin acento
            "contrate",             # Sin acento
            "la empresa me",
            "el vendedor",
            "quiero demandar",
            "quiero hacer una denuncia",
            "necesito hacer un reclamo",
            "me estafaron",
            "me cobraron",
            "no me devolvieron",
            "quiero denunciar"
        ]
        
        # Verificar patrones de consultas directas primero (más específicos)
        for pattern in direct_patterns:
            if pattern in query_normalized:
                return "direct"
        
        # Verificar patrones de consultas complejas
        complex_patterns_normalized = [normalize_text(pattern) for pattern in complex_patterns]
        for pattern in complex_patterns_normalized:
            if pattern in query_normalized:
                return "complex"
        
        # Si empieza con interrogación o contiene palabras clave de pregunta, probablemente es directa
        question_words = ["que", "cual", "cuando", "como", "donde", "por que", "puedo", "debo"]  # Sin acentos
        if query_normalized.startswith("¿") or query.startswith("¿") or any(word in query_normalized.split()[:3] for word in question_words):
            return "direct"
        
        # Por defecto, tratar como compleja para mantener el flujo actual
        return "complex"
    
    def _process_direct_query(self, query: str) -> Dict[str, Any]:
        """
        Procesa directamente una consulta sin usar el sistema de fases
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Dict: Respuesta directa del RAG
        """
        try:
            # Obtener historial actual para contexto
            config = {"configurable": {"thread_id": self.current_thread_id}}
            state = self.app.get_state(config)
            messages = list(state.values.get("messages", []))
            
            # Agregar la nueva consulta al historial
            human_message = HumanMessage(content=query)
            messages.append(human_message)
            
            # Contextualizar la consulta
            contextualized_query = self._contextualize_question(query, messages[:-1])
            
            # Generar respuesta usando RAG
            rag_response = self.rag_system.generate_response(
                contextualized_query,
                self._format_message_history(messages[:-1])
            )
            
            # Crear respuesta AI y actualizarla en el estado
            ai_message = AIMessage(content=rag_response["answer"])
            messages.append(ai_message)
            
            # Actualizar estado con la conversación completa
            updated_state = {
                "messages": messages,
                "contextualized_query": contextualized_query,
                "original_query": query,
                "sources": rag_response["sources"]
            }
            self.app.update_state(config, updated_state)
            
            return {
                "answer": rag_response["answer"],
                "sources": rag_response["sources"],
                "contextualized_query": contextualized_query,
                "original_query": query
            }
            
        except Exception as e:
            print(f"Error procesando consulta directa: {e}")
            return {
                "answer": "Lo siento, ocurrió un error al procesar tu consulta. Por favor, intenta nuevamente.",
                "sources": {"articulos": [], "casos": []},
                "contextualized_query": query,
                "original_query": query
            }