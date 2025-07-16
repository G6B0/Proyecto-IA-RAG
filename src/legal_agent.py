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

        print(f"\nProcesando consulta (fase {self.phase}): {query}")

        # Comando para ejecutar RAG
        if query.lower().strip() == "/finalizar":
            self.phase = 3
            return self._execute_phase_3()
        
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
            "answer": self._format_answer_with_sources(
                response["messages"][-1].content,
                response["sources"]
            ),
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