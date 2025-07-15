from typing import List, Dict, Any, Sequence, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated, TypedDict
from .config import Config
from .rag_system import RAGSystem
from .claim_system import ClaimIntentDetector, ConversationalDataCollector, ClaimData, ClaimStatus
from .jurisprudence_matcher import JurisprudenceMatcher
from .formal_claim_generator import FormalClaimGenerator
import uuid

class ConversationState(TypedDict):
    """Estado de la conversación para LangGraph con soporte para reclamos"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    contextualized_query: str
    original_query: str
    sources: Dict[str, List[str]]
    # Campos para el sistema de reclamos
    claim_mode: bool
    claim_status: str
    claim_data: Dict[str, Any]
    pending_question: Optional[str]

class LegalAgent:
    """Agente legal con capacidad de mantener historial de conversación y sistema de reclamos"""
    
    def __init__(self):
        self.config = Config()
        self.rag_system = RAGSystem()
        self.app = None
        self.memory = MemorySaver()
        self.current_thread_id = None
        self.session_initialized = False
        
        # Componentes del sistema de reclamos
        self.claim_detector = ClaimIntentDetector()
        self.data_collector = ConversationalDataCollector()
        self.jurisprudence_matcher = None  # Se inicializa después del RAG
        self.claim_generator = None  # Se inicializa después del RAG
    
    def initialize(self):
        """Inicializa el agente y el sistema RAG"""
        print("Inicializando agente legal...")
        self.rag_system.initialize()
        
        # Inicializar componentes que dependen del RAG
        self.jurisprudence_matcher = JurisprudenceMatcher(self.rag_system)
        self.claim_generator = FormalClaimGenerator(self.rag_system)
        
        self._build_graph()
        self.session_initialized = True
        # Crear un thread_id único para esta sesión
        self.current_thread_id = str(uuid.uuid4())
        print("Agente legal listo para usar")

    def _build_graph(self):
        """Construye el grafo LangGraph para el agente legal"""
        workflow = StateGraph(state_schema=ConversationState)

        def process_query(state: ConversationState) -> ConversationState:
            messages = state["messages"]
            current_query = messages[-1].content if messages else ""
            
            # Inicializar campos si no existen
            claim_mode = state.get("claim_mode", False)
            claim_status = state.get("claim_status", ClaimStatus.NOT_STARTED.value)
            claim_data_dict = state.get("claim_data", {})
            pending_question = state.get("pending_question", None)
            
            # Crear objeto ClaimData desde el diccionario
            claim_data = self._dict_to_claim_data(claim_data_dict)
            
            # Verificar si necesitamos activar el modo reclamo
            if not claim_mode:
                intent_result = self.claim_detector.detect_claim_intent(current_query)
                if intent_result["has_claim_intent"]:
                    claim_mode = True
                    claim_status = ClaimStatus.COLLECTING_DATA.value
                    
                    # Inicializar datos con información detectada
                    if "initial_data" in intent_result:
                        self._populate_initial_data(claim_data, intent_result["initial_data"])
                    
                    # Generar respuesta de activación
                    response = self._generate_claim_activation_response(intent_result, claim_data)
                    
                    # Obtener próxima pregunta
                    next_question = self.data_collector.get_next_question(claim_data)
                    
                    if next_question:
                        response += f"\n\n{next_question}"
                        pending_question = next_question
                    
                    ai_message = AIMessage(content=response)
                    
                    return {
                        "messages": [ai_message],
                        "contextualized_query": current_query,
                        "original_query": current_query,
                        "sources": {"articulos": [], "casos": []},
                        "claim_mode": claim_mode,
                        "claim_status": claim_status,
                        "claim_data": claim_data.to_dict(),
                        "pending_question": pending_question
                    }
            
            # Si estamos en modo reclamo, procesar la recolección de datos
            if claim_mode and claim_status == ClaimStatus.COLLECTING_DATA.value:
                # Intentar extraer datos de la respuesta
                data_extracted = self.data_collector.extract_data_from_response(current_query, claim_data)
                
                # Verificar si tenemos suficientes datos
                if self.data_collector.is_data_sufficient(claim_data):
                    claim_status = ClaimStatus.DATA_COMPLETE.value
                    response = self._generate_data_complete_response(claim_data)
                else:
                    # Obtener próxima pregunta
                    next_question = self.data_collector.get_next_question(claim_data)
                    
                    if next_question:
                        if data_extracted:
                            response = "Perfecto, entiendo. " + next_question
                        else:
                            response = "Necesito un poco más de información. " + next_question
                        
                        pending_question = next_question
                    else:
                        response = "Gracias por la información. ¿Hay algo más que puedas agregar sobre tu caso?"
                        pending_question = None
                
                ai_message = AIMessage(content=response)
                
                return {
                    "messages": [ai_message],
                    "contextualized_query": current_query,
                    "original_query": current_query,
                    "sources": {"articulos": [], "casos": []},
                    "claim_mode": claim_mode,
                    "claim_status": claim_status,
                    "claim_data": claim_data.to_dict(),
                    "pending_question": pending_question
                }
            
            # Si los datos están completos, procesar solicitud de generación
            if claim_mode and claim_status == ClaimStatus.DATA_COMPLETE.value:
                if "genera" in current_query.lower() or "reclamo" in current_query.lower():
                    claim_status = ClaimStatus.GENERATING_CLAIM.value
                    
                    # Generar reclamo formal
                    response = self._generate_formal_claim(claim_data)
                    
                    claim_status = ClaimStatus.CLAIM_READY.value
                    
                    ai_message = AIMessage(content=response)
                    
                    return {
                        "messages": [ai_message],
                        "contextualized_query": current_query,
                        "original_query": current_query,
                        "sources": {"articulos": [], "casos": []},
                        "claim_mode": claim_mode,
                        "claim_status": claim_status,
                        "claim_data": claim_data.to_dict(),
                        "pending_question": None
                    }
            
            # Procesamiento normal de consultas (modo no-reclamo o consultas adicionales)
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
                "sources": rag_response["sources"],
                "claim_mode": claim_mode,
                "claim_status": claim_status,
                "claim_data": claim_data.to_dict(),
                "pending_question": pending_question
            }
        
        workflow.add_edge(START, "process_query")
        workflow.add_node("process_query", process_query)

        self.app = workflow.compile(checkpointer=self.memory)
    
    def _dict_to_claim_data(self, claim_dict: Dict[str, Any]) -> ClaimData:
        """Convierte un diccionario a objeto ClaimData"""
        if not claim_dict:
            return ClaimData()
        
        return ClaimData(
            product_type=claim_dict.get("product_type"),
            product_name=claim_dict.get("product_name"),
            problem_description=claim_dict.get("problem_description"),
            purchase_date=claim_dict.get("purchase_date"),
            purchase_place=claim_dict.get("purchase_place"),
            purchase_amount=claim_dict.get("purchase_amount"),
            consumer_name=claim_dict.get("consumer_name"),
            consumer_rut=claim_dict.get("consumer_rut"),
            consumer_address=claim_dict.get("consumer_address"),
            consumer_phone=claim_dict.get("consumer_phone"),
            consumer_email=claim_dict.get("consumer_email"),
            incident_date=claim_dict.get("incident_date"),
            previous_attempts=claim_dict.get("previous_attempts", []),
            desired_solution=claim_dict.get("desired_solution"),
            warranty_info=claim_dict.get("warranty_info"),
            receipts_available=claim_dict.get("receipts_available")
        )
    
    def _populate_initial_data(self, claim_data: ClaimData, initial_data: Dict[str, Any]):
        """Popula datos iniciales detectados automáticamente"""
        if "potential_products" in initial_data:
            claim_data.product_type = initial_data["potential_products"][0]
        
        if "potential_stores" in initial_data:
            claim_data.purchase_place = initial_data["potential_stores"][0]
        
        if "potential_dates" in initial_data:
            claim_data.purchase_date = str(initial_data["potential_dates"][0])
    
    def _generate_claim_activation_response(self, intent_result: Dict, claim_data: ClaimData) -> str:
        """Genera respuesta cuando se activa el modo reclamo"""
        response = f"""**Detecté que quieres hacer un reclamo**

Te voy a ayudar a recopilar toda la información necesaria para generar un reclamo formal basado en la Ley del Consumidor chilena y jurisprudencia relevante.

Confianza de detección: {intent_result['confidence_score']:.0f}%"""
        
        # Mostrar datos ya detectados
        detected_data = []
        if claim_data.product_type:
            detected_data.append(f"- Producto: {claim_data.product_type}")
        
        if claim_data.purchase_place:
            detected_data.append(f"- Lugar de compra: {claim_data.purchase_place}")
        
        if claim_data.purchase_date:
            detected_data.append(f"- Fecha: {claim_data.purchase_date}")
        
        if detected_data:
            response += "\n\n**Información detectada:**\n" + "\n".join(detected_data)
        
        return response
    
    def _generate_data_complete_response(self, claim_data: ClaimData) -> str:
        """Genera respuesta cuando los datos están completos"""
        completion = claim_data.get_completion_percentage()
        
        response = f"""**Información recopilada exitosamente**

**Datos del reclamo** (completitud: {completion:.0f}%):
- **Producto:** {claim_data.product_type}
- **Problema:** {claim_data.problem_description}
- **Fecha de compra:** {claim_data.purchase_date}
- **Lugar de compra:** {claim_data.purchase_place}"""

        if claim_data.consumer_name:
            response += f"\n- **Consumidor:** {claim_data.consumer_name}"
        if claim_data.consumer_rut:
            response += f"\n- **RUT:** {claim_data.consumer_rut}"
        
        response += f"""

**Próximo paso:**
Ahora voy a buscar jurisprudencia similar y generar tu reclamo formal. Esto incluirá:
- Búsqueda de casos jurisprudenciales similares
- Aplicación de artículos relevantes de la Ley 19.496
- Generación de documento formal para SERNAC

Escribe "**genera el reclamo**" para proceder."""
        
        return response
    
    def _generate_formal_claim(self, claim_data: ClaimData) -> str:
        """Genera el reclamo formal completo"""
        response = "**Generando tu reclamo formal...**\n\n"
        
        try:
            # Buscar casos similares
            response += "**Paso 1:** Buscando jurisprudencia similar...\n"
            similar_cases = self.jurisprudence_matcher.find_similar_cases(claim_data)
            
            if similar_cases:
                response += f"Encontré {len(similar_cases)} casos jurisprudenciales similares:\n\n"
                for i, case in enumerate(similar_cases[:3], 1):
                    response += f"{i}. **{case.document.metadata.get('Rol', 'N/A')}** (Similitud: {case.similarity_score:.1%})\n"
                    response += f"   - Resultado: {case.outcome}\n"
                    response += f"   - {case.case_summary[:100]}...\n\n"
            else:
                response += "No se encontraron casos exactamente similares, pero se aplicará la normativa legal general.\n\n"
            
            response += "**Paso 2:** Generando documento formal...\n\n"
            
            # Generar reclamo formal
            formal_claim = self.claim_generator.generate_formal_claim(claim_data, similar_cases)
            
            response += "**Reclamo formal generado exitosamente**\n\n"
            response += "---\n\n"
            response += "**DOCUMENTO FORMAL COMPLETO:**\n\n"
            response += formal_claim.full_document
            
            response += "\n\n---\n\n"
            response += "**Información importante:**\n"
            response += "- Revisa el documento completo antes de presentarlo\n"
            response += "- Completa cualquier información faltante entre [corchetes]\n"
            response += "- Adjunta todos los documentos de respaldo (boletas, contratos, etc.)\n"
            response += "- Presenta el reclamo en SERNAC o en línea en www.sernac.cl"
            
        except Exception as e:
            response += f"Error generando el reclamo: {e}\n"
            response += "Por favor, intenta nuevamente o contacta al soporte."
        
        return response
    
    def chat(self, query: str) -> Dict[str, Any]:
        """
        Procesa una consulta del usuario manteniendo el contexto
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Dict: Respuesta completa con contexto
        """
        if not self.session_initialized:
            raise RuntimeError("Agente no inicializado. Llama a initialize() primero")
        
        print(f"\nProcesando consulta: {query}")
        
        # Crear mensaje humano
        human_message = HumanMessage(content=query)
        
        # Configuración del thread para persistencia
        config = {"configurable": {"thread_id": self.current_thread_id}}
        
        # Invocar el grafo con el mensaje
        response = self.app.invoke(
            {"messages": [human_message]},
            config=config
        )
        
        return {
            "answer": response["messages"][-1].content,
            "sources": response.get("sources", {"articulos": [], "casos": []}),
            "contextualized_query": response.get("contextualized_query", query),
            "original_query": response.get("original_query", query),
            "claim_mode": response.get("claim_mode", False),
            "claim_status": response.get("claim_status", ClaimStatus.NOT_STARTED.value),
            "claim_data": response.get("claim_data", {}),
            "pending_question": response.get("pending_question", None)
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
        base_status = {
            "initialized": self.session_initialized,
            "thread_id": self.current_thread_id,
            "rag_system_status": self.rag_system.get_status(),
            "messages_count": len(self.get_history())
        }
        
        # Agregar estado de reclamos
        claim_status = self.get_claim_status()
        base_status.update(claim_status)
        
        return base_status
    
    def get_claim_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema de reclamos"""
        if not self.session_initialized:
            return {"claim_mode": False, "claim_status": ClaimStatus.NOT_STARTED.value}
        
        try:
            config = {"configurable": {"thread_id": self.current_thread_id}}
            state = self.app.get_state(config)
            
            return {
                "claim_mode": state.values.get("claim_mode", False),
                "claim_status": state.values.get("claim_status", ClaimStatus.NOT_STARTED.value),
                "claim_data": state.values.get("claim_data", {}),
                "pending_question": state.values.get("pending_question", None)
            }
        except Exception as e:
            print(f"Error obteniendo estado de reclamo: {e}")
            return {"claim_mode": False, "claim_status": ClaimStatus.NOT_STARTED.value}
    
    def reset_claim_mode(self):
        """Reinicia el modo reclamo"""
        try:
            config = {"configurable": {"thread_id": self.current_thread_id}}
            current_state = self.app.get_state(config)
            
            # Actualizar estado para desactivar modo reclamo
            new_state = current_state.values.copy()
            new_state.update({
                "claim_mode": False,
                "claim_status": ClaimStatus.NOT_STARTED.value,
                "claim_data": {},
                "pending_question": None
            })
            
            self.app.update_state(config, new_state)
            print("Modo reclamo reiniciado")
            
        except Exception as e:
            print(f"Error reiniciando modo reclamo: {e}")
    
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
                "conversation": self.get_history(),
                "claim_status": self.get_claim_status()
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
                    "sources": {"articulos": [], "casos": []},
                    "claim_mode": False,
                    "claim_status": ClaimStatus.NOT_STARTED.value,
                    "claim_data": {},
                    "pending_question": None
                }
                
                # Restaurar estado de reclamo si existe
                if "claim_status" in conversation_data:
                    claim_data = conversation_data["claim_status"]
                    initial_state.update({
                        "claim_mode": claim_data.get("claim_mode", False),
                        "claim_status": claim_data.get("claim_status", ClaimStatus.NOT_STARTED.value),
                        "claim_data": claim_data.get("claim_data", {}),
                        "pending_question": claim_data.get("pending_question", None)
                    })
                
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