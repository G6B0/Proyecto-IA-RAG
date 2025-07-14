from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from .config import Config
from .rag_system import RAGSystem

class LegalAgent:
    """Agente legal con capacidad de mantener historial de conversación"""
    
    def __init__(self):
        self.config = Config()
        self.rag_system = RAGSystem()
        self.chat_history: List[BaseMessage] = []
        self.session_initialized = False
    
    def initialize(self):
        """Inicializa el agente y el sistema RAG"""
        print("Inicializando agente legal...")
        self.rag_system.initialize()
        self.session_initialized = True
        print("Agente legal listo para usar")
    
    def _format_chat_history(self) -> str:
        """
        Formatea el historial de chat para incluir en prompts
        
        Returns:
            str: Historial formateado
        """
        if not self.chat_history:
            return "No hay historial de conversación previo."
        
        formatted_history = []
        for message in self.chat_history[-6:]:  # Últimas 6 mensajes para no sobrecargar
            if isinstance(message, HumanMessage):
                formatted_history.append(f"Usuario: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"Asistente: {message.content}")
        
        return "\n".join(formatted_history)
    
    def _should_contextualize(self, query: str) -> bool:
        """
        Determina si la consulta necesita contextualización
        
        Args:
            query: Consulta del usuario
            
        Returns:
            bool: True si necesita contextualización
        """
        if not self.chat_history:
            return False
        
        # Palabras que indican referencia a conversación previa
        context_indicators = [
            "esto", "eso", "aquello", "anterior", "previo", "mencionado",
            "dijiste", "explicaste", "hablamos", "comentaste", "también",
            "además", "pero", "sin embargo", "y si", "qué pasa si",
            "en ese caso", "entonces", "ahora", "después", "seguir"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in context_indicators)
    
    def _contextualize_question(self, query: str) -> str:
        """
        Contextualiza la pregunta usando el historial
        
        Args:
            query: Consulta original
            
        Returns:
            str: Consulta contextualizada
        """
        if not self._should_contextualize(query):
            return query
        
        # Crear prompt para contextualización
        prompt = ChatPromptTemplate.from_template(self.config.CONTEXTUALIZE_PROMPT)
        
        formatted_prompt = prompt.format(
            chat_history=self._format_chat_history(),
            question=query
        )
        
        try:
            response = self.rag_system.llm.invoke(formatted_prompt)
            contextualized_query = response.content.strip()
            
            print(f"Consulta contextualizada: {contextualized_query}")
            return contextualized_query
            
        except Exception as e:
            print(f"Error en contextualización: {e}")
            return query
    
    def _add_to_history(self, human_message: str, ai_message: str):
        """
        Añade mensajes al historial
        
        Args:
            human_message: Mensaje del usuario
            ai_message: Respuesta del asistente
        """
        self.chat_history.append(HumanMessage(content=human_message))
        self.chat_history.append(AIMessage(content=ai_message))
        
        # Mantener historial manageable (últimas 20 mensajes)
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
    
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
        
        # Contextualizar la pregunta si es necesario
        contextualized_query = self._contextualize_question(query)
        
        # Generar respuesta usando RAG
        rag_response = self.rag_system.generate_response(
            contextualized_query,
            self._format_chat_history()
        )
        
        # Formatear respuesta final
        answer = rag_response["answer"]
        sources = rag_response["sources"]
        
        # Añadir fuentes a la respuesta
        formatted_answer = self._format_answer_with_sources(answer, sources)
        
        # Actualizar historial
        self._add_to_history(query, formatted_answer)
        
        return {
            "answer": formatted_answer,
            "sources": sources,
            "contextualized_query": contextualized_query,
            "original_query": query,
            "chat_history_length": len(self.chat_history)
        }
    
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
        if sources["articulos"]:
            formatted_answer += "\n\n**Artículos utilizados como fuente:**\n"
            for articulo in sources["articulos"]:
                formatted_answer += f"- {articulo}\n"
        
        # Añadir fuentes de casos
        if sources["casos"]:
            formatted_answer += "\n**Casos judiciales utilizados como fuente:**\n"
            for caso in sources["casos"]:
                formatted_answer += f"- {caso}\n"
        
        return formatted_answer
    
    def clear_history(self):
        self.chat_history = []
        print("Historial de conversación limpiado")
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Obtiene el historial de conversación en formato legible
        
        Returns:
            List[Dict]: Historial formateado
        """
        history = []
        for message in self.chat_history:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        
        return history
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado del agente
        
        Returns:
            Dict: Estado del agente
        """
        return {
            "initialized": self.session_initialized,
            "chat_history_length": len(self.chat_history),
            "rag_system_status": self.rag_system.get_status()
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
            
            # Reconstruir historial
            self.chat_history = []
            for message in conversation_data.get("conversation", []):
                if message["role"] == "user":
                    self.chat_history.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    self.chat_history.append(AIMessage(content=message["content"]))
            
            print(f"Conversación cargada desde {filename}")
            print(f"Mensajes cargados: {len(self.chat_history)}")
            
        except Exception as e:
            print(f"Error cargando conversación: {e}")