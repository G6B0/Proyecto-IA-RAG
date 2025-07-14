import sys
import os
from datetime import datetime

# Añadir el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.legal_agent import LegalAgent

class LegalAgentInterface:
    """Interfaz interactiva para el agente legal"""
    
    def __init__(self):
        self.agent = LegalAgent()
        self.running = False
    
    def display_welcome(self):
        """Muestra el mensaje de bienvenida"""
        print("="*60)
        print("AGENTE LEGAL RAG - LEY DEL CONSUMIDOR CHILENA")
        print("="*60)
        print("Especializado en la Ley 19.496 sobre Protección de los Derechos de los Consumidores")
        print("\nEste agente puede ayudarte a:")
        print("- Redactar denuncias y reclamos")
        print("- Consultar artículos específicos de la ley")
        print("- Encontrar jurisprudencia relevante")
        print("- Entender tus derechos como consumidor")
        print("\nComandos especiales:")
        print("- '/ayuda' - Mostrar ayuda")
        print("- '/historial' - Ver historial de conversación")
        print("- '/limpiar' - Limpiar historial")
        print("- '/guardar' - Guardar conversación")
        print("- '/cargar' - Cargar conversación")
        print("- '/estado' - Ver estado del sistema")
        print("- '/thread' - Ver/cambiar thread de conversación")
        print("- '/salir' - Salir del programa")
        print("="*60)
    
    def display_help(self):
        """Muestra la ayuda del sistema"""
        print("\n" + "="*50)
        print("AYUDA DEL SISTEMA")
        print("="*50)
        print("Ejemplos de consultas que puedes hacer:")
        print("\nDENUNCIAS Y RECLAMOS:")
        print("- 'Quiero hacer una denuncia contra una empresa por producto defectuoso'")
        print("- 'Me vendieron un celular que no funciona, ¿qué puedo hacer?'")
        print("- 'Necesito redactar un reclamo por publicidad engañosa'")
        print("\nCONSULTAS LEGALES:")
        print("- '¿Cuáles son mis derechos como consumidor?'")
        print("- '¿Qué dice el artículo 19 sobre productos defectuosos?'")
        print("- '¿Cuánto tiempo tengo para reclamar por un producto?'")
        print("\nJURISPRUDENCIA:")
        print("- '¿Hay casos similares donde hayan ganado los consumidores?'")
        print("- 'Busca fallos sobre devolución de dinero'")
        print("\nCONSEJOS:")
        print("- Sé específico en tus consultas")
        print("- Puedes hacer preguntas de seguimiento")
        print("- El agente recuerda la conversación anterior")
        print("- Siempre cita las fuentes legales")
        print("="*50)
    
    def display_status(self):
        """Muestra el estado del sistema"""
        status = self.agent.get_status()
        print("\n" + "="*40)
        print("ESTADO DEL SISTEMA")
        print("="*40)
        print(f"Agente inicializado: {'Sí' if status['initialized'] else 'No'}")
        print(f"Thread ID actual: {status.get('thread_id', 'N/A')}")
        print(f"Mensajes en historial: {status.get('messages_count', 0)}")
        
        rag_status = status['rag_system_status']
        print(f"Documentos de ley: {rag_status.get('law_docs_count', 'N/A')}")
        print(f"Documentos de casos: {rag_status.get('case_docs_count', 'N/A')}")
        
        # Mostrar configuración si está disponible
        if 'config' in rag_status:
            config = rag_status['config']
            print(f"Modelo LLM: {config.get('llm_model', 'N/A')}")
            print(f"Modelo embeddings: {config.get('embedding_model', 'N/A')}")
        
        print("="*40)
    
    def display_history(self):
        """Muestra el historial de conversación"""
        history = self.agent.get_history()
        
        if not history:
            print("\nNo hay historial de conversación")
            return
        
        print("\n" + "="*50)
        print("HISTORIAL DE CONVERSACIÓN")
        print("="*50)
        print(f"Thread ID: {self.agent.get_thread_id()}")
        
        for i, message in enumerate(history, 1):
            role = "Usuario" if message["role"] == "user" else "Asistente"
            print(f"\n{i}. {role}:")
            print(f"   {message['content'][:200]}{'...' if len(message['content']) > 200 else ''}")
        
        print("="*50)
    
    def save_conversation(self):
        """Guarda la conversación"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversacion_{timestamp}.json"
        
        try:
            self.agent.save_conversation(filename)
            print(f"Conversación guardada en: {filename}")
        except Exception as e:
            print(f"Error guardando conversación: {e}")
    
    def load_conversation(self):
        """Carga una conversación"""
        filename = input("Nombre del archivo a cargar: ").strip()
        
        if not filename:
            print("Nombre de archivo vacío")
            return
        
        try:
            self.agent.load_conversation(filename)
            print("Conversación cargada exitosamente")
        except Exception as e:
            print(f"Error cargando conversación: {e}")
    
    def show_thread_info(self):
        """Muestra información del thread actual"""
        thread_id = self.agent.get_thread_id()
        history_count = len(self.agent.get_history())
        
        print(f"\nThread ID actual: {thread_id}")
        print(f"Mensajes en historial: {history_count}")
        
        change = input("¿Quieres cambiar a otro thread? (s/n): ").lower().strip()
        if change == 's':
            new_thread = input("Nuevo thread ID (Enter para generar automático): ").strip()
            if not new_thread:
                import uuid
                new_thread = str(uuid.uuid4())
            
            self.agent.set_thread_id(new_thread)
            print(f"Cambiado a thread: {new_thread}")
    
    def process_command(self, user_input: str) -> bool:
        """
        Procesa comandos especiales
        
        Args:
            user_input: Entrada del usuario
            
        Returns:
            bool: True si se procesó un comando, False si es una consulta normal
        """
        command = user_input.lower().strip()
        
        if command == '/ayuda':
            self.display_help()
            return True
        
        elif command == '/historial':
            self.display_history()
            return True
        
        elif command == '/limpiar':
            self.agent.clear_history()
            print("Historial limpiado")
            return True
        
        elif command == '/guardar':
            self.save_conversation()
            return True
        
        elif command == '/cargar':
            self.load_conversation()
            return True
        
        elif command == '/estado':
            self.display_status()
            return True
        
        elif command == '/thread':
            self.show_thread_info()
            return True
        
        elif command == '/salir':
            print("Hasta luego. Gracias por usar el Agente Legal RAG")
            self.running = False
            return True
        
        return False
    
    def run(self):
        """Ejecuta la interfaz interactiva"""
        try:
            # Mostrar bienvenida
            self.display_welcome()
            
            # Inicializar agente
            print("\nInicializando sistema...")
            self.agent.initialize()
            
            print("Sistema listo. Puedes empezar a hacer consultas.")
            
            self.running = True
            
            while self.running:
                try:
                    # Obtener entrada del usuario
                    print("\n" + "-"*60)
                    user_input = input("Tu consulta: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Procesar comandos especiales
                    if self.process_command(user_input):
                        continue
                    
                    # Procesar consulta normal
                    print("\nProcesando...")
                    response = self.agent.chat(user_input)
                    
                    # Mostrar respuesta
                    print("\n" + "="*60)
                    print("RESPUESTA DEL AGENTE LEGAL")
                    print("="*60)
                    print(response["answer"])
                    
                    # Mostrar información adicional si es relevante
                    if response["original_query"] != response["contextualized_query"]:
                        print(f"\nConsulta interpretada: {response['contextualized_query']}")
                
                except KeyboardInterrupt:
                    print("\n\nSaliendo del programa...")
                    self.running = False
                    break
                
                except Exception as e:
                    print(f"\nError procesando consulta: {e}")
                    print("Por favor, intenta nuevamente.")
        
        except Exception as e:
            print(f"Error iniciando el sistema: {e}")
            print("Verifica que Ollama esté ejecutándose y que tengas los modelos instalados:")
            print("- ollama pull llama3:8b")
            print("- ollama pull nomic-embed-text")

def main():
    """Función principal"""
    interface = LegalAgentInterface()
    interface.run()

if __name__ == "__main__":
    main()