from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime
import json

class ClaimStatus(Enum):
    """Estados del proceso de reclamo"""
    NOT_STARTED = "not_started"
    COLLECTING_DATA = "collecting_data"
    DATA_COMPLETE = "data_complete"
    GENERATING_CLAIM = "generating_claim"
    CLAIM_READY = "claim_ready"

@dataclass
class ClaimData:
    """Estructura para almacenar datos del reclamo"""
    # Datos del producto/servicio
    product_type: Optional[str] = None
    product_name: Optional[str] = None
    problem_description: Optional[str] = None
    
    # Datos de compra
    purchase_date: Optional[str] = None
    purchase_place: Optional[str] = None
    purchase_amount: Optional[str] = None
    
    # Datos del consumidor
    consumer_name: Optional[str] = None
    consumer_rut: Optional[str] = None
    consumer_address: Optional[str] = None
    consumer_phone: Optional[str] = None
    consumer_email: Optional[str] = None
    
    # Datos del problema
    incident_date: Optional[str] = None
    previous_attempts: Optional[List[str]] = field(default_factory=list)
    desired_solution: Optional[str] = None
    
    # Datos adicionales
    warranty_info: Optional[str] = None
    receipts_available: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "product_type": self.product_type,
            "product_name": self.product_name,
            "problem_description": self.problem_description,
            "purchase_date": self.purchase_date,
            "purchase_place": self.purchase_place,
            "purchase_amount": self.purchase_amount,
            "consumer_name": self.consumer_name,
            "consumer_rut": self.consumer_rut,
            "consumer_address": self.consumer_address,
            "consumer_phone": self.consumer_phone,
            "consumer_email": self.consumer_email,
            "incident_date": self.incident_date,
            "previous_attempts": self.previous_attempts,
            "desired_solution": self.desired_solution,
            "warranty_info": self.warranty_info,
            "receipts_available": self.receipts_available
        }
    
    def get_completion_percentage(self) -> float:
        """Calcula el porcentaje de datos completados"""
        essential_fields = [
            'product_type', 'problem_description', 'purchase_date', 
            'purchase_place', 'consumer_name', 'consumer_rut'
        ]
        
        completed = sum(1 for field in essential_fields if getattr(self, field) is not None)
        return (completed / len(essential_fields)) * 100

class ClaimIntentDetector:
    """Detector de intención de reclamo en lenguaje natural"""
    
    def __init__(self):
        self.claim_keywords = [
            # Palabras directas de reclamo
            "reclamo", "reclamar", "queja", "quejar", "denuncia", "denunciar",
            "demanda", "demandar", "problema", "inconveniente", "molestia",
            
            # Problemas con productos
            "no funciona", "defectuoso", "roto", "malo", "dañado", "averiado",
            "no sirve", "no anda", "se rompió", "se dañó", "falla", "error",
            
            # Problemas con servicios
            "mal servicio", "mala atención", "no me atendieron", "me estafaron",
            "no cumplieron", "no entregaron", "me mintieron", "publicidad engañosa",
            
            # Acciones legales
            "mis derechos", "consumidor", "garantía", "devolución", "reembolso",
            "compensación", "indemnización", "sernac", "ley del consumidor"
        ]
        
        self.product_patterns = [
            r"compr[éeií]\s+un[ao]?\s+(\w+)",
            r"el\s+(\w+)\s+(?:que|no|se)",
            r"mi\s+(\w+)\s+(?:no|se|está)",
            r"(?:un|una|el|la)\s+(\w+)\s+(?:defectuoso|roto|malo|dañado)"
        ]
        
        self.store_patterns = [
            r"en\s+(falabella|ripley|paris|hites|corona|easy|homecenter|jumbo|lider|santa isabel|unimarc)",
            r"(?:en|de)\s+la\s+tienda\s+(\w+)",
            r"(?:en|de)\s+(\w+\s*\w*)\s+(?:compré|compre|adquirí)"
        ]
        
        self.date_patterns = [
            r"hace\s+(\d+)\s+(?:día|días|semana|semanas|mes|meses)",
            r"el\s+(\d{1,2})\s+de\s+(\w+)",
            r"(\d{1,2})/(\d{1,2})/(\d{4})",
            r"ayer|anteayer|la semana pasada|el mes pasado"
        ]
    
    def detect_claim_intent(self, text: str) -> Dict[str, Any]:
        """
        Detecta si el texto indica intención de hacer un reclamo
        
        Args:
            text: Texto del usuario
            
        Returns:
            Dict con información sobre la intención detectada
        """
        text_lower = text.lower()
        
        # Buscar palabras clave de reclamo
        claim_indicators = []
        for keyword in self.claim_keywords:
            if keyword in text_lower:
                claim_indicators.append(keyword)
        
        # Detectar información inicial
        initial_data = self._extract_initial_data(text)
        
        # Calcular score de confianza
        confidence_score = min(100, len(claim_indicators) * 20 + len(initial_data) * 15)
        
        has_claim_intent = confidence_score > 30
        
        return {
            "has_claim_intent": has_claim_intent,
            "confidence_score": confidence_score,
            "claim_indicators": claim_indicators,
            "initial_data": initial_data,
            "detected_patterns": self._get_detected_patterns(text)
        }
    
    def _extract_initial_data(self, text: str) -> Dict[str, Any]:
        """Extrae datos iniciales del texto"""
        data = {}
        
        # Extraer productos mencionados
        for pattern in self.product_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                data["potential_products"] = matches
                break
        
        # Extraer tiendas mencionadas
        for pattern in self.store_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                data["potential_stores"] = matches
                break
        
        # Extraer fechas mencionadas
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                data["potential_dates"] = matches
                break
        
        return data
    
    def _get_detected_patterns(self, text: str) -> List[str]:
        """Obtiene patrones detectados para logging"""
        patterns = []
        
        if any(word in text.lower() for word in ["no funciona", "defectuoso", "roto", "malo"]):
            patterns.append("product_problem")
        
        if any(word in text.lower() for word in ["compré", "compre", "adquirí"]):
            patterns.append("purchase_mention")
        
        if any(word in text.lower() for word in ["reclamo", "queja", "problema"]):
            patterns.append("complaint_direct")
        
        return patterns

class ConversationalDataCollector:
    """Recolector de datos a través de conversación natural"""
    
    def __init__(self):
        self.data_fields = {
            "product_type": {
                "priority": 1,
                "questions": [
                    "¿Qué tipo de producto o servicio tienes problemas?",
                    "¿Podrías contarme más detalles sobre el producto?",
                    "¿De qué producto específicamente se trata?"
                ],
                "extractors": [
                    r"(?:el|la|un|una)\s+(\w+)",
                    r"(?:producto|servicio)\s+(\w+)",
                    r"^(\w+)\s+(?:que|no|se)"
                ]
            },
            "problem_description": {
                "priority": 2,
                "questions": [
                    "¿Qué problema específico tiene el producto?",
                    "¿Podrías describir qué es lo que no está funcionando bien?",
                    "¿Cuál es exactamente el inconveniente que tienes?"
                ],
                "extractors": [
                    r"(?:no funciona|defectuoso|roto|malo|dañado|averiado|no sirve|no anda|se rompió|se dañó|falla|error)"
                ]
            },
            "purchase_date": {
                "priority": 3,
                "questions": [
                    "¿Cuándo compraste este producto?",
                    "¿Recuerdas la fecha de compra?",
                    "¿Hace cuánto tiempo lo adquiriste?"
                ],
                "extractors": [
                    r"hace\s+(\d+)\s+(?:día|días|semana|semanas|mes|meses)",
                    r"el\s+(\d{1,2})\s+de\s+(\w+)",
                    r"(\d{1,2})/(\d{1,2})/(\d{4})"
                ]
            },
            "purchase_place": {
                "priority": 4,
                "questions": [
                    "¿En qué tienda o lugar lo compraste?",
                    "¿Dónde hiciste la compra?",
                    "¿Cuál fue la empresa o tienda vendedora?"
                ],
                "extractors": [
                    r"en\s+(falabella|ripley|paris|hites|corona|easy|homecenter|jumbo|lider|santa isabel|unimarc)",
                    r"(?:en|de)\s+la\s+tienda\s+(\w+)",
                    r"(?:en|de)\s+(\w+\s*\w*)"
                ]
            },
            "consumer_name": {
                "priority": 5,
                "questions": [
                    "Para el reclamo formal, necesito tu nombre completo",
                    "¿Cuál es tu nombre completo?",
                    "¿Podrías decirme tu nombre y apellido?"
                ],
                "extractors": [
                    r"(?:me llamo|soy|mi nombre es)\s+([A-ZÁÉÍÓÚÑa-záéíóúñ\s]+)",
                    r"^([A-ZÁÉÍÓÚÑa-záéíóúñ\s]+)$"
                ]
            },
            "consumer_rut": {
                "priority": 6,
                "questions": [
                    "¿Cuál es tu RUT?",
                    "Necesito tu RUT para el reclamo formal",
                    "¿Podrías darme tu RUT con el dígito verificador?"
                ],
                "extractors": [
                    r"(\d{1,2}\.\d{3}\.\d{3}-[\dkK])",
                    r"(\d{7,8}-[\dkK])"
                ]
            }
        }
        
        self.current_field = None
        self.asked_questions = set()
    
    def get_next_question(self, claim_data: ClaimData) -> Optional[str]:
        """
        Obtiene la próxima pregunta a hacer basada en los datos faltantes
        
        Args:
            claim_data: Datos actuales del reclamo
            
        Returns:
            Pregunta a hacer o None si no hay más preguntas
        """
        # Ordenar campos por prioridad
        sorted_fields = sorted(self.data_fields.items(), key=lambda x: x[1]["priority"])
        
        for field_name, field_config in sorted_fields:
            field_value = getattr(claim_data, field_name)
            
            # Si el campo está vacío, hacer una pregunta
            if field_value is None:
                # Elegir una pregunta que no se haya hecho
                available_questions = [
                    q for q in field_config["questions"] 
                    if q not in self.asked_questions
                ]
                
                if available_questions:
                    question = available_questions[0]
                    self.asked_questions.add(question)
                    self.current_field = field_name
                    return question
        
        return None
    
    def extract_data_from_response(self, response: str, claim_data: ClaimData) -> bool:
        """
        Extrae datos de la respuesta del usuario
        
        Args:
            response: Respuesta del usuario
            claim_data: Objeto ClaimData a actualizar
            
        Returns:
            True si se extrajo algún dato, False si no
        """
        extracted = False
        
        # Si estamos esperando un campo específico
        if self.current_field and self.current_field in self.data_fields:
            field_config = self.data_fields[self.current_field]
            
            # Intentar extraer usando los patrones del campo
            for pattern in field_config["extractors"]:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    # Tomar el primer match (o procesarlo según el campo)
                    extracted_value = matches[0] if isinstance(matches[0], str) else " ".join(matches[0])
                    setattr(claim_data, self.current_field, extracted_value.strip())
                    extracted = True
                    break
            
            # Si no se pudo extraer con patrones, usar respuesta directa para algunos campos
            if not extracted and self.current_field in ["problem_description", "consumer_name"]:
                if len(response.strip()) > 2:  # Respuesta válida
                    setattr(claim_data, self.current_field, response.strip())
                    extracted = True
        
        # También intentar extraer datos de todos los campos (extracción oportunista)
        self._extract_opportunistic_data(response, claim_data)
        
        # Limpiar el campo actual si se extrajo algo
        if extracted:
            self.current_field = None
        
        return extracted
    
    def _extract_opportunistic_data(self, text: str, claim_data: ClaimData):
        """Extrae datos oportunísticamente de cualquier texto"""
        for field_name, field_config in self.data_fields.items():
            # Solo extraer si el campo está vacío
            if getattr(claim_data, field_name) is None:
                for pattern in field_config["extractors"]:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        extracted_value = matches[0] if isinstance(matches[0], str) else " ".join(matches[0])
                        setattr(claim_data, field_name, extracted_value.strip())
                        break
    
    def is_data_sufficient(self, claim_data: ClaimData) -> bool:
        """
        Verifica si se tienen suficientes datos para generar un reclamo
        
        Args:
            claim_data: Datos del reclamo
            
        Returns:
            True si hay suficientes datos
        """
        essential_fields = ["product_type", "problem_description", "purchase_date", "purchase_place"]
        
        for field in essential_fields:
            if getattr(claim_data, field) is None:
                return False
        
        return True
    
    def get_missing_essential_data(self, claim_data: ClaimData) -> List[str]:
        """Obtiene lista de datos esenciales faltantes"""
        essential_fields = {
            "product_type": "tipo de producto",
            "problem_description": "descripción del problema", 
            "purchase_date": "fecha de compra",
            "purchase_place": "lugar de compra",
            "consumer_name": "nombre del consumidor",
            "consumer_rut": "RUT del consumidor"
        }
        
        missing = []
        for field, description in essential_fields.items():
            if getattr(claim_data, field) is None:
                missing.append(description)
        
        return missing