from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from .claim_system import ClaimData
from .jurisprudence_matcher import JurisprudenceMatch
from .rag_system import RAGSystem

@dataclass
class GeneratedClaim:
    """Representa un reclamo formal generado"""
    claim_text: str
    legal_foundation: str
    jurisprudence_support: str
    requested_remedy: str
    full_document: str
    metadata: Dict[str, Any]

class FormalClaimGenerator:
    """Generador de reclamos formales basado en datos estructurados y jurisprudencia"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        
        # Templates para diferentes tipos de reclamos
        self.claim_templates = {
            "defecto_producto": {
                "intro": "Vengo en representar mi calidad de consumidor final, de conformidad con lo dispuesto en el artículo 1° N° 1 de la Ley N° 19.496 sobre Protección de los Derechos de los Consumidores, a formular el siguiente reclamo en contra de {empresa}:",
                "hechos": "PRIMERO: Con fecha {fecha_compra}, adquirí de {empresa} un {producto} por la suma de ${monto}, según consta en {documentos}.\n\nSEGUNDO: El producto adquirido presenta {problema}, lo que constituye un vicio o defecto que impide su uso normal y conforme a su destino.\n\nTERCERO: He intentado resolver esta situación {intentos_previos}, sin obtener una respuesta satisfactoria.",
                "derecho": "Los hechos expuestos configuran una infracción a los derechos del consumidor establecidos en la Ley N° 19.496, específicamente:",
                "peticiones": "Por tanto, solicito a esa Dirección Regional del SERNAC:"
            },
            "mal_servicio": {
                "intro": "En mi calidad de consumidor, conforme al artículo 1° de la Ley N° 19.496, vengo en formular reclamo en contra de {empresa} por deficiencias en la prestación de servicios:",
                "hechos": "PRIMERO: Con fecha {fecha_servicio}, contraté los servicios de {empresa} para {descripcion_servicio}.\n\nSEGUNDO: La empresa no cumplió con {incumplimiento}, incurriendo en mal servicio e incumplimiento de lo ofrecido.\n\nTERCERO: Pese a mis requerimientos, la empresa no ha dado solución al problema.",
                "derecho": "Los hechos descritos constituyen infracciones a la Ley del Consumidor:",
                "peticiones": "En virtud de lo expuesto, solicito:"
            },
            "publicidad_engañosa": {
                "intro": "Como consumidor afectado por publicidad engañosa, según lo dispuesto en el artículo 28 de la Ley N° 19.496, formulo el siguiente reclamo:",
                "hechos": "PRIMERO: La empresa {empresa} publicitó {producto_servicio} con las siguientes características: {publicidad}.\n\nSEGUNDO: Al adquirir el producto/servicio, constaté que {diferencia_realidad}.\n\nTERCERO: Esta situación constituye publicidad engañosa según el artículo 28 de la Ley del Consumidor.",
                "derecho": "La conducta descrita infringe las siguientes disposiciones legales:",
                "peticiones": "Por las consideraciones expuestas, solicito:"
            }
        }
        
        # Artículos relevantes por tipo de problema
        self.relevant_articles = {
            "defecto_producto": [
                {"numero": "3°", "contenido": "Son derechos y deberes básicos del consumidor: a) La libre elección del bien o servicio..."},
                {"numero": "12", "contenido": "Todo producto o servicio debe cumplir con la normativa vigente..."},
                {"numero": "19", "contenido": "Los proveedores responderán por la idoneidad y calidad de los productos..."},
                {"numero": "20", "contenido": "Los proveedores de productos serán responsables de los daños materiales y morales..."}
            ],
            "mal_servicio": [
                {"numero": "3°", "contenido": "Son derechos y deberes básicos del consumidor..."},
                {"numero": "23", "contenido": "Los proveedores de servicios serán responsables de los daños materiales y morales..."},
                {"numero": "41", "contenido": "Las empresas que presten servicios de utilidad pública..."}
            ],
            "publicidad_engañosa": [
                {"numero": "28", "contenido": "Comete infracción a las disposiciones de la presente ley el que, a sabiendas o debiendo saberlo..."},
                {"numero": "3° letra b)", "contenido": "El derecho a una información veraz y oportuna sobre los bienes y servicios ofrecidos..."}
            ]
        }
        
        # Tipos de reparación por problema
        self.remedy_types = {
            "defecto_producto": [
                "Cambio del producto por uno nuevo y en perfecto estado",
                "Devolución del dinero pagado con reajustes e intereses",
                "Reparación gratuita del producto",
                "Indemnización por daños materiales y morales"
            ],
            "mal_servicio": [
                "Prestación adecuada del servicio contratado",
                "Devolución de lo pagado por el servicio deficiente",
                "Compensación por perjuicios ocasionados",
                "Mejoramiento de los estándares de servicio"
            ],
            "publicidad_engañosa": [
                "Cumplimiento de lo publicitado",
                "Devolución del dinero con reajustes",
                "Indemnización por daños y perjuicios",
                "Corrección de la publicidad engañosa"
            ]
        }
    
    def generate_formal_claim(self, claim_data: ClaimData, jurisprudence_matches: List[JurisprudenceMatch]) -> GeneratedClaim:
        """
        Genera un reclamo formal basado en los datos recolectados y jurisprudencia
        
        Args:
            claim_data: Datos del reclamo
            jurisprudence_matches: Casos jurisprudenciales similares
            
        Returns:
            Reclamo formal generado
        """
        # Determinar tipo de reclamo
        claim_type = self._determine_claim_type(claim_data)
        
        # Obtener artículos relevantes de la ley
        relevant_law_articles = self._get_relevant_law_articles(claim_data, claim_type)
        
        # Generar cada sección del reclamo
        header = self._generate_header(claim_data)
        intro = self._generate_intro(claim_data, claim_type)
        facts = self._generate_facts_section(claim_data, claim_type)
        legal_foundation = self._generate_legal_foundation(claim_data, claim_type, relevant_law_articles)
        jurisprudence_section = self._generate_jurisprudence_section(jurisprudence_matches)
        petitions = self._generate_petitions(claim_data, claim_type)
        closing = self._generate_closing(claim_data)
        
        # Ensamblar documento completo
        full_document = f"{header}\n\n{intro}\n\n{facts}\n\n{legal_foundation}\n\n{jurisprudence_section}\n\n{petitions}\n\n{closing}"
        
        return GeneratedClaim(
            claim_text=facts,
            legal_foundation=legal_foundation,
            jurisprudence_support=jurisprudence_section,
            requested_remedy=petitions,
            full_document=full_document,
            metadata={
                "claim_type": claim_type,
                "generation_date": datetime.now().isoformat(),
                "data_completeness": claim_data.get_completion_percentage(),
                "jurisprudence_matches": len(jurisprudence_matches)
            }
        )
    
    def _determine_claim_type(self, claim_data: ClaimData) -> str:
        """Determina el tipo de reclamo basado en el problema descrito"""
        if not claim_data.problem_description:
            return "defecto_producto"  # Default
        
        problem_lower = claim_data.problem_description.lower()
        
        # Publicidad engañosa
        if any(word in problem_lower for word in ["engañ", "ment", "falso", "no es lo que", "diferente a lo ofrecido"]):
            return "publicidad_engañosa"
        
        # Mal servicio
        if any(word in problem_lower for word in ["servicio", "atención", "no me atendieron", "demora", "tardanza"]):
            return "mal_servicio"
        
        # Defecto de producto (default)
        return "defecto_producto"
    
    def _get_relevant_law_articles(self, claim_data: ClaimData, claim_type: str) -> List[Dict]:
        """Obtiene artículos relevantes de la ley usando el RAG system"""
        # Buscar artículos específicos para el tipo de reclamo
        base_articles = self.relevant_articles.get(claim_type, self.relevant_articles["defecto_producto"])
        
        # Usar RAG para obtener contenido completo de artículos
        search_query = f"artículo {claim_data.product_type or 'producto'} {claim_data.problem_description or 'defectuoso'}"
        law_docs = self.rag_system.retrieve_law_documents(search_query)
        
        # Combinar artículos base con los encontrados por RAG
        enhanced_articles = []
        
        for article in base_articles:
            # Buscar el contenido completo del artículo en los documentos RAG
            full_content = self._find_article_content(article["numero"], law_docs)
            enhanced_articles.append({
                "numero": article["numero"],
                "contenido": full_content or article["contenido"]
            })
        
        return enhanced_articles
    
    def _find_article_content(self, article_number: str, law_docs: List) -> Optional[str]:
        """Busca el contenido completo de un artículo en los documentos de ley"""
        for doc in law_docs:
            if f"artículo {article_number}" in doc.page_content.lower():
                return doc.page_content
        return None
    
    def _generate_header(self, claim_data: ClaimData) -> str:
        """Genera el encabezado del reclamo"""
        return f"""SERVICIO NACIONAL DEL CONSUMIDOR (SERNAC)
DIRECCIÓN REGIONAL METROPOLITANA

RECLAMO POR INFRACCIÓN A LA LEY DEL CONSUMIDOR

Consumidor: {claim_data.consumer_name or '[NOMBRE DEL CONSUMIDOR]'}
RUT: {claim_data.consumer_rut or '[RUT DEL CONSUMIDOR]'}
Dirección: {claim_data.consumer_address or '[DIRECCIÓN DEL CONSUMIDOR]'}
Teléfono: {claim_data.consumer_phone or '[TELÉFONO DEL CONSUMIDOR]'}
Email: {claim_data.consumer_email or '[EMAIL DEL CONSUMIDOR]'}

Proveedor: {claim_data.purchase_place or '[NOMBRE DE LA EMPRESA]'}

Fecha: {datetime.now().strftime('%d de %B de %Y')}"""
    
    def _generate_intro(self, claim_data: ClaimData, claim_type: str) -> str:
        """Genera la introducción del reclamo"""
        template = self.claim_templates[claim_type]["intro"]
        
        return template.format(
            empresa=claim_data.purchase_place or "[EMPRESA]"
        )
    
    def _generate_facts_section(self, claim_data: ClaimData, claim_type: str) -> str:
        """Genera la sección de hechos"""
        template = self.claim_templates[claim_type]["hechos"]
        
        # Formatear fecha de compra
        fecha_compra = self._format_date(claim_data.purchase_date) if claim_data.purchase_date else "[FECHA DE COMPRA]"
        
        # Formatear monto
        monto = self._format_amount(claim_data.purchase_amount) if claim_data.purchase_amount else "[MONTO]"
        
        # Generar descripción de intentos previos
        intentos_previos = self._format_previous_attempts(claim_data.previous_attempts)
        
        return f"""HECHOS

{template.format(
    fecha_compra=fecha_compra,
    empresa=claim_data.purchase_place or "[EMPRESA]",
    producto=claim_data.product_type or "[PRODUCTO]",
    monto=monto,
    documentos="boleta/factura de compra" if claim_data.receipts_available else "[DOCUMENTOS]",
    problema=claim_data.problem_description or "[DESCRIPCIÓN DEL PROBLEMA]",
    intentos_previos=intentos_previos
)}"""
    
    def _generate_legal_foundation(self, claim_data: ClaimData, claim_type: str, articles: List[Dict]) -> str:
        """Genera la fundamentación legal"""
        foundation = f"""FUNDAMENTOS LEGALES

{self.claim_templates[claim_type]["derecho"]}

"""
        
        for i, article in enumerate(articles, 1):
            foundation += f"{i}. **Artículo {article['numero']} de la Ley N° 19.496:**\n"
            foundation += f"   {article['contenido'][:300]}...\n\n"
        
        foundation += """Estas disposiciones han sido vulneradas por el proveedor, generando responsabilidad civil y administrativa conforme a la normativa de protección al consumidor."""
        
        return foundation
    
    def _generate_jurisprudence_section(self, matches: List[JurisprudenceMatch]) -> str:
        """Genera la sección de jurisprudencia"""
        if not matches:
            return """JURISPRUDENCIA

No se encontraron casos jurisprudenciales específicamente similares, sin embargo, la jurisprudencia ha sido consistente en proteger los derechos de los consumidores en casos análogos."""
        
        section = """JURISPRUDENCIA

Los tribunales han fallado en casos similares de la siguiente manera:

"""
        
        for i, match in enumerate(matches[:3], 1):  # Máximo 3 casos
            rol = match.document.metadata.get('Rol', 'N/A')
            corte = match.document.metadata.get('Corte_origen', 'N/A')
            fecha = match.document.metadata.get('Fecha_Sentencia', 'N/A')
            
            section += f"{i}. **Caso Rol {rol}** ({corte}, {fecha}):\n"
            section += f"   Similitud: {match.similarity_score:.1%}\n"
            section += f"   Resultado: {match.outcome}\n"
            section += f"   Resumen: {match.case_summary[:200]}...\n\n"
        
        section += "Esta jurisprudencia respalda la procedencia del presente reclamo y la responsabilidad del proveedor."
        
        return section
    
    def _generate_petitions(self, claim_data: ClaimData, claim_type: str) -> str:
        """Genera las peticiones del reclamo"""
        remedies = self.remedy_types.get(claim_type, self.remedy_types["defecto_producto"])
        
        petitions = f"""{self.claim_templates[claim_type]["peticiones"]}

1. **Que se acoja el presente reclamo** y se declare que {claim_data.purchase_place or '[EMPRESA]'} ha incurrido en infracción a la Ley del Consumidor.

2. **Que se ordene al proveedor** implementar las siguientes medidas reparatorias:
"""
        
        for i, remedy in enumerate(remedies, 1):
            petitions += f"   {chr(96+i)}) {remedy}\n"
        
        petitions += f"""
3. **Que se apliquen las sanciones** administrativas correspondientes según la gravedad de la infracción.

4. **Que se adopten medidas** para evitar que situaciones similares se repitan en el futuro.

5. **En subsidio**, cualquier otra medida que el SERNAC estime pertinente para la protección de los derechos del consumidor."""
        
        return petitions
    
    def _generate_closing(self, claim_data: ClaimData) -> str:
        """Genera el cierre del reclamo"""
        return f"""
Por tanto, ruego a Ud. tener por presentado este reclamo y acogerlo en todas sus partes.

Saluda atentamente,


_________________________
{claim_data.consumer_name or '[NOMBRE DEL CONSUMIDOR]'}
RUT: {claim_data.consumer_rut or '[RUT DEL CONSUMIDOR]'}"""
    
    def _format_date(self, date_str: str) -> str:
        """Formatea una fecha para el documento"""
        if not date_str:
            return "[FECHA]"
        
        # Si es "hace X días/semanas/meses", convertir a fecha aproximada
        if "hace" in date_str.lower():
            # Simplificado - en producción se calcularía la fecha exacta
            return f"aproximadamente {date_str}"
        
        return date_str
    
    def _format_amount(self, amount_str: str) -> str:
        """Formatea un monto para el documento"""
        if not amount_str:
            return "[MONTO]"
        
        # Extraer números y formatear
        import re
        numbers = re.findall(r'\d+', amount_str.replace('.', '').replace(',', ''))
        if numbers:
            amount = int(numbers[0])
            return f"{amount:,}".replace(',', '.')
        
        return amount_str
    
    def _format_previous_attempts(self, attempts: List[str]) -> str:
        """Formatea los intentos previos de solución"""
        if not attempts:
            return "contactando directamente a la empresa, sin obtener respuesta satisfactoria"
        
        if len(attempts) == 1:
            return attempts[0]
        
        formatted = ", ".join(attempts[:-1]) + f" y {attempts[-1]}"
        return formatted
    
    def generate_claim_summary(self, claim: GeneratedClaim) -> str:
        """Genera un resumen ejecutivo del reclamo"""
        return f"""RESUMEN DEL RECLAMO GENERADO

Tipo de reclamo: {claim.metadata['claim_type'].replace('_', ' ').title()}
Fecha de generación: {datetime.fromisoformat(claim.metadata['generation_date']).strftime('%d/%m/%Y')}
Completitud de datos: {claim.metadata['data_completeness']:.1f}%
Casos jurisprudenciales encontrados: {claim.metadata['jurisprudence_matches']}

El reclamo ha sido estructurado conforme a la normativa legal vigente e incluye:
- Fundamentación legal específica
- Jurisprudencia de apoyo (cuando disponible)
- Peticiones concretas y realizables
- Formato formal para presentación ante SERNAC

IMPORTANTE: Revise el documento completo antes de presentarlo y complete cualquier información faltante marcada entre corchetes."""