from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime, timedelta
from langchain_core.documents import Document
from .claim_system import ClaimData
from .rag_system import RAGSystem

@dataclass
class JurisprudenceMatch:
    """Representa un match de jurisprudencia con un caso del usuario"""
    document: Document
    similarity_score: float
    match_reasons: List[str]
    relevant_articles: List[str]
    case_summary: str
    outcome: str  # favorable, desfavorable, parcial

class JurisprudenceMatcher:
    """Matcher inteligente de jurisprudencia basado en datos estructurados del reclamo"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        
        # Categorías de productos y sus sinónimos
        self.product_categories = {
            "electrónicos": [
                "celular", "teléfono", "smartphone", "tablet", "laptop", "computador", 
                "televisor", "tv", "radio", "auriculares", "parlante", "cámara",
                "consola", "nintendo", "playstation", "xbox"
            ],
            "electrodomésticos": [
                "refrigerador", "lavadora", "secadora", "microondas", "aspiradora",
                "plancha", "tostadora", "licuadora", "batidora", "cafetera"
            ],
            "vehículos": [
                "auto", "automóvil", "carro", "vehículo", "moto", "motocicleta", 
                "camión", "camioneta", "bicicleta"
            ],
            "ropa": [
                "ropa", "camisa", "pantalón", "zapatos", "zapatillas", "chaqueta",
                "vestido", "polera", "jeans", "abrigo"
            ],
            "muebles": [
                "sofá", "silla", "mesa", "cama", "ropero", "estante", "escritorio",
                "sillón", "comedor", "dormitorio"
            ],
            "servicios": [
                "servicio", "instalación", "reparación", "mantención", "delivery",
                "envío", "garantía", "soporte"
            ]
        }
        
        # Tipos de problemas y sus sinónimos
        self.problem_types = {
            "defecto_fabricación": [
                "no funciona", "defectuoso", "roto", "malo", "dañado", "averiado",
                "no sirve", "no anda", "se rompió", "se dañó", "falla", "error",
                "vicio oculto", "falla de fábrica"
            ],
            "no_conformidad": [
                "no es lo que pedí", "diferente", "no corresponde", "equivocado",
                "no es igual", "distinto", "no coincide"
            ],
            "publicidad_engañosa": [
                "publicidad engañosa", "me mintieron", "no cumple lo prometido",
                "false advertising", "información falsa", "engaño"
            ],
            "mal_servicio": [
                "mal servicio", "mala atención", "no me atendieron", "groseros",
                "demora", "tardanza", "no cumplieron", "no entregaron"
            ],
            "garantía": [
                "garantía", "warranty", "no respetan garantía", "garantía vencida",
                "reparación", "cambio", "devolución"
            ]
        }
        
        # Patrones de montos para calcular similitud
        self.amount_ranges = {
            "bajo": (0, 100000),
            "medio": (100000, 500000),
            "alto": (500000, 2000000),
            "muy_alto": (2000000, float('inf'))
        }
        
        # Empresas conocidas y sus variantes
        self.known_companies = {
            "falabella": ["falabella", "saga falabella"],
            "ripley": ["ripley", "ripley chile"],
            "paris": ["paris", "tiendas paris"],
            "hites": ["hites"],
            "corona": ["corona"],
            "easy": ["easy", "easy chile"],
            "homecenter": ["homecenter", "sodimac"],
            "jumbo": ["jumbo"],
            "lider": ["lider", "walmart"],
            "santa isabel": ["santa isabel"],
            "unimarc": ["unimarc"],
            "mall": ["mall", "centro comercial"],
            "supermercado": ["supermercado", "super"]
        }
    
    def find_similar_cases(self, claim_data: ClaimData, limit: int = 5) -> List[JurisprudenceMatch]:
        """
        Encuentra casos jurisprudenciales similares basados en los datos del reclamo
        
        Args:
            claim_data: Datos del reclamo del usuario
            limit: Número máximo de casos a retornar
            
        Returns:
            Lista de matches ordenados por similitud
        """
        if not self.rag_system.initialized:
            raise RuntimeError("RAG System no inicializado")
        
        # Generar queries de búsqueda basadas en los datos
        search_queries = self._generate_search_queries(claim_data)
        
        # Recuperar documentos relevantes
        relevant_docs = []
        for query in search_queries:
            docs = self.rag_system.retrieve_case_documents(query)
            relevant_docs.extend(docs)
        
        # Remover duplicados
        unique_docs = self._remove_duplicates(relevant_docs)
        
        # Calcular similitud para cada documento
        matches = []
        for doc in unique_docs:
            similarity_score, match_reasons = self._calculate_similarity(claim_data, doc)
            
            if similarity_score > 0.3:  # Umbral mínimo de similitud
                match = JurisprudenceMatch(
                    document=doc,
                    similarity_score=similarity_score,
                    match_reasons=match_reasons,
                    relevant_articles=self._extract_relevant_articles(doc),
                    case_summary=self._generate_case_summary(doc),
                    outcome=self._determine_outcome(doc)
                )
                matches.append(match)
        
        # Ordenar por similitud y retornar los mejores
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches[:limit]
    
    def _generate_search_queries(self, claim_data: ClaimData) -> List[str]:
        """Genera queries de búsqueda basadas en los datos del reclamo"""
        queries = []
        
        # Query básica con producto y problema
        if claim_data.product_type and claim_data.problem_description:
            queries.append(f"{claim_data.product_type} {claim_data.problem_description}")
        
        # Query por categoría de producto
        product_category = self._get_product_category(claim_data.product_type)
        if product_category:
            queries.append(f"{product_category} defectuoso")
        
        # Query por tipo de problema
        problem_type = self._get_problem_type(claim_data.problem_description)
        if problem_type:
            queries.append(f"{problem_type} {claim_data.product_type or 'producto'}")
        
        # Query por empresa
        if claim_data.purchase_place:
            company = self._normalize_company_name(claim_data.purchase_place)
            queries.append(f"{company} {claim_data.product_type or 'producto'}")
        
        # Query por monto si está disponible
        if claim_data.purchase_amount:
            amount_range = self._get_amount_range(claim_data.purchase_amount)
            queries.append(f"indemnización {amount_range}")
        
        # Query genérica de respaldo
        queries.append("producto defectuoso consumidor")
        
        return queries[:3]  # Limitar a 3 queries más relevantes
    
    def _calculate_similarity(self, claim_data: ClaimData, doc: Document) -> Tuple[float, List[str]]:
        """
        Calcula similitud entre el reclamo y un documento de jurisprudencia
        
        Returns:
            Tuple de (score, razones_del_match)
        """
        score = 0.0
        reasons = []
        
        doc_text = doc.page_content.lower()
        doc_metadata = doc.metadata
        
        # Similitud por producto (peso: 30%)
        if claim_data.product_type:
            product_score = self._calculate_product_similarity(claim_data.product_type, doc_text)
            score += product_score * 0.3
            if product_score > 0.5:
                reasons.append(f"Producto similar: {claim_data.product_type}")
        
        # Similitud por problema (peso: 40%)
        if claim_data.problem_description:
            problem_score = self._calculate_problem_similarity(claim_data.problem_description, doc_text)
            score += problem_score * 0.4
            if problem_score > 0.5:
                reasons.append(f"Problema similar: {claim_data.problem_description}")
        
        # Similitud por empresa (peso: 15%)
        if claim_data.purchase_place:
            company_score = self._calculate_company_similarity(claim_data.purchase_place, doc_text)
            score += company_score * 0.15
            if company_score > 0.5:
                reasons.append(f"Empresa similar: {claim_data.purchase_place}")
        
        # Similitud por artículos mencionados (peso: 10%)
        article_score = self._calculate_article_similarity(doc_text, doc_metadata)
        score += article_score * 0.1
        if article_score > 0.5:
            reasons.append("Artículos legales relevantes")
        
        # Similitud por resultado (peso: 5%)
        outcome_score = self._calculate_outcome_similarity(doc_text)
        score += outcome_score * 0.05
        if outcome_score > 0.5:
            reasons.append("Resultado favorable al consumidor")
        
        return min(score, 1.0), reasons
    
    def _calculate_product_similarity(self, product: str, doc_text: str) -> float:
        """Calcula similitud por producto"""
        product_lower = product.lower()
        
        # Coincidencia exacta
        if product_lower in doc_text:
            return 1.0
        
        # Coincidencia por categoría
        category = self._get_product_category(product)
        if category:
            category_words = self.product_categories[category]
            matches = sum(1 for word in category_words if word in doc_text)
            return min(matches / len(category_words), 1.0)
        
        return 0.0
    
    def _calculate_problem_similarity(self, problem: str, doc_text: str) -> float:
        """Calcula similitud por tipo de problema"""
        problem_lower = problem.lower()
        
        # Coincidencia directa
        if problem_lower in doc_text:
            return 1.0
        
        # Coincidencia por tipo de problema
        for problem_type, keywords in self.problem_types.items():
            if any(keyword in problem_lower for keyword in keywords):
                matches = sum(1 for keyword in keywords if keyword in doc_text)
                if matches > 0:
                    return min(matches / len(keywords), 1.0)
        
        return 0.0
    
    def _calculate_company_similarity(self, company: str, doc_text: str) -> float:
        """Calcula similitud por empresa"""
        company_lower = company.lower()
        
        # Coincidencia exacta
        if company_lower in doc_text:
            return 1.0
        
        # Coincidencia por empresa conocida
        normalized_company = self._normalize_company_name(company)
        if normalized_company in doc_text:
            return 0.8
        
        return 0.0
    
    def _calculate_article_similarity(self, doc_text: str, doc_metadata: Dict) -> float:
        """Calcula similitud por artículos legales mencionados"""
        # Buscar artículos de la ley 19.496 en el texto
        article_pattern = r"art[íi]culo\s*(\d+)"
        articles = re.findall(article_pattern, doc_text, re.IGNORECASE)
        
        # Artículos importantes de la ley del consumidor
        important_articles = ["3", "12", "19", "20", "21", "23", "24", "25", "26", "27"]
        
        if articles:
            relevant_articles = [art for art in articles if art in important_articles]
            return min(len(relevant_articles) / len(important_articles), 1.0)
        
        return 0.0
    
    def _calculate_outcome_similarity(self, doc_text: str) -> float:
        """Calcula similitud por resultado favorable"""
        favorable_indicators = [
            "acoge", "demanda acogida", "favorable", "indemnización", "compensación",
            "devolución", "reembolso", "condena", "multa", "sanción"
        ]
        
        matches = sum(1 for indicator in favorable_indicators if indicator in doc_text)
        return min(matches / len(favorable_indicators), 1.0)
    
    def _get_product_category(self, product: str) -> Optional[str]:
        """Obtiene la categoría de un producto"""
        if not product:
            return None
            
        product_lower = product.lower()
        
        for category, products in self.product_categories.items():
            if any(prod in product_lower for prod in products):
                return category
        
        return None
    
    def _get_problem_type(self, problem: str) -> Optional[str]:
        """Obtiene el tipo de problema"""
        if not problem:
            return None
            
        problem_lower = problem.lower()
        
        for problem_type, keywords in self.problem_types.items():
            if any(keyword in problem_lower for keyword in keywords):
                return problem_type
        
        return None
    
    def _normalize_company_name(self, company: str) -> str:
        """Normaliza el nombre de la empresa"""
        if not company:
            return ""
            
        company_lower = company.lower()
        
        for normalized, variants in self.known_companies.items():
            if any(variant in company_lower for variant in variants):
                return normalized
        
        return company_lower
    
    def _get_amount_range(self, amount: str) -> str:
        """Obtiene el rango de monto"""
        if not amount:
            return "medio"
        
        # Extraer número del string
        amount_num = re.findall(r'\d+', amount.replace('.', '').replace(',', ''))
        if not amount_num:
            return "medio"
        
        amount_value = int(amount_num[0])
        
        for range_name, (min_val, max_val) in self.amount_ranges.items():
            if min_val <= amount_value < max_val:
                return range_name
        
        return "medio"
    
    def _remove_duplicates(self, docs: List[Document]) -> List[Document]:
        """Remueve documentos duplicados"""
        seen_content = set()
        unique_docs = []
        
        for doc in docs:
            # Usar los primeros 100 caracteres como identificador
            content_id = doc.page_content[:100]
            if content_id not in seen_content:
                seen_content.add(content_id)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _extract_relevant_articles(self, doc: Document) -> List[str]:
        """Extrae artículos relevantes mencionados en el documento"""
        doc_text = doc.page_content
        
        # Buscar menciones de artículos
        article_patterns = [
            r"art[íi]culo\s*(\d+)",
            r"art\.\s*(\d+)",
            r"artículo\s*(\d+)°?"
        ]
        
        articles = []
        for pattern in article_patterns:
            matches = re.findall(pattern, doc_text, re.IGNORECASE)
            articles.extend(matches)
        
        # Remover duplicados y ordenar
        unique_articles = sorted(list(set(articles)), key=int)
        return [f"Artículo {art}" for art in unique_articles]
    
    def _generate_case_summary(self, doc: Document) -> str:
        """Genera un resumen del caso"""
        content = doc.page_content
        
        # Tomar las primeras 300 caracteres o hasta el primer punto
        summary = content[:300]
        
        # Buscar el final de la primera oración completa
        first_sentence_end = summary.find('. ')
        if first_sentence_end > 100:  # Solo si hay una oración razonablemente larga
            summary = summary[:first_sentence_end + 1]
        
        return summary.strip()
    
    def _determine_outcome(self, doc: Document) -> str:
        """Determina el resultado del caso"""
        content = doc.page_content.lower()
        
        # Indicadores de resultado favorable
        favorable_indicators = [
            "acoge", "demanda acogida", "favorable", "indemnización otorgada",
            "condena", "multa", "sanción", "devolución ordenada"
        ]
        
        # Indicadores de resultado desfavorable
        unfavorable_indicators = [
            "rechaza", "demanda rechazada", "desfavorable", "absuelve",
            "no lugar", "improcedente"
        ]
        
        favorable_score = sum(1 for indicator in favorable_indicators if indicator in content)
        unfavorable_score = sum(1 for indicator in unfavorable_indicators if indicator in content)
        
        if favorable_score > unfavorable_score:
            return "favorable"
        elif unfavorable_score > favorable_score:
            return "desfavorable"
        else:
            return "parcial"
    
    def get_match_summary(self, matches: List[JurisprudenceMatch]) -> str:
        """Genera un resumen de los matches encontrados"""
        if not matches:
            return "No se encontraron casos similares en la jurisprudencia."
        
        summary = f"Se encontraron {len(matches)} casos similares:\n\n"
        
        for i, match in enumerate(matches, 1):
            summary += f"{i}. **Caso {match.document.metadata.get('Rol', 'N/A')}**\n"
            summary += f"   - Similitud: {match.similarity_score:.1%}\n"
            summary += f"   - Razones: {', '.join(match.match_reasons)}\n"
            summary += f"   - Resultado: {match.outcome}\n"
            summary += f"   - Resumen: {match.case_summary[:150]}...\n\n"
        
        return summary