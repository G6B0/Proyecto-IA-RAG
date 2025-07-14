import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import Config

class DataLoader:
    """Cargador de datos para artículos legales y fallos judiciales"""
    
    def __init__(self):
        self.config = Config()
        
    def load_law_documents(self) -> List[Document]:
        """
        Carga y procesa los artículos de la ley desde CSV
        
        Returns:
            List[Document]: Lista de documentos procesados
        """
        try:
            df = pd.read_csv(self.config.LAW_FILE)
            return self._process_law_documents(df)
        except Exception as e:
            print(f"Error cargando artículos de ley: {e}")
            return []
    
    def load_case_documents(self) -> List[Document]:
        """
        Carga y procesa los fallos judiciales desde CSV
        
        Returns:
            List[Document]: Lista de documentos procesados
        """
        try:
            df = pd.read_csv(self.config.CASES_FILE)
            return self._process_case_documents(df)
        except Exception as e:
            print(f"Error cargando fallos judiciales: {e}")
            return []
    
    def _process_law_documents(self, df: pd.DataFrame) -> List[Document]:
        """
        Procesa los artículos de ley en documentos con chunking
        
        Args:
            df: DataFrame con los artículos de ley
            
        Returns:
            List[Document]: Documentos procesados
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE_LAW,
            chunk_overlap=self.config.CHUNK_OVERLAP_LAW,
            add_start_index=True
        )
        
        documents = []
        
        for index, row in df.iterrows():
            texto_articulo = row['Texto_articulo']
            articulo = str(row['Articulo'])
            
            # Si el artículo es largo, dividirlo
            if len(texto_articulo) > self.config.CHUNK_SIZE_LAW:
                splits = splitter.split_text(texto_articulo)
                for split in splits:
                    documents.append(
                        Document(
                            page_content=split,
                            metadata={'Articulo': articulo, 'tipo': 'ley'}
                        )
                    )
            else:
                documents.append(
                    Document(
                        page_content=texto_articulo,
                        metadata={'Articulo': articulo, 'tipo': 'ley'}
                    )
                )
        
        print(f"Cargados {len(documents)} documentos de artículos legales")
        return documents
    
    def _process_case_documents(self, df: pd.DataFrame) -> List[Document]:
        """
        Procesa los fallos judiciales en documentos con chunking
        
        Args:
            df: DataFrame con los fallos judiciales
            
        Returns:
            List[Document]: Documentos procesados
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE_CASES,
            chunk_overlap=self.config.CHUNK_OVERLAP_CASES,
            add_start_index=True
        )
        
        documents = []
        
        for index, row in df.iterrows():
            texto_sentencia = row['Texto_sentencia']
            
            # Extraer metadata
            metadata = {
                'Rol': row['Rol'],
                'Fecha_Sentencia': str(row['Fecha_Sentencia']),
                'Corte_origen': row['Corte de origen'],
                'Leyes_mencionadas': row['Leyes_mencionadas'],
                'tipo': 'fallo'
            }
            
            # Procesar texto
            if len(texto_sentencia) > 1500:  # Umbral para dividir
                splits = splitter.split_text(texto_sentencia)
                for split in splits:
                    documents.append(
                        Document(
                            page_content=split,
                            metadata=metadata
                        )
                    )
            else:
                documents.append(
                    Document(
                        page_content=texto_sentencia,
                        metadata=metadata
                    )
                )
        
        print(f"Cargados {len(documents)} documentos de fallos judiciales")
        return documents
    
    def load_all_documents(self) -> tuple[List[Document], List[Document]]:
        """
        Carga todos los documentos
        
        Returns:
            tuple: (documentos_ley, documentos_fallos)
        """
        print("Cargando documentos...")
        
        if not self.config.validate_files():
            raise FileNotFoundError("No se encontraron los archivos de datos")
        
        law_docs = self.load_law_documents()
        case_docs = self.load_case_documents()
        
        print(f"Total: {len(law_docs)} artículos, {len(case_docs)} fallos")
        
        return law_docs, case_docs