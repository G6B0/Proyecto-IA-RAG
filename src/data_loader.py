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
    
    def load_case_documents(self, batch_size: int = 100) -> List[Document]:
        """
        Carga y procesa los fallos judiciales desde CSV por lotes
        
        Args:
            batch_size: Número de filas del CSV a procesar por lote
            
        Returns:
            List[Document]: Lista de documentos procesados
        """
        try:
            df = pd.read_csv(self.config.CASES_FILE)
            return self._process_case_documents_in_batches(df, batch_size)
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
    
    def _process_case_documents_in_batches(self, df: pd.DataFrame, batch_size: int) -> List[Document]:
        """
        Procesa los fallos judiciales en documentos usando chunks pre-existentes por lotes
        
        Args:
            df: DataFrame con los fallos judiciales (texto ya dividido en chunks)
            batch_size: Número de filas del CSV a procesar por lote
            
        Returns:
            List[Document]: Documentos procesados
        """
        import ast  # Para convertir strings que representan listas
        
        all_documents = []
        total_rows = len(df)
        
        print(f"Procesando {total_rows} casos en lotes de {batch_size}")
        
        # Procesar DataFrame en lotes
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_df = df.iloc[batch_start:batch_end]
            
            print(f"Procesando lote {batch_start//batch_size + 1}: filas {batch_start} a {batch_end}")
            
            batch_documents = []
            
            for index, row in batch_df.iterrows():
                # Procesar el texto de la sentencia que viene como string de lista
                texto_sentencia_raw = row['Texto_sentencia']
                
                # Convertir de string de lista a lista real
                try:
                    if isinstance(texto_sentencia_raw, str) and texto_sentencia_raw.startswith('['):
                        # Es una lista en formato string, convertir a lista real
                        chunks_lista = ast.literal_eval(texto_sentencia_raw)
                    else:
                        # Si no es una lista, tratarlo como un solo chunk
                        chunks_lista = [str(texto_sentencia_raw)]
                except (ValueError, SyntaxError):
                    # Si falla la conversión, usar el texto tal como está
                    chunks_lista = [str(texto_sentencia_raw)]
                
                # Extraer metadata base
                metadata_base = {
                    'Rol': row['Rol'],
                    'Fecha_Sentencia': str(row['Fecha_Sentencia']),
                    'Corte_origen': row['Corte de origen'],
                    'Leyes_mencionadas': row['Leyes_mencionadas'],
                    'Articulos_mencionados': row['Artículos_mencionados'],
                    'tipo': 'fallo'
                }
                
                # Crear un documento por cada chunk
                for chunk_idx, chunk_text in enumerate(chunks_lista):
                    # Agregar índice del chunk a los metadatos
                    metadata = metadata_base.copy()
                    metadata['chunk_index'] = chunk_idx
                    metadata['total_chunks'] = len(chunks_lista)
                    
                    batch_documents.append(
                        Document(
                            page_content=chunk_text.strip(),
                            metadata=metadata
                        )
                    )
            
            all_documents.extend(batch_documents)
            print(f"Lote {batch_start//batch_size + 1} procesado: {len(batch_documents)} documentos")
        
        print(f"Total cargados: {len(all_documents)} documentos de fallos judiciales")
        return all_documents
    
    def _process_case_documents(self, df: pd.DataFrame) -> List[Document]:
        """
        Método original mantenido para compatibilidad
        """
        return self._process_case_documents_in_batches(df, batch_size=100)
    
    def load_all_documents(self, case_batch_size: int = 100) -> tuple[List[Document], List[Document]]:
        """
        Carga todos los documentos
        
        Args:
            case_batch_size: Tamaño del lote para procesar casos
            
        Returns:
            tuple: (documentos_ley, documentos_fallos)
        """
        print("Cargando documentos...")
        
        if not self.config.validate_files():
            raise FileNotFoundError("No se encontraron los archivos de datos")
        
        law_docs = self.load_law_documents()
        case_docs = self.load_case_documents(batch_size=case_batch_size)
        
        print(f"Total: {len(law_docs)} artículos, {len(case_docs)} fallos")
        
        return law_docs, case_docs