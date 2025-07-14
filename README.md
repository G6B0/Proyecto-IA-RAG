# Proyecto RAG Legal - Ley del Consumidor en Chile

### Integrantes
- Integrante 1: [Gabriel Huerta]()
- Integrante 2: [Nicolás Soto](https://github.com/nesoto)
- Integrante 3: [Kaori Encina]()

## Descripción

Sistema de Recuperación Aumentada con Generación (RAG) que utiliza Ollama con Llama 3:8B para proporcionar asistencia legal especializada en la Ley 19.496 sobre Protección de los Derechos de los Consumidores de Chile. El sistema combina la ley con jurisprudencia relevante para generar respuestas contextualizadas y documentos legales.

## Características

- **Procesamiento de documentos legales**: Análisis de la Ley del Consumidor y fallos judiciales
- **Sistema RAG**: Recuperación inteligente de contexto legal relevante
- **Interfaz interactiva**: CLI para consultas legales
- **Memoria conversacional**: Mantiene contexto entre preguntas
- **Generación de documentos**: Capacidad para redactar documentos legales basados en la ley y jurisprudencia

## Estructura del Proyecto

```
Proyecto-IA-RAG/
├── data/                           # Archivos de datos
│   ├── Ley_consumidor_limpio.csv   # Ley 19.496 procesada
│   └── Fallos_judiciales_ley_19.496.csv  # Jurisprudencia
├── src/                            # Código fuente
│   ├── __init__.py
│   ├── config.py                   # Configuración central
│   ├── data_loader.py              # Carga y procesamiento de datos
│   ├── legal_agent.py              # Agente legal principal
│   └── rag_system.py               # Sistema RAG
├── main.py                         # Archivo a ejecutar
├── requirements.txt                # Dependencias
├── .gitignore
└── README.md                       # Este archivo
```

## Requisitos del Sistema

### Software Requerido

1. **Python 3.8+**
2. **Ollama** (para ejecutar modelos LLM localmente)

### Modelos de Ollama Necesarios

- `llama3:8b` - Modelo principal para generación de texto
- `nomic-embed-text` - Modelo para embeddings

## Instalación

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/user/Proyecto-IA-RAG.git
cd Proyecto-IA-RAG
```

### Paso 2: Instalar Ollama

#### Windows
```bash
# Descargar desde https://ollama.com/download/windows
# O usar winget:
winget install Ollama.Ollama
```

#### macOS
```bash
brew install ollama
```

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Paso 3: Descargar Modelos de Ollama

```bash
# Iniciar el servicio de Ollama, si ya esta corriendo, omitir este paso
ollama serve

# En otra terminal, descargar los modelos:
ollama pull llama3:8b
ollama pull nomic-embed-text
```

### Paso 4: Crear Entorno Virtual (Recomendado)

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Paso 5: Instalar Dependencias de Python

```bash
pip install -r requirements.txt
```

### Paso 6: Verificar Archivos de Datos

Asegúrate de que los siguientes archivos estén en la carpeta `data/`:
- `Ley_consumidor_limpio.csv`
- `Fallos_judiciales_ley_19.496.csv`

## Ejecución del Proyecto

### Ejecutar la Aplicación

```bash
python main.py
```

## Configuración

El archivo `src/config.py` contiene todas las configuraciones del sistema:

- **Modelos**: Puedes cambiar los modelos de Ollama
- **Chunking**: Ajustar tamaños de fragmentos para procesamiento
- **Retrieval**: Número de documentos a recuperar (K)
- **Prompts**: Personalizar los prompts del sistema

## Solución de Problemas

### Error: "Ollama no está corriendo"
```bash
# Iniciar Ollama:
ollama serve
```

### Error: "Modelo no encontrado"
```bash
# Verificar modelos instalados:
ollama list

# Si faltan, instalar:
ollama pull llama3:8b
ollama pull nomic-embed-text
```

### Error: "Archivos de datos no encontrados"
- Verificar que los archivos CSV estén en la carpeta `data/`
- Revisar que los nombres coincidan exactamente con los especificados en `config.py`

### Problemas de memoria
- Reducir `CHUNK_SIZE` en `config.py`
- Usar un modelo más pequeño (ej: `llama3:7b` en lugar de `llama3:8b`)

## Datos

El proyecto utiliza dos fuentes de datos principales:

1. **Ley del Consumidor**: Texto completo de la Ley 19.496
2. **Jurisprudencia**: Fallos judiciales relacionados con la ley

Los datos deben estar en formato CSV con la estructura esperada por el sistema.