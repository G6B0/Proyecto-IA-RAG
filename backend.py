from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.legal_agent import LegalAgent  # Asegúrate de que 'src' esté bien ubicado

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Pregunta(BaseModel):
    pregunta: str

agent = LegalAgent()
agent.initialize()

@app.post("/ask")
def responder(pregunta: Pregunta):
    resultado = agent.chat(pregunta.pregunta)
    return {
        "respuesta": resultado["answer"],
        "sources": resultado.get("sources", {})
    }

