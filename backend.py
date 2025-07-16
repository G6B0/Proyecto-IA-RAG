from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.legal_agent import LegalAgent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajusta según seguridad
    allow_methods=["*"],
    allow_headers=["*"],
)

class Pregunta(BaseModel):
    pregunta: str

agent = LegalAgent()
agent.initialize()  # Inicializa el agente una vez al arrancar el backend

@app.post("/ask")
def ask(pregunta: Pregunta):
    respuesta = agent.chat(pregunta.pregunta)
    print("Respuesta del agente:", respuesta)
    return respuesta

