import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from prototipo import get_response
load_dotenv()


app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(message: dict):
    user_message = message.get('message')
    if not user_message:
        return JSONResponse(content={"error": "Mensagem não fornecida"}, status_code=400)

    response_text = await get_response(user_message)
    return JSONResponse(content={"response": response_text})

# Executa o servidor
