import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Carrega as variáveis do arquivo .env
load_dotenv()

# Função para pegar a resposta do LLM e configura o cliente Groq
async def get_response(user_input):
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.environ.get("api"),  #congirurar no .env com a Chave API
    )

    messages = [
        SystemMessage(content="Você é um assistente que irá responder tudo em português."),
        HumanMessage(content=user_input)
    ]

    parser = StrOutputParser()

    chain = llm | parser

    texto = chain.invoke(messages)

    return texto
# python servidor.py