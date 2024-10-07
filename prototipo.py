import os
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Conectar ao cliente do Qdrant com timeout aumentado
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30  # Aumente o timeout para 30 segundos ou conforme necessário
)

#==========================================================================================
#                                 Funções de Recuperação no Qdrant
#==========================================================================================

def buscar_documentos_qdrant(query_embedding, k=5):
    query_vector = query_embedding.squeeze().numpy().tolist()
    response = qdrant_client.search(
        collection_name="Nasa",
        query_vector=query_vector,
        limit=k
    )
    return [hit.payload["text"] for hit in response]

# Função para gerar o embedding da pergunta do usuário
def generate_query_embedding(question):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        model_output = model(**inputs)
        return mean_pooling(model_output, inputs['attention_mask'])

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#==========================================================================================
#                                 Integração com LLM
#==========================================================================================

async def processar_pergunta(user_input, historico_mensagens):
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0.4,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.environ.get("api"),
    )

    # Gera o embedding da pergunta do usuário
    query_embedding = generate_query_embedding(user_input)

    # Recupera os documentos relevantes do Qdrant
    relevant_chunks = buscar_documentos_qdrant(query_embedding)

    # Concatenando os textos relevantes para formar o contexto
    context = " ".join(relevant_chunks)

    # Criando as mensagens com o contexto e histórico
    messages = [
        SystemMessage(content="use context only if specified, below in the message."),
        HumanMessage(content=context),  # Adiciona o contexto
    ] + historico_mensagens + [HumanMessage(content=user_input)]  # Adiciona o histórico e a pergunta atual

    parser = StrOutputParser()
    chain = llm | parser
    texto = chain.invoke(messages)

    return texto



async def get_response(user_input):
    historico_mensagens = []  # Lista para armazenar o histórico de mensagens

    # Adiciona a mensagem do usuário ao histórico de mensagens
    historico_mensagens.append(HumanMessage(content=user_input))
    
    # Processa a pergunta e obtém a resposta
    resposta = await processar_pergunta(user_input, historico_mensagens)
    
    # Adiciona a resposta ao histórico de mensagens
    historico_mensagens.append(HumanMessage(content=resposta))
    
    return resposta