import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Adicione esta linha
import asyncio
import numpy as np  # Importa numpy para manipulação de arrays
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore  # Importa o InMemoryDocstore
import faiss  # Importa o FAISS


#==========================================================================================
#                                 Carregando arquivo
#==========================================================================================
def carregar_dados_json(arquivo):
    with open(arquivo, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    return dados

# Carrega os dados do arquivo JSON
dados_api = carregar_dados_json('groq/api_data.json')
print("Dados carregados.")  # Verificação

#==========================================================================================
#                                 Concatenando JSON
#==========================================================================================

def concatenar_campos_json(dados):
    valores = []
    if isinstance(dados, list):
        for item in dados:
            if isinstance(item, dict):
                for chave, valor in item.items():
                    if isinstance(valor, str):
                        valores.append(valor)
                    elif isinstance(valor, (list, dict)):
                        valores.append(json.dumps(valor))
    elif isinstance(dados, dict):
        for chave, valor in dados.items():
            if isinstance(valor, str):
                valores.append(valor)
            elif isinstance(valor, (list, dict)):
                valores.append(json.dumps(valor))
    return ' '.join(valores)

# Concatena os campos
resultado_concatenado = concatenar_campos_json(dados_api)
print("Resultado concatenado.")  # Verificação

#==========================================================================================
#                                 CHUNKS
#==========================================================================================

def chunk_string(string, chunk_size):
    return [string[i:i + chunk_size] for i in range(0, len(string), chunk_size)]

chunk_size = 500
chunks = chunk_string(resultado_concatenado, chunk_size)
print(f"{len(chunks)} chunks gerados.")  # Verificação

# Exibir os chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}: {chunk[:30]}...")  # Exibe uma parte do chunk

#==========================================================================================
#                                 Embeddings
#==========================================================================================

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Primeiro elemento do model_output contém todos os embeddings de tokens
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Carregando o tokenizer e o modelo
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def generate_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        try:
            print(f"Processando chunk: {chunk[:30]}...")  # Exibe o início do chunk para depuração
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)  

            with torch.no_grad():
                model_output = model(**inputs)
                chunk_embedding = mean_pooling(model_output, inputs['attention_mask'])
                embeddings.append(chunk_embedding)

            print("Embedding gerado com sucesso.")  # Indica que o embedding foi gerado
        except Exception as e:
            print(f"Erro ao gerar embedding para o chunk: {chunk[:30]}...: {e}")

    return embeddings

# Gerar embeddings
embeddings = generate_embeddings(chunks)

# Verificar se a lista de embeddings não está vazia
if embeddings:
    print(f"Número de Embeddings Gerados: {len(embeddings)}")
else:
    print("Nenhum embedding foi gerado.")

# Converte os embeddings para um formato que o FAISS pode usar
embedding_vectors = np.array([emb.detach().numpy() for emb in embeddings]).squeeze()  # Converte os tensores para numpy e remove dimensões extras

#==========================================================================================
#                                 Configurando FAISS
#==========================================================================================

# Criação do índice FAISS
index = faiss.IndexFlatL2(embedding_vectors.shape[1])  # Cria o índice FAISS com o número de dimensões dos embeddings

# Adiciona os embeddings ao índice FAISS
index.add(embedding_vectors)

# Configura o armazenamento de documentos
docstore = InMemoryDocstore()

# Cria o vetorstore do FAISS
vector_store = FAISS(
    embedding_function=lambda x: embedding_vectors[x],  # A função de embedding pode ser configurada aqui
    index=index,
    docstore=docstore,
    index_to_docstore_id={i: f"chunk_{i}" for i in range(len(chunks))}  # Mapeia o índice do documento para um ID
)

# Exibir os embeddings
for i, emb in enumerate(embeddings):
    print(f"Embedding do Chunk {i + 1}:")
    print(emb.numpy())  # Converte o tensor para numpy para visualização
    print()

#==========================================================================================
#                                 Funções de Recuperação
#==========================================================================================

# Função para gerar o embedding da pergunta do usuário
def generate_query_embedding(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        model_output = model(**inputs)
        return mean_pooling(model_output, inputs['attention_mask'])

# Função para recuperar os documentos relevantes do FAISS
def retrieve_documents(query_embedding, k=5):
    distances, indices = index.search(query_embedding.numpy(), k)  # 'k' é o número de documentos a serem recuperados
    return [chunks[i] for i in indices[0]]  # Retorna os textos correspondentes

#==========================================================================================
#                                 CHAT LLM
#==========================================================================================
load_dotenv()
async def get_response(user_input):
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.environ.get("api"),
    )

    
    # Gera o embedding da pergunta do usuário
    query_embedding = generate_query_embedding(user_input)

    # Recupera os documentos relevantes
    relevant_chunks = retrieve_documents(query_embedding)

    # Concatenando os textos relevantes para formar o contexto
    context = " ".join(relevant_chunks)

    # Criando as mensagens com o contexto
    messages = [
        SystemMessage(content="any way speak only english. use o context requered."),
        HumanMessage(content=context),  # Adiciona o contexto
        HumanMessage(content=user_input)  # Adiciona a pergunta do usuário
    ]

    parser = StrOutputParser()
    chain = llm | parser
    texto = chain.invoke(messages)
    
    return texto
