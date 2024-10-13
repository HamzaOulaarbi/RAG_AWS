#Embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockEmbeddings
import boto3

import os
from dotenv import load_dotenv
import boto3
# Charger les variables d'environnement
load_dotenv()

# Obtenir les credentials AWS
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')

# Créer une session boto3
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)


def get_embedding_func(model_str):
    if model_str=="Bedrock_aws":
        bedrock_client = session.client(service_name="bedrock-runtime")
        bedrock_model_id="amazon.titan-embed-text-v1"
        bedrock_embeddings=BedrockEmbeddings(model_id=bedrock_model_id, client=bedrock_client)
        embeddings = bedrock_embeddings
    else : 
        # model_str="llama2"
        model_str='mistral'
        embeddings = OllamaEmbeddings(model=model_str)
    # embeddings = OpenAIEmbeddings()
    return embeddings

# # Charger les variables d'environnement
# from dotenv import load_dotenv
# load_dotenv()

# # Obtenir les credentials AWS
# aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
# aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
# aws_region = os.getenv('AWS_REGION')

# def get_embedding_func(model_str):
#     if model_str == "Bedrock_aws":
#         # Créer le client directement avec les credentials
#         bedrock_client = boto3.client(
#             service_name="bedrock-runtime",
#             aws_access_key_id=aws_access_key_id,
#             aws_secret_access_key=aws_secret_access_key,
#             region_name=aws_region
#         )
        
#         bedrock_model_id = "amazon.titan-embed-text-v1"
#         bedrock_embeddings = BedrockEmbeddings(model_id=bedrock_model_id, client=bedrock_client)
#         embeddings = bedrock_embeddings
#     else:
#         # Exemple avec un autre modèle
#         model_str = 'mistral'  # Remplace par le modèle désiré
#         embeddings = OllamaEmbeddings(model=model_str)
        
#     return embeddings

