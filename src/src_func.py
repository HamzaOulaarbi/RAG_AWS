
# try : 
from langchain.text_splitter import RecursiveCharacterTextSplitter #to be confirmed
from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama
# from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import get_embedding_func
# from operator import itemgetter
from PyPDF2 import PdfReader
import boto3
import os
from dotenv import load_dotenv




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split _document
def split_documents(documents): #(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(documents)

def generate_db_faiss(text_chunks,model_choice):
    db = FAISS.from_texts(text_chunks, embedding=get_embedding_func.get_embedding_func(model_choice))
    folder_path=f"faiss_bdd__{model_choice}/"
    file_name=f"index_{model_choice}"
    db.save_local(
        index_name=file_name, 
        folder_path=folder_path)
#   ## upload to S3
#     s3_client = boto3.client("s3")
#     # # BUCKET_NAME = os.getenv("BUCKET_NAME")
#     BUCKET_NAME="s3forrag"
#     # s3_client.upload_file(Filename=folder_path + "/" + f"{file_name}.faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
#     # s3_client.upload_file(Filename=folder_path + "/" + f"{file_name}.pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")
#     s3_client.upload_file(Filename=folder_path + "/" + f"{file_name}.faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
#     s3_client.upload_file(Filename=folder_path + "/" + f"{file_name}.pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")
    
    return db

def prompt_template_func(): 
    template = """
    Répondez en français à la question en vous basant sur le contexte ci-dessous. 
    Si vous ne pouvez pas répondre à la question, répondez "Je ne sais pas".
    Contexte : {context}
    Question : {question}
    """
    prompt = PromptTemplate.from_template(template)
    # prompt.format(context="Here is some context", question="Here is a question")
    return prompt

# # chain_structure
# def my_rag_chain(db, model_choice):
#     # model_str="llama2"
#     # model_str="mistral"
#     model=Ollama(model=model_choice)
#     # call retriever 
#     retriever=db.as_retriever(search_type="similarity", search_kwargs={'k': 3})
#     # call Parser StrOutputParser
#     parser = StrOutputParser()
#     chain = (
#     {
#         "context": itemgetter("question") | retriever,
#         "question": itemgetter("question"),
#     }
#     | prompt_template_func() # prompt
#     | model
#     # | parser
#     )
#     # Search the DB.
#     return chain

