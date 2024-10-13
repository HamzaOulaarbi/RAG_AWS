import src_func
import get_embedding_func
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
# from langchain.llms.bedrock import Bedrock
# from langchain_community.llms.bedrock import Bedrock
from langchain_aws import BedrockLLM
import boto3
import os
from dotenv import load_dotenv

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



# def lambda_handler(event, context):

#     print("Hello AWS!")
#     print("event = {}".format(event))
#     return {
#         'statusCode': 200,
#     }

st.set_page_config(page_title ="Ask PDFs", page_icon="books")
st.header("Ask PDFs :books:")
user_question = st.text_input("Ask a question about your documents:")
# if user_question:
#     handle_userinput(user_question)

with st.sidebar:
    st.title("Menu:")
    # Liste déroulante pour sélectionner le modèle
    model_choice = st.selectbox(
        "Choisissez le modèle à utiliser :",
        ("Bedrock_aws","Llama2", "Mistral") )
    # Affichage du choix sélectionné
    st.write(f"Vous avez sélectionné le modèle : {model_choice.lower()}")

    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
    if st.button("Submit & Process", key="process_button"): #and api_key:  # Check if API key is provided before processing
        with st.spinner("Processing..."):
            documents = src_func.get_pdf_text(pdf_docs)
            chunks =src_func.split_documents(documents)
            db=src_func.generate_db_faiss(chunks,model_choice)

if st.button("Process"):
    with st.spinner("Processing"):
        file_name=f"index_{model_choice}"
        folder_path=f"faiss_bdd__{model_choice}/"
        db = FAISS.load_local(index_name=file_name, 
                              folder_path=folder_path,  
                              embeddings=get_embedding_func.get_embedding_func(model_choice), allow_dangerous_deserialization=True)     
        
        # def load_db_from_S3(BUCKET_NAME):
        #     s3_client = boto3.client("s3")
        #     s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename="")
        #     s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename="")
        #     return db
        # db=load_db_from_S3(BUCKET_NAME="s3forrag")
        # st.write ("With chain approach")
        # answer=src_func.my_rag_chain(db,model_choice).invoke({'question': user_question})
        # st.write(answer)

        results = db.similarity_search_with_relevance_scores(user_question, k=3)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        st.write(f"Founded context :\n\n{context_text} ")
        PROMPT_TEMPLATE = """
        Répondez en français à la question en vous basant sur le contexte ci-dessous. 
        Si vous ne pouvez pas répondre à la question, répondez "Je ne sais pas".
        \nContexte : {context}
        \nQuestion : {question}\n\n
        """
        prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=user_question)
        # st.write(prompt)

        if model_choice =='Bedrock_aws':
            #llm bedrock anthropic.claude-
            # bedrock_client = boto3.client("bedrock-runtime")
            bedrock_client = session.client("bedrock-runtime")

            bedrock_llm_model_id="anthropic.claude-v2:1"
            model=BedrockLLM(model_id=bedrock_llm_model_id, client=bedrock_client, model_kwargs={'max_tokens_to_sample': 512})
        else : # for ollama model
            model=Ollama(model=model_choice)

        response_text = model.invoke(prompt)
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: \n\n {response_text}. \n\n Sources: {sources}"
        st.write(formatted_response)
        # st.write(answer.response_metadata)