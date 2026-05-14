from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore

# GROQ IMPORT
from langchain_groq import ChatGroq

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# LOAD ENV
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# EMBEDDINGS
embeddings = download_hugging_face_embeddings()

# PINECONE INDEX
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# RETRIEVER
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# GROQ MODEL
chatModel = ChatGroq(
    model_name="llama-3.1-8b-instant"
)

# PROMPT
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# RAG CHAIN
question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

# HOME ROUTE
@app.route("/")
def index():
    return render_template('chat.html')

# CHAT ROUTE
@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form["msg"]

    print("User:", msg)

    response = rag_chain.invoke(
        {"input": msg}
    )

    print("Bot:", response["answer"])

    return str(response["answer"])

# RUN APP
if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )