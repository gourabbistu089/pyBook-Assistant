from flask import Flask, render_template, jsonify, request
from src.helper import download_gemini_embeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
from flask import session

import os

# ------------------- Flask Setup -------------------
app = Flask(__name__)
load_dotenv()

# ------------------- Environment -------------------
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ------------------- Embeddings & Retriever -------------------
embeddings = download_gemini_embeddings()
index_name = "pybookreader"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ------------------- LLM Setup -------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

# ------------------- Prompt Template -------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# ------------------- Helper to format retrieved context -------------------
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# ------------------- Chain Definition -------------------
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "input": RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

# ------------------- Flask Routes -------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    user_msg = request.form["msg"]
    print(f"User: {user_msg}")

    try:
        response = main_chain.invoke(user_msg)
        print(f"AI: {response}")
        return str(response)
    except Exception as e:
        print("Error:", e)
        return "Sorry, something went wrong."


# ------------------- Run App -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
