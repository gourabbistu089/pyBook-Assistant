from src.helper import load_pdf_file, text_split, download_gemini_embeddings
import pinecone
from langchain_pinecone import Pinecone as PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

print("🔄 Loading PDF files...")
extracted_data = load_pdf_file(data='Data/')
print(f"✅ Loaded {len(extracted_data)} pages")

print("🔄 Splitting text...")
text_chunks = text_split(extracted_data)
print(f"✅ Created {len(text_chunks)} chunks")

print("🔄 Loading embeddings...")
embeddings = download_gemini_embeddings()

# Test embedding dimension
test_embedding = embeddings.embed_query("test")
embedding_dim = len(test_embedding)
print(f"📐 Embedding dimension: {embedding_dim}")

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

index_name = "pybookreader"
print(f"🔄 Working with index: {index_name}")

# Check existing indexes
existing_indexes = pc.list_indexes().names()
print(f"📋 Existing indexes: {existing_indexes}")

if index_name not in existing_indexes:
    print("🔄 Creating new index...")
    pc.create_index(
        name=index_name,
        dimension=embedding_dim,  # Actual dimension use karo
        metric="cosine",
        spec=pinecone.ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        )
    )
    print("✅ Index created!")
else:
    print("✅ Index already exists!")

print("🔄 Creating vector store...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

print("🎉 All done! Vector store ready.")



































#####################################################


# from src.helper import load_pdf_file, text_split, download_gemini_embeddings
# from pinecone import Pinecone 
# from pinecone import ServerlessSpec
# from langchain_pinecone import Pinecone
# from dotenv import load_dotenv
# import os

# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# # print(f"API Key: {PINECONE_API_KEY}") #debug

# extracted_data=load_pdf_file(data='Data/')
# text_chunks=text_split(extracted_data)
# embeddings = download_gemini_embeddings()

# pc = Pinecone(api_key=PINECONE_API_KEY)

# index_name = "pybookreader"
# # if not pc.Index(index_name):
# pc.create_index(
#     name=index_name,
#     dimension=768,
#     metric="cosine",
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1",
#     )
# )

# # Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = Pinecone.from_documents(
#     documents=text_chunks,
#     index_name=index_name,
#     embedding=embeddings
# )
