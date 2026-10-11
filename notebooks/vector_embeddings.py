import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()


openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is missing.")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is missing.")

embedding_model = "sentence-transformers/all-MiniLM-L12-v2"

embeddings = OpenAIEmbeddings(
    model=embedding_model,
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
)


documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Python is widely used for data science and AI applications.",
    "AI models can understand and generate human language.",
]


vector_documents = embeddings.embed_documents(documents)

print(f"Created {len(vector_documents)} embeddings")
print(f"Vector dimension: {len(vector_documents[0])}")

for i, vector in enumerate(vector_documents):
    print(f"\nDocument {i}:")
    print(vector)


pc = Pinecone(api_key=pinecone_api_key)

index = pc.Index("my-rag-index")



vectors = []

for i, vector in enumerate(vector_documents):

    vectors.append({
        "id": f"doc-{i}",
        "values": vector,
        "metadata": {
            "text": documents[i]
        }
    })



# index.upsert(vectors=vectors)

# print("\nVectors uploaded to Pinecone!")

query = "Machine Learning"

query_vector = embeddings.embed_query(query)

result = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True
)

print(result)

stats = index.describe_index_stats()

print("\nPinecone stats:")
print(stats)