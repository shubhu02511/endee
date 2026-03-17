import os
from sentence_transformers import SentenceTransformer
import endee
from endee.schema import VectorItem
# Monkeypatch official Endee SDK bug where it expects a dict instead of a Pydantic model internally
if not hasattr(VectorItem, 'get'):
    VectorItem.get = lambda self, key, default=None: getattr(self, key, default)

# 1. Initialize the SentenceTransformer model
# We use a small, fast model that produces 384-dimensional vectors
print("Loading sentence-transformers model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_endee_client():
    """Connects to Endee server and handles setup."""
    print("Connecting to Endee server at localhost:8080...")
    # By default, Endee Python client connects to http://localhost:8080/api/v1
    client = endee.Endee()
    return client

def setup_index(client, index_name="semantic_search_index", dimension=384):
    """Creates the index if it doesn't exist, else returns the existing one."""
    print(f"Ensuring index '{index_name}' exists...")
    try:
        # We try to create the index
        client.create_index(
            name=index_name,
            dimension=dimension,
            space_type="cosine",
            precision=endee.Precision.INT8
        )
        print("Index created successfully.")
    except Exception as e:
        # If the index already exists, the SDK might throw an error or we can just fetch it
        print(f"Index creation note (it may already exist): {e}")

    return client.get_index(name=index_name)

def main():
    # Documents to index
    documents = [
        {"id": "doc1", "text": "Endee is a high-performance open-source vector database built for AI search."},
        {"id": "doc2", "text": "Retrieval-Augmented Generation (RAG) improves LLM responses by adding context."},
        {"id": "doc3", "text": "Sentence transformers are great for creating text embeddings."},
        {"id": "doc4", "text": "Agentic workflows allow AI to use tools and plan steps before answering."},
        {"id": "doc5", "text": "Docker makes it easy to deploy applications inside containers."}
    ]

    print("Encoding documents into vectors...")
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts)

    # Convert to format required for Endee
    vector_items = []
    for doc, emb in zip(documents, embeddings):
        vector_items.append({
            "id": doc["id"],
            "vector": emb.tolist(),
            "meta": {"text": doc["text"]}
        })

    # Connect to Endee
    client = create_endee_client()
    index = setup_index(client, index_name="semantic_search_index", dimension=384)
    
    print(f"Upserting {len(vector_items)} documents into Endee...")
    index.upsert(vector_items)
    print("Documents successfully indexed!")
    
    # Now let's perform a Semantic Search
    query_text = "What is a vector database used for?"
    print(f"\nPerforming search for query: '{query_text}'")
    query_embedding = model.encode(query_text).tolist()
    
    results = index.query(vector=query_embedding, top_k=2)
    
    print("\n--- Search Results ---")
    for res in results:
        if isinstance(res, dict):
            res_id = res.get('id')
            res_sim = res.get('similarity', 0.0)
            meta = res.get('meta', {})
        else:
            res_id = getattr(res, 'id', 'Unknown')
            res_sim = getattr(res, 'similarity', 0.0)
            meta = getattr(res, 'meta', {})
            
        print(f"ID: {res_id}, Similarity: {res_sim:.4f}")
        
        if isinstance(meta, dict):
            text = meta.get('text', 'No text available')
        elif meta is not None:
            text = getattr(meta, 'text', str(meta))
        else:
            text = 'No text available'
            
        print(f"Text: {text}\n")

if __name__ == "__main__":
    main()
