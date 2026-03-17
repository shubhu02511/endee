# Semantic Search with Endee Vector Database

This repository demonstrates a **Practical AI/ML use case** using the **Endee open-source vector database**. We have built a Semantic Search application in Python that converts text documents into dense vector embeddings using `sentence-transformers` and retrieves highly relevant results based on meaning rather than exact keyword matches.

## 🚀 Project Overview
The **Semantic Search** script (`semantic_search.py`) takes a small corpus of text documents, generates 384-dimensional embeddings for them, and stores these vectors in a local Endee database instance. When a user submits a search query, the system embeds the query into the same vector space, and Endee performs a high-speed cosine similarity search to retrieve the most contextually relevant documents.

## 🧠 System Design
The application relies on two main components:
1. **Embedding Model (`sentence-transformers/all-MiniLM-L6-v2`)**: A lightweight, fast, and high-quality model from Hugging Face that converts text sentences into 384-dimensional dense vectors.
2. **Vector Database (Endee)**: A high-performance storage Engine optimized for similarity search. Endee efficiently stores the document embeddings and their associated text metadata, returning nearest neighbors in milliseconds.

### Use of Endee
- **Endee Python SDK (`endee`)**: Used to programmatically create indexes, upsert vectors, and perform queries from within the Python script.
- **Index Dimensions**: Configured to `384` to match the output size of the chosen `SentenceTransformer` model.
- **Space Type**: Set to `cosine` for semantic similarity measuring.
- **Precision**: Uses `INT8` quantization for optimal memory efficiency without significant accuracy loss.

---

## 🛠️ Setup Instructions

### 1. Mandatory Prerequisites
Before proceeding, you must complete the following steps to ensure uniform evaluation:
1. **Star** the official Endee GitHub repository: https://github.com/endee-io/endee
2. **Fork** the repository to your personal GitHub account.
3. Keep this code hosted on your **forked** repository as your project base.

### 2. Install Docker Desktop (Required for Windows/Mac)
If you are running on Windows, **Docker Desktop is mandatory** to run the Endee server.
- Download and install [Docker Desktop for Windows (AMD64)](https://docs.docker.com/desktop/install/windows-install/) (or the corresponding version for your OS).
- Ensure the Docker Engine is running.

### 3. Start the Endee Server
Use Docker to start the Endee database server locally on port 8080:
```bash
docker run \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  --restart unless-stopped \
  endeeio/endee-server:latest
```

### 4. Install Python Dependencies
Ensure you have Python 3.8+ installed. Install the required libraries:
```bash
pip install -r requirements.txt
```
*(Dependencies include `endee`, `sentence-transformers`, and `torch`)*

### 5. Run the Semantic Search Application
```bash
python semantic_search.py
```

**Expected Output:**
The script will initialize the model, connect to the local Endee server, ensure the `semantic_search_index` exists, encode the sample documents, insert them, and finally execute a test semantic search query (e.g., "What is a vector database used for?").

You will see the closest matching documents logged in your terminal!
