import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def search(query_text, k=3):
    # 1. Charger les chunks
    chunks_path = "index/chunks.jsonl"
    if not os.path.exists(chunks_path):
        print("Erreur : Fichier chunks.jsonl introuvable.")
        return
    
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    # 2. Charger l'index FAISS
    index_path = "index/vector.index"
    if not os.path.exists(index_path):
        print("Erreur : Index FAISS introuvable. Lancez d'abord embed.py.")
        return
    
    index = faiss.read_index(index_path)

    # 3. Charger le modèle d'embedding (le même que pour l'indexation)
    model_name = "intfloat/multilingual-e5-small"
    model = SentenceTransformer(model_name)

    # 4. Encoder la requête avec le préfixe 
    formatted_query = f"query: {query_text}"
    query_vector = model.encode([formatted_query], normalize_embeddings=True)
    query_vector = np.array(query_vector).astype("float32")

    # 5. Rechercher dans l'index FAISS les k plus proches voisins
    distances, indices = index.search(query_vector, k)

    print(f"\n--- Résultats pour la question : « {query_text} » ---\n")
    
    for i in range(k):
        idx = indices[0][i]
        score = distances[0][i]
        chunk = chunks[idx]
        
        print(f"Résultat {i+1} (Score :ansa {score:.4f})")
        print(f"Source : {chunk['chunk_id']} | Section : {chunk['section']}")
        print(f"Extrait : {chunk['text'][:300]}...")
        print("-" * 50)

if __name__ == "__main__":
    question = "Comment fonctionne le découpage des documents ?"
    search(question, k=3)