import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from google import genai  

def answer_with_rag(query_text, k=3):
    # 1. Vérification et chargement des chunks
    chunks_path = "index/chunks.jsonl"
    index_path = "index/vector.index"
    
    if not os.path.exists(chunks_path) or not os.path.exists(index_path):
        print("Erreur : L'index ou les chunks sont introuvables. Exécutez embed.py d'abord.")
        return

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    # 2. Charger l'index FAISS et le modèle d'embedding
    index = faiss.read_index(index_path)
    model = SentenceTransformer("intfloat/multilingual-e5-small")

    # 3. Encoder la requête et chercher les chunks pertinents
    formatted_query = f"query: {query_text}"
    query_vector = model.encode([formatted_query], normalize_embeddings=True)
    query_vector = np.array(query_vector).astype("float32")

    distances, indices = index.search(query_vector, k)

    # 4. Construire le contexte à partir des extraits trouvés
    context_blocks = []
    sources_used = set()
    
    for i in range(k):
        idx = indices[0][i]
        chunk = chunks[idx]
        context_blocks.append(f"[{chunk['chunk_id']} — Section : {chunk['section']}]\n{chunk['text']}")
        sources_used.add(chunk['chunk_id'])

    context_str = "\n\n".join(context_blocks)

    # 5. Préparer le prompt d'augmentation
    prompt = f"""Tu es un assistant rigoureux. Réponds à la question posée en t'appuyant UNIQUEMENT sur les extraits de documents fournis ci-dessous. 
Si la réponse ne figure pas dans les extraits, dis clairement que tu ne sais pas. Cite tes sources en mentionnant l'identifiant du chunk.

--- EXTRAITS DES DOCUMENTS ---
{context_str}
-----------------------------

Question : {query_text}
"""

    print(f"\n--- Question posée : {query_text} ---\n")
    print(f"Chunks récupérés : {list(sources_used)}")
    print("Génération de la réponse en cours...\n")

 # 6. Appel au LLM (Gemini)
    client = genai.Client()
    response = client.models.generate_content(
        model='gemini-3.8-flash',
        contents=prompt,
    )
    print("=== Réponse du RAG ===")
    print(response.text)

if __name__ == "__main__":
    # Test 
    question = "Donner moi la source des couches utiliser dans le projet?"
    answer_with_rag(question, k=3)