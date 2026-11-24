import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def main():
    # 1. Charger les chunks depuis le fichier JSONL
    chunks_path = "index/chunks.jsonl"
    if not os.path.exists(chunks_path):
        print(f"Erreur : Le fichier {chunks_path} est introuvable. Exécutez d'abord l'ingestion.")
        return

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"{len(chunks)} chunks chargés pour la vectorisation.")

    # 2. Extraire les textes
    texts = [f"passage: {c['text']}" for c in chunks]

    # 3. Charger le modèle d'embedding multilingue
    model_name = "intfloat/multilingual-e5-small"
    print(f"Chargement du modèle d'embedding : {model_name}...")
    model = SentenceTransformer(model_name)

    # 4. Calculer les embeddings 
    print("Calcul des embeddings en cours...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    # 5. Créer l'index FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings)

    # 6. Sauvegarder l'index et les métadonnées associées
    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, "index/vector.index")
    
    # On sauvegarde aussi la liste des chunks pour pouvoir retrouver le texte à partir de l'indice du vecteur
    print(f"Succès ! Index FAISS sauvegardé avec {index.ntotal} vecteurs (dimension : {dimension}).")

if __name__ == "__main__":
    main()