# Hybrid RAG

Assistant de recherche documentaire combinant la recherche vectorielle avec
**FAISS**, des embeddings multilingues de **Sentence Transformers** et la
génération augmentée de contexte via **Google Gemini**.

Ce dépôt indexe les documents Markdown du projet géospatial « Accès aux soins –
région Casablanca-Settat », puis permet d'interroger leurs contenus avec des
réponses contextualisées et citables.

## Fonctionnalités

- Découpage automatique des documents Markdown en sections et blocs.
- Création de chunks avec identifiants, métadonnées et provenance.
- Encodage multilingue avec `intfloat/multilingual-e5-small`.
- Recherche sémantique par similarité cosinus via un index FAISS.
- Génération de réponses avec Gemini à partir exclusivement des extraits retrouvés.
- Citations des sources sous la forme `document#section.chunk`.
- Recherche locale disponible sans appel au modèle génératif.

## Architecture

```text
data/docs/*.md
      |
      v
rag/ingest.py       Découpe et génération des chunks
      |
      v
index/chunks.jsonl  Métadonnées et textes indexables
      |
      v
rag/embed.py        Création des embeddings
      |
      v
index/vector.index  Index FAISS
      |
      +--> rag/search.py   Recherche sémantique uniquement
      |
      +--> rag/generate.py Génération RAG avec Gemini
```

## Structure du projet

```text
hybrid-rag/
├── data/
│   └── docs/
│       ├── README_CADRAGE.md
│       └── README_COLLECTE.md
├── index/
│   ├── chunks.jsonl
│   └── vector.index
├── rag/
│   ├── ingest.py
│   ├── embed.py
│   ├── search.py
│   └── generate.py
└── README.md
```

## Prérequis

- Windows 10/11
- Python 3.11
- Dépendances Python :
  - `numpy`
  - `sentence-transformers`
  - `faiss-cpu`
  - `google-genai`
- Une clé API Google pour la génération avec Gemini

## Installation

Depuis un terminal PowerShell ouvert à la racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy sentence-transformers faiss-cpu google-genai
```

Configurez ensuite la clé API Gemini :

```powershell
$env:GOOGLE_API_KEY = "VOTRE_CLE_API"
```

Pour conserver la variable dans les variables d'environnement utilisateur :

```powershell
setx GOOGLE_API_KEY "VOTRE_CLE_API"
```

Ouvrez ensuite un nouveau terminal afin que la nouvelle variable soit disponible.
Ne commitez jamais une clé API dans le dépôt.

## Utilisation

Toutes les commandes doivent être exécutées depuis la racine du projet.

### 1. Découper les documents

```powershell
python rag\ingest.py
```

La commande lit tous les fichiers `*.md` de `data/docs` et produit
`index/chunks.jsonl`. Les sections ne sont pas découpées arbitrairement :
le script préserve les paragraphes, listes, tableaux et blocs de code lorsque
possible, avec une limite cible de 200 mots par chunk.

### 2. Construire l'index vectoriel

```powershell
python rag\embed.py
```

Cette étape charge le modèle d'embedding, calcule les vecteurs normalisés puis
crée `index/vector.index` avec une recherche FAISS basée sur le produit scalaire
(`IndexFlatIP`).

### 3. Rechercher les documents pertinents

```powershell
python rag\search.py
```

La recherche affiche les meilleurs chunks et leur identifiant. Pour modifier la
question ou le nombre de résultats, adaptez l'appel final dans `rag/search.py`.

### 4. Générer une réponse RAG

```powershell
python rag\generate.py
```

Le script interroge l'index, fournit les trois chunks les plus proches au
modèle Gemini et demande une réponse fondée uniquement sur ces extraits.

Pour poser une autre question, modifiez l'appel final dans `rag/generate.py` :

```python
answer_with_rag("Votre question ici", k=3)
```

## Pipeline RAG

1. Un document est découpé en sections puis en blocs textuels.
2. Chaque bloc est enregistré dans `chunks.jsonl` avec son titre, sa section,
   son fichier source et son identifiant.
3. Les chunks sont encodés avec le préfixe `passage:`.
4. Une question utilisateur est encodée avec le préfixe `query:`.
5. FAISS retourne les vecteurs les plus proches.
6. Les textes associés sont injectés dans un prompt restrictif.
7. Gemini répond en citant les identifiants des chunks utilisés.

## Modèle et paramètres actuels

| Élément | Valeur |
| --- | --- |
| Embedding | `intfloat/multilingual-e5-small` |
| Index | FAISS `IndexFlatIP` |
| Normalisation | Oui |
| Taille maximale cible | 200 mots par chunk |
| Résultats par défaut | 3 |
| Modèle génératif configuré | `gemini-3.8-flash` |

Le nom du modèle génératif doit correspondre à un modèle disponible pour votre
compte Google AI / votre projet Gemini. Modifiez la propriété `model` dans
`rag/generate.py` si nécessaire.

## Performances et limites

- FAISS utilise une recherche exacte en mémoire ; elle convient à ce prototype
  et aux corpus de taille modérée.
- Les réponses restent limitées au contexte récupéré : les informations absentes
  des documents doivent être signalées comme inconnues.
- La qualité dépend de la pertinence du découpage et du modèle d'embedding.
- Le modèle génératif peut varier selon son accès, sa disponibilité et la région.
- Le contexte récupéré est ajouté au prompt avec ses identifiants, mais il n'y a pas
  encore de journalisation structurée des sources utilisées.

## Dépannage

### `GOOGLE_API_KEY` est absente

Définissez `GOOGLE_API_KEY` dans le terminal ou passez explicitement une clé au
constructeur `genai.Client(api_key="...")` sans enregistrer cette valeur dans le
code.

### Fichiers indexés introuvables

Exécutez successivement :

```powershell
python rag\ingest.py
python rag\embed.py
```

Puis relancez la recherche ou la génération.

### Problème de chargement du modèle Sentence Transformers

Vérifiez la connexion Internet lors du premier téléchargement du modèle et
les permissions d'accès au cache Hugging Face.

### Erreur liée au modèle Gemini

Vérifiez que le nom configuré dans `rag/generate.py` est disponible pour votre
clé API et que cette clé est active.

## Améliorations possibles

- Ajouter un système d'arguments en ligne de commande à tous les scripts.
- Externaliser les noms de modèles, chemins et paramètres dans un fichier de
  configuration.
- Ajouter un fichier `requirements.txt` ou `pyproject.toml` reproductible.
- Mettre en place des tests unitaires et des évaluations de recherche.
- Afficher directement les extraits et liens vers les documents sources.
- Ajouter du reranking et une recherche hybride lexical/vectorielle.
- Ajouter un cache pour les embeddings et les réponses.

## Contexte du projet géospatial

Les documents indexés décrivent le cadrage et la collecte de données pour
identifier les zones insuffisamment couvertes en soins dans la région
Casablanca-Settat. Le projet couvre notamment :

- les limites administratives ;
- les établissements de santé ;
- le réseau routier ;
- la population fine WorldPop ;
- la population légale RGPH 2024 ;
- les niveaux de soins et les seuils d'accessibilité.

Ces informations constituent des données de démonstration pour le RAG et ne sont
pas une analyse sanitaire finalisée.