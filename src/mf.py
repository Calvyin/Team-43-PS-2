import json
import faiss
from rapidfuzz import process
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

# Load the dataset with multiple JSON objects
with open("datasets/json-fixer.json") as f:
    db_vals = json.load(f)  # Assuming the JSON is an array of objects

# Define the fields to extract for encoding
fields_to_extract = ["name", "shortName", "industry", "securityType", "ticker", "finCode"]

# Prepare data for encoding by concatenating relevant fields
entries_to_encode = []
for entry in db_vals:
    fields = []
    for key in fields_to_extract:
        if key in entry:
            value = entry[key]
            # If the value is a dictionary or list, serialize it to a string
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            fields.append(str(value))
    concatenated_fields = " ".join(fields)  # Combine all fields into a single string
    entries_to_encode.append(concatenated_fields)

# This section initializes the AI models
model = SentenceTransformer("all-MiniLM-L6-v2")
try:
    reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
except Exception as e:
    print(f"Reranker initialization failed. Error: {e}")
    reranker = None

# Encode the concatenated data for indexing
embeddings = model.encode(entries_to_encode).astype("float32")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Using Faiss for similarity search
def search_with_embeddings(query: str, top: int = 5) -> list[str]:
    query_embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top)
    return [entries_to_encode[i] for i in indices[0]]  # Return matching entries

# Fuzzy matching using RapidFuzz
def fuzzy_match(query: str, threshold: int = 70) -> str | None:
    result = process.extractOne(query, entries_to_encode, score_cutoff=threshold)
    return result[0] if result else None

# Re-rank matches (if necessary)
def rerank_matches(query: str, matches: list[str]) -> list[str]:
    if not matches or not reranker:
        return matches
    scored_pairs = [(query, match) for match in matches]
    scores = reranker.compute_score(scored_pairs)
    return [match for _, match in sorted(zip(scores, matches), reverse=True)]

# Final function to control all search mechanisms
def match_security(query: str) -> str:
    if match := fuzzy_match(query):
        return match

    matches = search_with_embeddings(query)
    return rerank_matches(query, matches)[0]

# Test the function with the new JSON dataset
print(match_security("JCOOCS"))