import json
import faiss
from sentence_transformers import SentenceTransformer

# Step 1: Load the JSON file
with open("datasets/json-fixer.json") as f:
    db_vals = json.load(f)  # Ensure the JSON file is an array of company objects

# Fields to extract for encoding and matching
fields_to_extract = ["finCode", "name", "shortName", "securityType", "sector"]

# Prepare data for encoding by concatenating specific fields
entries_to_encode = []
fund_details = []  # To store all fund details for later retrieval

for entry in db_vals:
    extracted_data = []
    for key in fields_to_extract:
        if key in entry:
            value = entry[key]
            # Serialize lists or dictionaries into strings
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            extracted_data.append(str(value))
    entries_to_encode.append(" ".join(extracted_data))  # Combine all extracted fields into a single string
    fund_details.append(entry)  # Store the full fund details for later retrieval

# Step 2: Initialize the AI model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Encode the concatenated data
embeddings = model.encode(entries_to_encode).astype("float32")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Step 4: Search for Matches
def search_funds(query: str, top_k: int = 5):
    query_embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        result = {
            "finCode": fund_details[idx]["finCode"],
            "name": fund_details[idx]["name"],
            "shortName": fund_details[idx].get("shortName", ""),
            "sector": fund_details[idx].get("sector", ""),
            "securityType": fund_details[idx].get("securityType", ""),
            "match_score": distances[0][i]
        }
        results.append(result)
    return results

# Step 5: Query Example
query = "Funds with HDFC holdings"
results = search_funds(query, top_k=3)

# Print the Results
print("Matching Funds:")
for result in results:
    print(f"finCode: {result['finCode']}, Name: {result['name']}, Match Score: {result['match_score']}")