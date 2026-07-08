import sqlite3
import json
import os
from transformers import pipeline

def main():
    os.environ["USE_TF"] = "0"
    os.environ["USE_JAX"] = "0"
    
    conn = sqlite3.connect("data/indexes/metadata.db")
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, text FROM chunks WHERE length(text) > 300 ORDER BY RANDOM() LIMIT 100")
    rows = cursor.fetchall()
    
    print("Loading generator model...")
    generator = pipeline("text2text-generation", model="google/flan-t5-small", device="cpu")
    
    queries = []
    for i, (chunk_id, text) in enumerate(rows):
        prompt = f"Generate a question that can be answered by the following text:\n\n{text[:500]}"
        output = generator(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]
        
        queries.append({
            "query_id": f"q{i+1:03d}",
            "query": output,
            "relevant_chunk_ids": [chunk_id]
        })
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/100")
            
    golden = {
        "status": "PENDING_HUMAN_VERIFICATION",
        "version": "1.0",
        "queries": queries
    }
    
    os.makedirs("evaluation/benchmarks", exist_ok=True)
    with open("evaluation/benchmarks/golden_set.json", "w") as f:
        json.dump(golden, f, indent=2)
        
    print("Successfully generated golden_set.json")

if __name__ == "__main__":
    main()
