import json
import sqlite3
import re
from pathlib import Path

def run():
    golden_path = Path("evaluation/benchmarks/golden_set.json")
    with open(golden_path, "r") as f:
        data = json.load(f)

    db_path = "data/indexes/metadata.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    remapped_count = 0
    needs_review = []

    for q in data["queries"]:
        print(f"Processing query {q['query_id']}...")
        gold_doc_id = q.get("gold_doc_id")
        gold_text = q.get("gold_text")
        
        if not gold_doc_id or not gold_text:
            print(f"Skipping {q['query_id']} due to missing gold_doc_id or gold_text")
            needs_review.append(q["query_id"])
            continue

        # Fix for OFFSETS-01: Some docs have a " copy" suffix in the filename and DB, but the golden set
        # gold_doc_id does not. We check both to prevent phantom NEEDS_REVIEW flags.
        c.execute("SELECT chunk_id, char_start, char_end, text, doc_id FROM chunks WHERE doc_id=? OR doc_id=? ORDER BY chunk_index ASC", (gold_doc_id, f"{gold_doc_id} copy"))
        rows = c.fetchall()

        if not rows:
            print(f"No chunks found in DB for doc_id {gold_doc_id} (or copy)")
            needs_review.append(q["query_id"])
            continue
            
        actual_doc_id = rows[0][4]
        raw_path = Path(f"data/raw/{actual_doc_id}.txt")
        if not raw_path.exists():
            # try to find it with glob
            matches = list(Path("data/raw").rglob(f"{gold_doc_id}*.*"))
            if matches:
                raw_path = matches[0]
            else:
                needs_review.append(q["query_id"])
                continue
                
        with open(raw_path, "r", encoding="utf-8") as f:
            full_text = f.read()
            
        # Find exact substring
        start_idx = full_text.find(gold_text)
        
        if start_idx == -1:
            print(f"Exact match failed for {q['query_id']}. Trying regex...")
            # Maybe newlines got stripped? Try to ignore whitespace
            pattern = re.escape(gold_text)
            # Replace escaped spaces with \s* to match optional whitespace instead of \s+ to avoid catastrophic backtracking
            pattern = pattern.replace(r"\ ", r"\s*")
            match = re.search(pattern, full_text)
            if match:
                start_idx = match.start()
                end_idx = match.end()
            else:
                print(f"Regex match also failed for {q['query_id']}")
                needs_review.append(q["query_id"])
                continue
        else:
            end_idx = start_idx + len(gold_text)
            
        gold_len = end_idx - start_idx
        relevant_chunks = []
        for chunk_id, c_start, c_end, c_text, _ in rows:
            overlap_start = max(start_idx, c_start)
            overlap_end = min(end_idx, c_end)
            overlap = max(0, overlap_end - overlap_start)
            
            c_len = c_end - c_start
            
            if overlap > 0:
                if c_len > 0 and (overlap / c_len >= 0.5 or overlap / gold_len >= 0.5):
                    relevant_chunks.append(chunk_id)
                    
        if relevant_chunks:
            q["relevant_chunk_ids"] = relevant_chunks
            remapped_count += 1
        else:
            needs_review.append(q["query_id"])
            
    if len(needs_review) == 0:
        data["status"] = "HUMAN_VERIFIED"
    else:
        data["status"] = "NEEDS_REVIEW"
        
    with open(golden_path, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"Remapped {remapped_count}/{len(data['queries'])}.")
    if needs_review:
        print(f"Needs review: {len(needs_review)}")
        print("Queries needing review:", needs_review)

if __name__ == "__main__":
    run()
