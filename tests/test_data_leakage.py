import json
from pathlib import Path

def test_no_data_leakage_in_training_triplets():
    golden_path = Path("evaluation/benchmarks/golden_set.json")
    triplets_path = Path("data/training/domain_triplets.jsonl")
    
    # If files don't exist yet, we can't test
    if not golden_path.exists() or not triplets_path.exists():
        return
        
    with open(golden_path) as f:
        golden_data = json.load(f)
        
    gold_chunk_ids = set()
    for q in golden_data.get("queries", []):
        for cid in q.get("relevant_chunk_ids", []):
            gold_chunk_ids.add(cid)
            
    # Read generated triplets and check for leakage
    leaked = []
    with open(triplets_path) as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            source_id = item.get("source_chunk_id")
            if source_id and source_id in gold_chunk_ids:
                leaked.append(source_id)
                
    assert len(leaked) == 0, f"Found {len(leaked)} leaked gold chunks in training data!"
