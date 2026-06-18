import json
import random
from pathlib import Path

def main():
    source_path = Path("hotpot_dev_distractor_v1.json")
    target_path = Path("data/hotpot_test.json")
    
    if not source_path.exists():
        print(f"Error: {source_path} not found in workspace!")
        return
        
    print(f"Loading {source_path}...")
    with open(source_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    print(f"Found {len(raw_data)} examples. Converting format...")
    
    converted = []
    for item in raw_data:
        # Check level
        level = item.get("level", "medium")
        if level not in ["easy", "medium", "hard"]:
            level = "medium"
            
        # Format context
        context_chunks = []
        for title, sentences in item.get("context", []):
            text = " ".join(sentences).strip()
            context_chunks.append({
                "title": title,
                "text": text
            })
            
        converted.append({
            "qid": item["_id"],
            "difficulty": level,
            "question": item["question"],
            "gold_answer": item["answer"],
            "context": context_chunks
        })
        
    # Sample 60 examples to have 120 total records (react + reflexion)
    # Use a seed for reproducibility
    random.seed(42)
    sampled = random.sample(converted, 60)
    
    # Ensure directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated {len(sampled)} test examples at {target_path}")

if __name__ == "__main__":
    main()
