import sys
import os
import json
import time

# Ensure we can import from the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from retrieval.llm_router import QueryRouter

def run_router_tests():
    data_path = os.path.join(project_root, 'router_test_data.json')
    results_path = os.path.join(project_root, 'router_test_results.json')
    
    # Load test data
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Test data file not found at {data_path}")
        return
        
    print("Initializing LLM Router...")
    router = QueryRouter()
    results = []
    
    correct = 0
    total = len(data)
    
    print(f"\nRunning {total} router tests...\n")
    print(f"{'Query':<60} | {'Expected':<14} | {'Rerouted To':<14} | {'Match'}")
    print("-" * 110)
    
    for item in data:
        query = item['query']
        expected = item['expected_category']
        
        # We don't need a full generation cycle, just the classification
        start_time = time.time()
        predicted = router.classify_query(query)
        end_time = time.time()
        
        latency = end_time - start_time
        
        is_match = expected == predicted
        if is_match:
            correct += 1
            
        # Record what the sentence was about and where it rerouted
        result = {
            "query": query,
            "expected_category": expected,
            "predicted_category": predicted,
            "is_correct": is_match,
            "latency_seconds": round(latency, 2)
        }
        results.append(result)
        
        # Truncate query for display if too long
        display_query = query if len(query) < 57 else query[:54] + "..."
        match_icon = "✅" if is_match else "❌"
        
        print(f"{display_query:<60} | {expected:<14} | {predicted:<14} | {match_icon} ({latency:.2f}s)")
        
    accuracy = (correct / total) * 100
    print("-" * 110)
    print(f"\nFinal Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    # Save results to output JSON
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nDetailed results saved to:\n{results_path}")

if __name__ == "__main__":
    run_router_tests()
