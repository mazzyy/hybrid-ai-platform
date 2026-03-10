import json
import os
from collections import defaultdict

def analyze_results():
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'router_test_results.json')
    
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find results file at {results_path}")
        return
        
    total = len(data)
    if total == 0:
        print("No data found in results file.")
        return
        
    correct = sum(1 for item in data if item['is_correct'])
    
    print("="*80)
    print(f"ROUTER PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"OVERALL ACCURACY: {correct}/{total} ({(correct/total)*100:.1f}%)")
    print("="*80)
    
    categories = ['rag', 'confidential', 'complex', 'simple']
    
    # Confusion matrix: matrix[expected][predicted]
    matrix = {c: {p: 0 for p in categories} for c in categories}
    
    # Category stats
    cat_total = defaultdict(int)
    cat_correct = defaultdict(int)
    
    errors = []
    
    for item in data:
        expected = item['expected_category']
        predicted = item['predicted_category']
        
        # Ensure categories exist in matrix if LLM returned something unexpected
        if expected not in matrix:
            matrix[expected] = {p: 0 for p in categories}
            categories.append(expected)
            for c in categories:
                if expected not in matrix[c]:
                    matrix[c][expected] = 0
            
        if predicted not in matrix[expected]:
            for c in categories:
                matrix[c][predicted] = 0
            if predicted not in categories:
                categories.append(predicted)
                
        matrix[expected][predicted] += 1
            
        cat_total[expected] += 1
        if item['is_correct']:
            cat_correct[expected] += 1
        else:
            errors.append(item)
            
    print("\n[ ACCURACY BY CATEGORY ]")
    print(f"{'CATEGORY':<15} | {'CORRECT':<10} | {'TOTAL':<10} | {'ACCURACY':<10} | {'ERRORS':<10}")
    print("-" * 70)
    for cat in [c for c in categories if cat_total[c] > 0]:
        t = cat_total[cat]
        c = cat_correct[cat]
        print(f"{cat.upper():<15} | {c:<10} | {t:<10} | {(c/t)*100:>8.1f}% | {t-c:<10}")
            
    print("\n[ CONFUSION MATRIX ]")
    print("Rows: Expected Category, Columns: Predicted Category\n")
    
    # Print header
    header = f"{'':<15} |"
    for cat in categories:
        header += f"{cat.upper():>14} |"
    print(header)
    print("-" * len(header))
    
    for expected in [c for c in categories if cat_total[c] > 0]:
        row = f"{expected.upper():<15} |"
        for predicted in categories:
            val = matrix[expected].get(predicted, 0)
            row += f"{val:>14} |"
        print(row)
        
    print("\n[ ERROR ANALYSIS ]")
    if not errors:
        print("No errors found!")
    else:
        error_types = defaultdict(list)
        for e in errors:
            key = f"Expected '{e['expected_category'].upper()}' but got '{e['predicted_category'].upper()}'"
            error_types[key].append(e['query'])
            
        # Sort by frequency of error type
        sorted_errors = sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True)
            
        for err_type, queries in sorted_errors:
            percentage_of_total = (len(queries) / total) * 100
            percentage_of_errors = (len(queries) / len(errors)) * 100
            
            print(f"\n- {err_type}")
            print(f"  Count: {len(queries)} cases ({percentage_of_total:.1f}% of all queries, {percentage_of_errors:.1f}% of all errors)")
            # Show up to 5 examples
            for q in queries[:5]:
                display_q = q if len(q) < 80 else q[:77] + "..."
                print(f"   > {display_q}")
            if len(queries) > 5:
                print(f"   > ... and {len(queries)-5} more.")

if __name__ == '__main__':
    analyze_results()
