import csv

def count_values(values: list) -> dict:
    results = {}
    for val in values:
        if val not in results:
            results[val] = 1
        else:
            results[val] += 1

    return results

def return_majority(values:list):
    results = count_values(values)
    max_key = max(results, key=results.get)
    max_value = results[max_key]
    min_key = min(results, key=results.get)
    min_value = results[min_key]
    if max_value == min_value and min_value < len(values):
        return "EQUAL"
    else:
        return max_key
    
def check_unanimity(values:list) -> bool:
    last_value = None
    for value in values:
        if last_value == None:
            last_value = value
        if last_value != value:
            return False
    return True
        
#print(count_values([1,3,2]))