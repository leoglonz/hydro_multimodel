from collections import Counter

def classToint(string_list):
    # Using Counter to count occurrences
    string_counts = Counter(string_list)

    # Find strings that occur only once
    unique_strings = [s for s, count in string_counts.items() if count == 1]
    string_to_int_map = {string: index for index, string in enumerate(unique_strings)}

    # Assign integers to each string in the original array
    integers = [string_to_int_map[s] if s in string_to_int_map else None for s in string_list]

    return unique_strings

