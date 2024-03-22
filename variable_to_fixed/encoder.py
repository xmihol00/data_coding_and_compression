import math

#substrings = ["Bar", "ba", "ra", "bo", "u", "r", "a"]
#substrings = ["Barbara", "aru", "B", "b", "u", "r", "a", "o"]
#substrings = ["Bar", "bar", "a", "b", "u", "r", "o"]
substrings = ["ar", "B", "a", "b", "u", "r", "o"] #5.13
#substrings = ["Bar", "baraa", "b", "o", "r", "a", "u"]
text = "BarbaraaBarboraubaruBaruBarbaraa"

selected_substrings = []
while text:
    valid_substrings = []
    for substring in substrings:
        if text.startswith(substring):
            valid_substrings.append(substring)
    
    longest_substring = max(valid_substrings, key=len, default=None)
    selected_substrings.append(longest_substring)
    text = text[len(longest_substring):]

print(selected_substrings)

substrings_set = list(set(selected_substrings))
sorted_substrings = sorted(substrings_set, key=len, reverse=True)
print(sorted_substrings)

gamma_codes = ["1", "010", "011", "00100", "00101", "00110", "00111", "0001000", "0001001", "0001010", "0001011", "0001100", "0001101", "0001110", "0001111", "000010000", "000010001"]
for substr in sorted_substrings:
    binary = ""
    for char in substr:
        binary += f"{ord(char):08b}"
    print(gamma_codes[len(substr)])
    print(binary, f"# {substr}")

print("1")
print("1")

index_size = math.ceil(math.log2(len(sorted_substrings)))
for substr in selected_substrings:
    index = sorted_substrings.index(substr)
    print(f"{index:0{index_size}b}", f"# {substr}")
