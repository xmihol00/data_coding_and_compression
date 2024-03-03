def huffman_encode(sequence, alphabet_len):
    result = ""
    for char in sequence:
        encoded = ("1" * char) + "0"
        encoded = encoded[:alphabet_len-1] + "\n"
        result += encoded
    
    return result