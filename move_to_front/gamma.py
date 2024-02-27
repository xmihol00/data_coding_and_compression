def gama_encode(sequence):
    result = ""
    for char in sequence:
        char += 1
        bit_len = char.bit_length()-1
        bin_char = str(bin(char)[2:])
        encoded = "0"*bit_len + bin_char[:bit_len+1] + "\n"
        result += encoded
    
    return result