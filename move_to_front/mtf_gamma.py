import numpy as np

def MTF(text, dictionary=None):
    if not dictionary: 
        dictionary = [chr(i) for i in range(0,255)]
    result = []
    for c in text:
        rank = dictionary.index(c)
        result.append(rank)
        dictionary.pop(rank)
        dictionary.insert(0, c)

    return result

def iMTF(sequence, dictionary=None):
    if not dictionary:
        dictionary = [chr(i) for i in range(0,255)]
    result = ""
    for rank in sequence:
        result += dictionary[rank]
        e = dictionary.pop(rank)
        dictionary.insert(0, e)
    return result

def gama_encode(sequence):
    for char in sequence:
        char += 1
        bit_len = char.bit_length()
        bin_char = str(bin(char)[3:])
        encoded =  "0"*bit_len + "1" + bin_char[:bit_len]
        print(encoded)

def gama_decode(sequence):
    pass

if __name__ == "__main__":
    #text = "KRALOVNA_KLARA_NA_KLAVIR_KRALOVI_HRALA"
    text = "AAAAAAAABBBBBBBBCCCCCCCCDDDDDDDD"
    dictonary = list(set(text))
    dictonary = sorted(dictonary)
    print(dictonary)
    mtf = MTF(text, dictonary)
    imtf = iMTF(mtf, dictonary)
    print(mtf)
    print(imtf)

    gama_encode(mtf)
