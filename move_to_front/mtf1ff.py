from itertools import permutations
from gamma import gama_encode

def MTF1FF(text, dictionary=None):
    if not dictionary: 
        dictionary = [chr(i) for i in range(0,255)]
    else:
        dictionary = dictionary.copy()

    result = []
    for c in text:
        rank = dictionary.index(c)
        result.append(rank)
        dictionary.pop(rank)
        if rank >= 2:
            dictionary.insert(1, c)
        else:
            dictionary.insert(0, c)

    return result

def iMTF1FF(sequence, dictionary=None):
    if not dictionary:
        dictionary = [chr(i) for i in range(0,255)]
    else:
        dictionary = dictionary.copy()

    result = ""
    for rank in sequence:
        result += dictionary[rank]
        e = dictionary.pop(rank)
        if rank >= 2:
            dictionary.insert(1, e)
        else:
            dictionary.insert(0, e)
    return result

if __name__ == "__main__":
    text = "KRALOVNA_KLARA_NA_KLAVIR_KRALOVI_HRALA"
    
    if True:
        smallest_bit_count = 2**32
        mtf = MTF1FF(text)
        encoded = gama_encode(mtf)
        length = len(encoded) - 88
        if length < smallest_bit_count:
            print("New smallest:", length)
            smallest_bit_count = length
            smallest_mtf = mtf
            smallest_dictionary = [chr(i) for i in range(0,255)]
            smallest_encoded = encoded

        for i, perm in enumerate(permutations(list(set(text)))):
            dictionary = list(perm)
            mtf = MTF1FF(text, dictionary)
            encoded = gama_encode(mtf)
            length = len(encoded)
            if length < smallest_bit_count:
                print("New smallest:", length)
                smallest_bit_count = length
                smallest_mtf = mtf
                smallest_dictionary = dictionary
                smallest_encoded = encoded
    else:
        smallest_dictionary = ['A', 'R', 'K', 'L', 'N', 'V', 'O', 'I', 'H', '_']
        smallest_mtf = MTF1FF(text, smallest_dictionary)
        smallest_encoded = gama_encode(smallest_mtf)

    print(f"{len(smallest_dictionary):08b}", "# slovnik")
    for c in smallest_dictionary:
        print(f"{ord(c):08b}", f"# {c}")
    print(smallest_encoded)