from itertools import permutations
from gamma import gama_encode

def MTF_move_ahead(text, dictionary=None, k=2):
    if not dictionary: 
        dictionary = [chr(i) for i in range(0,255)]
    else:
        dictionary = dictionary.copy()

    result = []
    for c in text:
        rank = dictionary.index(c)
        result.append(rank)
        dictionary.pop(rank)
        index = max(0, rank-k)
        dictionary.insert(index, c)

    return result

def iMTF_move_ahead(sequence, dictionary=None, k=2):
    if not dictionary:
        dictionary = [chr(i) for i in range(0,255)]
    else:
        dictionary = dictionary.copy()

    result = ""
    for rank in sequence:
        result += dictionary[rank]
        e = dictionary.pop(rank)
        index = max(0, rank-k)
        dictionary.insert(index, e)
    return result

if __name__ == "__main__":
    text = "KRALOVNA_KLARA_NA_KLAVIR_KRALOVI_HRALA"
    if True:
        smallest_bit_count = 2**32
        print(len(list(permutations(list(set(text))))))
        for i, perm in enumerate(permutations(list(set(text)))):
            dictionary = list(perm)
            mtf = MTF_move_ahead(text, dictionary)
            encoded = gama_encode(mtf)
            length = len(encoded)
            if length < smallest_bit_count:
                print("New smallest:", length)
                smallest_bit_count = length
                smallest_mtf = mtf
                smallest_dictionary = dictionary
                smallest_encoded = encoded
    else:
        smallest_dictionary = ['K', 'L', 'A', '_', 'N', 'R', 'V', 'H', 'I', 'O']
        smallest_mtf = MTF_move_ahead(text, smallest_dictionary)
        smallest_encoded = gama_encode(smallest_mtf)

    print(f"{len(smallest_dictionary):08b}", "# slovnik")
    for c in smallest_dictionary:
        print(f"{ord(c):08b}", f"# {c}")
    print(smallest_encoded)
