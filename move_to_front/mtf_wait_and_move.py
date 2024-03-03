from itertools import permutations
from gamma import gama_encode
from huffman import huffman_encode

def MTF_wait_and_move(text, dictionary=None, k=2):
    if not dictionary: 
        dictionary = [chr(i) for i in range(0,255)]
    else:
        dictionary = dictionary.copy()

    counters = {c: 0 for c in dictionary}
    thresholds = {c: i+k for i, c in zip(range(255), dictionary)}

    result = []
    for c in text:
        rank = dictionary.index(c)
        result.append(rank)
        counters[c] += 1
        if counters[c] > thresholds[c]:
            dictionary.pop(rank)
            dictionary.insert(0, c)

    return result

def iMTF_wait_and_move(sequence, dictionary=None, k=2):
    if not dictionary:
        dictionary = [chr(i) for i in range(0,255)]
    else:
        dictionary = dictionary.copy()

    counters = {c: 0 for c in dictionary}

    result = ""
    for rank in sequence:
        result += dictionary[rank]
        counters[e] += 1
        if counters[e] > k:
            e = dictionary.pop(rank)
            dictionary.insert(0, e)

    return result

if __name__ == "__main__":
    text = "KRALOVNA_KLARA_NA_KLAVIR_KRALOVI_HRALA"
    for char in set(text):
        print(char, text.count(char))

    if True:
        smallest_bit_count = 2**32
        for i in range(1, 10):
            mtf = MTF_wait_and_move(text, k=i)
            encoded = gama_encode(mtf)
            length = len(encoded) - 88
            if length < smallest_bit_count:
                print("New smallest:", length)
                smallest_bit_count = length
                smallest_mtf = mtf
                smallest_dictionary = [chr(i) for i in range(0,255)]
                smallest_encoded = encoded

        alphabet = sorted(list(set(text)))
        alphabet_len = len(alphabet)
        print(alphabet_len)
        for i, perm in enumerate(permutations(alphabet)):
            dictionary = list(perm)
            mtf = MTF_wait_and_move(text, dictionary, 6)
            encoded = gama_encode(mtf)
            #encoded = huffman_encode(mtf, alphabet_len)
            length = len(encoded)
            if length < smallest_bit_count:
                print("New smallest:", length)
                smallest_bit_count = length
                smallest_mtf = mtf
                smallest_dictionary = dictionary
                smallest_encoded = encoded
        
    else:
        smallest_dictionary = ['K', 'R', 'A', 'N', 'V', 'L', 'O', '_', 'I', 'H']
        smallest_mtf = MTF_wait_and_move(text, smallest_dictionary)
        smallest_encoded = gama_encode(smallest_mtf)

    print(f"{len(smallest_dictionary):08b}", "# slovnik")
    for c in smallest_dictionary:
        print(f"{ord(c):08b}", f"# {c}")
    print(smallest_encoded)
