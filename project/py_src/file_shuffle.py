import numpy as np

files = ["compressed_files/fibonaci_A-Z_1x317810.txt", "compressed_files/shifted_fibonaci_A-Z_1x317880.txt"]
for file in files:
    with open(file, 'rb') as f:
        data = f.read()
        data = np.array(np.frombuffer(data, dtype=np.uint8))
        np.random.shuffle(data)
        
    with open(file.replace("fibonacci", "shuffled_fibonacci"), 'wb') as f:
        f.write(data)