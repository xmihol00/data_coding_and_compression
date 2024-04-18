import glob
import numpy as np
import os

WIDTH = 512
HEIGHT = 512
SRC_DIR = "."
DST_DIR = "."
os.makedirs(DST_DIR, exist_ok=True)

for filename in glob.glob(os.path.join(SRC_DIR, "*.raw")):
    basename = os.path.basename(filename)
    pgm_filename = os.path.join(DST_DIR, f"{os.path.splitext(basename)[0]}.pgm") 
    with open(filename, "rb") as f:
        data = f.read()
    
    data = np.frombuffer(data, dtype=np.uint8).reshape(HEIGHT, WIDTH)
    with open(pgm_filename, "w") as f:
        f.write("P2\n")
        f.write(f"{WIDTH} {HEIGHT}\n")
        f.write("255\n")
        for row in data:
            f.write(" ".join(map(lambda x: f"{x: 3}", row)))
            f.write("\n")