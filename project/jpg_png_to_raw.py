from PIL import Image
import io
import glob
import os

SRC_DIR = "jpg_png"
DST_DIR = "data"
os.makedirs(DST_DIR, exist_ok=True)

for in_filename in glob.glob(os.path.join(SRC_DIR, "*.raw")):
    basename = os.path.basename(in_filename)
    out_filename = os.path.join(DST_DIR, f"{os.path.splitext(basename)[0]}.raw")

    image = Image.open("example.png")
    with io.BytesIO() as output:
        image.save(output, format='raw')
        raw_bytes_png = output.getvalue()

    with open(out_filename, "wb") as f:
        f.write(raw_bytes_png)
