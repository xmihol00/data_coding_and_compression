import os

os.makedirs("data_edge_case", exist_ok=True)

with open("data_edge_case/large_triples_8388611x6.bin", "w") as f:
    triples = "xyz" * 16777216
    f.write(triples)
    f.write(".!?:;,")
