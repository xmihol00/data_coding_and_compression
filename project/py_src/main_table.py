import numpy as np
import pandas as pd
import glob
import os

os.makedirs("tex_tables", exist_ok=True)
filenames = sorted(list(glob.glob("data_benchmark/*.raw")))
base_filenames = [os.path.basename(filename) for filename in filenames]

columns = pd.MultiIndex.from_tuples([("Entropy", "")] + [(column, mode) for column in ["Compressed bits per pixel", "Compression time", "Decompression time"] 
                                                                        for mode in ["S", "A", "S+M", "A+M"]])
result_df = pd.DataFrame(columns=columns, index=base_filenames)

for filename, base_filename in zip(filenames, base_filenames):
    with open(filename, "rb") as f:
        data = f.read()
        data = np.frombuffer(data, dtype=np.uint8)
        histogram = np.bincount(data, minlength=256)
        probabilities = histogram / len(data)
        probabilities = probabilities[probabilities != 0] # drop zero probabilities
        inverse_probabilities = 1 / probabilities
        log_probabilities = np.log2(inverse_probabilities)
        entropy = np.dot(probabilities, log_probabilities.T)
        result_df.loc[base_filename, ("Entropy", "")] = entropy

# header: file_name,measurement_name,value,number_of_threads,adaptive, model
df = pd.read_csv("csv_measurements/sizes_and_data_analysis.csv", header=0, index_col=False)
df = df.loc[df["number_of_threads"] == 1, :]
df["file_name"] = df["file_name"].apply(lambda x: x.split("/")[-1])
df_compressed_file_size = df.loc[df["measurement_name"] == "Compressed file size", :]
df_uncompressed_file_size = df.loc[df["measurement_name"] == "Uncompressed file size", :]
for row_compressed, row_uncompressed in zip(df_compressed_file_size.iterrows(), df_uncompressed_file_size.iterrows()):
    row_compressed = row_compressed[1]
    row_uncompressed = row_uncompressed[1]
    if row_compressed["file_name"] != row_uncompressed["file_name"]:
        print("Error: file names do not match")
        exit()
    if row_compressed["adaptive"] != row_uncompressed["adaptive"]:
        print("Error: adaptive does not match")
        exit()
    if row_compressed["model"] != row_uncompressed["model"]:
        print("Error: model does not match")
        exit()
    if row_compressed["number_of_threads"] != row_uncompressed["number_of_threads"]:
        print("Error: number of threads does not match")
        exit()
    
    compressed_ratio_row = row_compressed.copy()
    compressed_ratio_row["measurement_name"] = "Compressed bits per pixel"
    compressed_ratio_row["value"] = (row_compressed["value"] / row_uncompressed["value"]) * 8


    df.loc[len(df)] = compressed_ratio_row

df = df.loc[df["measurement_name"] == "Compressed bits per pixel", :]
for adaptive, model, in zip([False, True, False, True], [False, False, True, True]):
    filtered_df = df[(df["adaptive"] == adaptive) & (df["model"] == model)]
    mode_name = f"{'S' if not adaptive else 'A'}{'' if not model else '+M'}"
    for base_filename in base_filenames:
        try:
            row = filtered_df.loc[filtered_df["file_name"] == base_filename, "value"].values[0]
            result_df.loc[base_filename, ("Compressed bits per pixel", mode_name)] = row
        except:
            pass

# header: file_name,measurement_name,value,number_of_threads,adaptive, model
df = pd.read_csv(os.path.join("csv_measurements", "compression_full_performance_analysis.csv"), header=0, index_col=False)
df["file_name"] = df["file_name"].apply(lambda x: x.split("/")[-1])

aggregates = ["mean"]
for adaptive, model in zip([False, True, False, True], [False, False, True, True]):
    filtered_df = df[(df["adaptive"] == adaptive) & (df["model"] == model)].copy()
    filtered_df.drop(columns=["adaptive", "model", "measurement_name"], inplace=True)
    stats = filtered_df.groupby(["file_name", "number_of_threads"]).agg(aggregates)["value"]
    mode_name = f"{'S' if not adaptive else 'A'}{'' if not model else '+M'}"
    for base_filename in base_filenames:
        filtered_stats = stats.loc[base_filename, :]
        result_df.loc[base_filename, ("Compression time", mode_name)] = filtered_stats["mean"].min()

# header: file_name,measurement_name,value,number_of_threads,adaptive, model
df = pd.read_csv(os.path.join("csv_measurements", "decompression_full_performance_analysis.csv"), header=0, index_col=False)
df["file_name"] = df["file_name"].apply(lambda x: x.split("/")[-1])

aggregates = ["mean"]
for adaptive, model in zip([False, True, False, True], [False, False, True, True]):
    filtered_df = df[(df["adaptive"] == adaptive) & (df["model"] == model)].copy()
    filtered_df.drop(columns=["adaptive", "model", "measurement_name"], inplace=True)
    stats = filtered_df.groupby(["file_name", "number_of_threads"]).agg(aggregates)["value"]
    mode_name = f"{'S' if not adaptive else 'A'}{'' if not model else '+M'}"
    for base_filename in base_filenames:
        filtered_stats = stats.loc[base_filename, :]
        result_df.loc[base_filename, ("Decompression time", mode_name)] = filtered_stats["mean"].min()
    

(result_df.style
        .set_caption(f"Entropy, Compression effectiveness and time, decompression time")
        .format("{:.4f}")
        .to_latex(os.path.join("tex_tables", "main_table.tex")))