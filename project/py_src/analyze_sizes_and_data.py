import pandas as pd
import os

os.makedirs("tex_tables", exist_ok=True)

# header: file_name,measurement_name,value,number_of_threads,adaptive, model
df = pd.read_csv("sizes_and_data_analysis.csv", header=0, index_col=False)
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
    compressed_ratio_row["measurement_name"] = "Compression ratio"
    compressed_ratio_row["value"] = row_uncompressed["value"] / row_compressed["value"]

    space_savings_row = row_compressed.copy()
    space_savings_row["measurement_name"] = "Space savings"
    space_savings_row["value"] = 1 - row_compressed["value"] / row_uncompressed["value"]

    df.loc[len(df)] = compressed_ratio_row
    df.loc[len(df)] = space_savings_row

unique_file_names = df["file_name"].unique()
aggregates = ["min", "max", "mean", "std"]
unique_measurement_names = df["measurement_name"].unique().tolist()
column_tuples = [(measurement_name, aggregate) for measurement_name in unique_measurement_names for aggregate in aggregates]
columns = pd.MultiIndex.from_tuples(column_tuples)

for output_file, adaptive, model in zip(
    ["static_compression.tex", "adaptive_compression.tex", "static_compression_with_model.tex", "adaptive_compression_with_model.tex"], 
    [False, True, False, True], [False, False, True, True]
):
    filtered_df = df[(df["adaptive"] == adaptive) & (df["model"] == model)]
    stats = filtered_df.groupby(["file_name", "measurement_name"]).agg(aggregates)["value"]
    aggregated_df = pd.DataFrame(columns=columns, index=unique_file_names)
    for file_name in unique_file_names:
        for measurement_name in unique_measurement_names:
            try:
                row = stats.loc[file_name, measurement_name]
                aggregated_df.loc[file_name, (measurement_name, "min")] = row["min"]
                aggregated_df.loc[file_name, (measurement_name, "max")] = row["max"]
                aggregated_df.loc[file_name, (measurement_name, "mean")] = row["mean"]
                aggregated_df.loc[file_name, (measurement_name, "std")] = row["std"]
            except:
                pass
    
    aggregated_df = aggregated_df[["Uncompressed file size", "Compressed file size", "Compression ratio", "Space savings"]]
    aggregated_df.drop(columns=[("Uncompressed file size", "std"), ("Uncompressed file size", "min"), ("Uncompressed file size", "max")], inplace=True)
    print(aggregated_df)
    (aggregated_df.style
        .set_caption(f"{'Static' if not adaptive else 'Adaptive'} compression {'without' if not model else 'with'} model")
        .format("{:.2f}")
        .to_latex(os.path.join("tex_tables", output_file)))

