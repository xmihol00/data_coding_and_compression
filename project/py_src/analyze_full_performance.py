import pandas as pd
import os

os.makedirs("tex_tables", exist_ok=True)

# header: file_name,measurement_name,value,number_of_threads,adaptive, model
for input_file_prefix in ["compression", "decompression"]:
    df = pd.read_csv(os.path.join("csv_measurements", input_file_prefix + "_full_performance_analysis.csv"), header=0, index_col=False)
    df["file_name"] = df["file_name"].apply(lambda x: x.split("/")[-1])
    print(df["file_name"])

    unique_file_names = df["file_name"].unique()
    aggregates = ["mean", "std"]
    number_of_threads = df["number_of_threads"].unique().tolist()
    column_tuples = [(num_threads, aggregate) for num_threads in number_of_threads for aggregate in aggregates]
    columns = pd.MultiIndex.from_tuples(column_tuples)

    for output_file, adaptive, model in zip(
        ["static_full_performance.tex", "adaptive_full_performance.tex", "static_model_full_performance.tex", "adaptive_model_full_performance.tex"], 
        [False, True, False, True], [False, False, True, True]
    ):
        filtered_df = df[(df["adaptive"] == adaptive) & (df["model"] == model)].copy()
        filtered_df.drop(columns=["adaptive", "model", "measurement_name"], inplace=True)
        stats = filtered_df.groupby(["file_name", "number_of_threads"]).agg(aggregates)["value"]
        aggregated_df = pd.DataFrame(columns=columns, index=unique_file_names)
        for file_name in unique_file_names:
            for num_threads in number_of_threads:
                try:
                    row = stats.loc[file_name, num_threads]
                    aggregated_df.loc[file_name, (num_threads, "mean")] = row["mean"]
                    aggregated_df.loc[file_name, (num_threads, "std")] = row["std"]
                except:
                    pass
        
        (aggregated_df.style
            .set_caption(f"Performance of full {'static' if not adaptive else 'adaptive'} {input_file_prefix} {'without a' if not model else 'with implemented difference'} model")
            .format("{:.2f}")
            .to_latex(os.path.join("tex_tables", f"{input_file_prefix}_{output_file}")))
