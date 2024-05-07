
data_dir="data_benchmark_scaling"
if [ ! -d $data_dir ]; then
    data_dir="../$data_dir"
fi
benchmar_dir="data_benchmark"
if [ ! -d $benchmar_dir ]; then
    benchmar_dir="../$benchmar_dir"
fi

mkdir -p compressed_files
mkdir -p decompressed_files

for output_type in algorithm; do
    executable="./huff_codec"
    if [ ! -f $executable ]; then
        cd ..
        make ${output_type}_measure
        cd -
        executable=".$executable"
    else
        make ${output_type}_measure
    fi

    for size_multiple in 1 2 4 8 16 32; do
        rm -rf $data_dir/*
        file_size=0
        for file in $benchmar_dir/*.raw; do
            basename=$(basename $file)
            for ((i=0; i<$size_multiple; i++)); do
                cat "$file" >> "$data_dir/$basename"
            done
            file_size=$(wc -c < "$data_dir/$basename")
            echo -e "\e[0;35m$data_dir/$basename: $file_size B\e[0m"
        done
        echo -e "\e[0;35mUsing files: compression_${file_size}_scaling_analysis.csv and decompression_${file_size}_scaling_analysis.csv\e[0m"
        echo "file_name,measurement_name,value,number_of_threads,adaptive,model" > compression_${file_size}_scaling_analysis.csv
        echo "file_name,measurement_name,value,number_of_threads,adaptive,model" > decompression_${file_size}_scaling_analysis.csv

        for i in {1..100}; do
            echo -e "\n\e[0;34mRunning $output_type measurement $i\e[0m\n"
            sleep 1
            for switch in "" "-a" "-m" "-m -a"; do
                for threads in 1 2 4 8 16 32; do
                    rm -f compressed_files/*
                    rm -f decompressed_files/*
                    for file in $data_dir/*.raw; do
                        echo -e "\e[0;35mProcessing $file with $threads threads and switch $switch\e[0m"
                        basename=$(basename $file)
                        width=512
                        $executable -c -i $file -o compressed_files/$basename -w $width $switch -t $threads >> compression_${file_size}_scaling_analysis.csv
                        $executable -d -i compressed_files/$basename -o decompressed_files/$basename -t $threads >> decompression_${file_size}_scaling_analysis.csv
                        compressed_size=$(wc -c < compressed_files/$basename)
                        echo "$basename, $threads, $compressed_size"
                    done
                done
            done
        done
    done
done

rm -f compressed_files/*
rm -f decompressed_files/*