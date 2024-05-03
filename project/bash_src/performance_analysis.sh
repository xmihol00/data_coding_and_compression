
data_dir="data_benchmark"
if [ ! -d $data_dir ]; then
    data_dir="../$data_dir"
fi 

mkdir -p compressed_files
mkdir -p decompressed_files

for i in {1..100}; do
    for output_type in partial algorithm full; do
        executable="./huff_codec"
        if [ ! -f $executable ]; then
            cd ..
            make ${output_type}_measure
            cd -
            executable=".$executable"
        else
            make ${output_type}_measure
        fi
        echo "file_name,measurement_name,value,number_of_threads,adaptive,model" > compression_${output_type}_performance_analysis.csv
        echo "file_name,measurement_name,value,number_of_threads,adaptive,model" > decompression_${output_type}_performance_analysis.csv
        for switch in "" "-a" "-m" "-m -a"; do
            for threads in 1 2 4 8 16 32; do
                rm -f compressed_files/*
                rm -f decompressed_files/*
                for file in $data_dir/*.raw; do
                    echo -e "\e[0;35mProcessing $file with $threads threads and switch $switch\e[0m"
                    basename=$(basename $file)
                    file_size=$(wc -c < $file)
                    width=$(echo "sqrt($file_size)" | bc -l | cut -d'.' -f1)
                    $executable -c -i $file -o compressed_files/$basename -w $width $switch -t $threads >> compression_${output_type}_performance_analysis.csv
                    $executable -d -i compressed_files/$basename -o decompressed_files/$basename -t $threads >> decompression_${output_type}_performance_analysis.csv
                    compressed_size=$(wc -c < compressed_files/$basename)
                    echo "$basename, $threads, $compressed_size"
                done
            done
        done
    done
done

rm -f compressed_files/*
rm -f decompressed_files/*