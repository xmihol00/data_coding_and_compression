
data_dir="data_benchmark"
if [ ! -d $data_dir ]; then
    data_dir="../$data_dir"
fi 

executable="./huff_codec"
if [ ! -f $executable ]; then
    executable=".$executable"
fi

mkdir -p compressed_files
mkdir -p decompressed_files

for switch in "" "-a" "-m" "-m -a"; do
    for threads in 1 2 4 8; do
        rm -f compressed_files/*
        rm -f decompressed_files/*
        for file in $data_dir/*.raw; do
            basename=$(basename $file)
            file_size=$(wc -c < $file)
            width=$(echo "sqrt($file_size)" | bc -l | cut -d'.' -f1)
            $executable -c -i $file -o compressed_files/$basename -w $width $switch -t $threads
            compressed_size=$(wc -c < compressed_files/$basename)
            echo "$basename, $threads, $compressed_size"            
        done
    done
done

rm -f compressed_files/*
rm -f decompressed_files/*