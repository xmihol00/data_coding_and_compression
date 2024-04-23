
data_dir="data"
if [ ! -d $data_dir ]; then
    data_dir="../$data_dir"
fi 

all_tests_passed=true

thread_combinations=(
    "1 1"
    "2 1"
    "1 2"
    "2 2"
    "2 4"
    "4 2"
    "4 4"
    "8 4"
    "4 8"
    "8 8"
    "8 1"
)

for switch in "" "-a" "-m" "-m -a"; do
    echo ""
    echo -e "\e[0;35mTesting with switch $switch\e[0m"
    echo ""
    for thread_combination in "${thread_combinations[@]}"; do
        echo -e "\e[0;36mTesting with thread combination $thread_combination\e[0m"

        read compress_threads decompress_threads <<< $thread_combination

        rm -f compressed_files/*
        rm -f decompressed_files/*
        for file in $data_dir/*.raw; do
            echo "Testing $file"
            basename=$(basename $file)
            file_size=$(wc -c < $file)
            width=$(echo "sqrt($file_size)" | bc -l | cut -d'.' -f1)
            echo "compress command: ./huff_codec -c -i $file -o compressed_files/$basename -w $width $switch -t $compress_threads"
            ./huff_codec -c -i $file -o compressed_files/$basename -w $width $switch -t $compress_threads 2>/dev/null
            echo "decompress command: ./huff_codec -d -i compressed_files/$basename -o decompressed_files/$basename -t $decompress_threads"
            ./huff_codec -d -i compressed_files/$basename -o decompressed_files/$basename -t $decompress_threads 2>/dev/null
            diff $file decompressed_files/$basename 2>/dev/null 1>/dev/null
            if [ $? -eq 0 ]; then
                echo -e "\e[0;32mPASSED\e[0m"
            else
                echo -e "\e[0;31mFAILED\e[0m"
                all_tests_passed=false
                sleep 2
            fi
            echo ""
        done
    done
done

if [ $all_tests_passed = true ]; then
    echo -e "\e[0;34mAll tests passed\e[0m"
else
    echo -e "\e[0;31mSome tests failed\e[0m"
fi

rm -f compressed_files/*
rm -f decompressed_files/*
