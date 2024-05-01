
data_dir="data_edge_case"
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

mkdir -p compressed_files
mkdir -p decompressed_files

for i in {1..100}; do
    echo -e "\e[0;35mGenerating random file $i\e[0m"
    file=$(python3 generate_random_file.py)

    current_failed=false
    for switch in "" "-m"; do
        echo -e "\e[0;35mTesting $file with switch $switch\e[0m"
        echo ""
        for thread_combination in "${thread_combinations[@]}"; do
            echo -e "\e[0;36mTesting with thread combination $thread_combination\e[0m"

            read compress_threads decompress_threads <<< $thread_combination

            basename=$(basename $file)
            rm -f compressed_files/$basename
            rm -f decompressed_files/$basename
            
            width=$(echo "$basename" | sed -n 's/.*x\([0-9]*\)\..*/\1/p')
            echo "compress command: ./huff_codec -c -i $file -o compressed_files/$basename -w $width $switch -t $compress_threads"
            ./huff_codec -c -i $file -o compressed_files/$basename -w $width $switch -t $compress_threads 
            echo "decompress command: ./huff_codec -d -i compressed_files/$basename -o decompressed_files/$basename -t $decompress_threads"
            ./huff_codec -d -i compressed_files/$basename -o decompressed_files/$basename -t $decompress_threads
            diff $file decompressed_files/$basename 2>/dev/null 1>/dev/null
            if [ $? -eq 0 ]; then
                echo -e "\e[0;32mPASSED\e[0m"
            else
                echo -e "\e[0;31mFAILED\e[0m"
                all_tests_passed=false
                current_failed=true
                sleep 2
            fi
            echo ""

            rm -f compressed_files/$basename
            rm -f decompressed_files/$basename
        done
    done

    if [ $current_failed = false ]; then
        echo -e "\e[0;32mPassed test $file\e[0m"
        rm -f $file
    else
        echo -e "\e[0;31mFailed test $file\e[0m"
    fi
    echo ""
done

if [ $all_tests_passed = true ]; then
    echo -e "\e[0;34mAll tests passed\e[0m"
else
    echo -e "\e[0;31mSome tests failed\e[0m"
fi

rm -f compressed_files/*
rm -f decompressed_files/*
