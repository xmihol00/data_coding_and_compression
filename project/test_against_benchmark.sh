
data_dir="data"
if [ ! -d $data_dir ]; then
    data_dir="../$data_dir"
fi 

all_tests_passed=true
for file in $data_dir/*; do
    echo "Testing $file"
    basename=$(basename $file)
    character_count=$(wc -c < $file)
    echo "compress command: ./huff_codec -c -i $file -o compressed_files/$basename -w $character_count"
    ./huff_codec -c -i $file -o compressed_files/$basename -w $character_count 2>/dev/null
    echo "decompress command: ./huff_codec -d -i compressed_files/$basename -o decompressed_files/$basename"
    ./huff_codec -d -i compressed_files/$basename -o decompressed_files/$basename 2>/dev/null
    diff $file decompressed_files/$basename
    if [ $? -eq 0 ]; then
        echo -e "\e[0;32mPASSED\e[0m"
    else
        echo -e "\e[0;31mFAILED\e[0m"
        all_tests_passed=false
    fi
done

if [ $all_tests_passed = true ]; then
    echo -e "\e[0;34mAll tests passed\e[0m"
else
    echo -e "\e[0;31mSome tests failed\e[0m"
fi