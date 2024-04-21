
data_dir="data"
if [ ! -d $data_dir ]; then
    data_dir="../$data_dir"
fi 

all_tests_passed=true

for switch in "" "-a" "-m" "-m -a"; do
    echo ""
    echo -e "Testing with switch \e[0;34m$switch\e[0m"
    rm -f compressed_files/*
    rm -f decompressed_files/*
    for file in $data_dir/*; do
        echo "Testing $file"
        basename=$(basename $file)
        file_size=$(wc -c < $file)
        width=$(echo "sqrt($file_size)" | bc -l | cut -d'.' -f1)
        echo "compress command: ./huff_codec -c -i $file -o compressed_files/$basename -w $width $switch"
        ./huff_codec -c -i $file -o compressed_files/$basename -w $width $switch 2>/dev/null
        echo "decompress command: ./huff_codec -d -i compressed_files/$basename -o decompressed_files/$basename"
        ./huff_codec -d -i compressed_files/$basename -o decompressed_files/$basename 2>/dev/null
        diff $file decompressed_files/$basename 2>/dev/null 1>/dev/null
        if [ $? -eq 0 ]; then
            echo -e "\e[0;32mPASSED\e[0m"
        else
            echo -e "\e[0;31mFAILED\e[0m"
            all_tests_passed=false
        fi
        echo ""
    done
done

if [ $all_tests_passed = true ]; then
    echo -e "\e[0;34mAll tests passed\e[0m"
else
    echo -e "\e[0;31mSome tests failed\e[0m"
fi

rm -f compressed_files/*
rm -f decompressed_files/*
