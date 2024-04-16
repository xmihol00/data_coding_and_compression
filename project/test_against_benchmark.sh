
for file in data/*; do
    echo "Testing $file"
    basename=$(basename $file)
    character_count=$(wc -c < $file)
    ./huff_codec -c -i $file -o compressed_files/$basename -w $character_count 2>/dev/null
    ./huff_codec -d -i compressed_files/$basename -o decompressed_files/$basename 2>/dev/null
    diff -s $file decompressed_files/$basename
done
