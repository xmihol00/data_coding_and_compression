
data_dir="data"
if [ ! -d $data_dir ]; then
    data_dir="../$data_dir"
fi 

for file in $data_dir/*.raw; do
    echo "Testing $file"
    basename=$(basename $file)
    character_count=$(wc -c < $file)
    echo "compress command: ./huff_codec -c -i $file -o compressed_files/$basename -w $character_count"
    ./huff_codec -c -i $file -o compressed_files/$basename -w $character_count 2>/dev/null
    echo "decompress command: ./huff_codec -d -i compressed_files/$basename -o decompressed_files/$basename"
    ./huff_codec -d -i compressed_files/$basename -o decompressed_files/$basename 2>/dev/null
    diff $file decompressed_files/$basename
done
