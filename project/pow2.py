
#length = 0
#with open('test_files/a-j_pow2.bin', 'wb') as f:
#    for i, j in zip(range(ord('a'), ord('k')), range(0, 256)):
#        j = 2 ** j
#        byte_str = i.to_bytes(1, byteorder='big')
#        byte_str = byte_str * j
#        f.write(byte_str)
#        print(byte_str)
#        length += len(byte_str)
#
#print(f'Length: {length}')

length = 0
with open('test_files/A-z_pow2.bin', 'wb') as f:
    for i, j in zip(range(ord('A'), ord('z')+1), range(1, 257)):
        byte_str = i.to_bytes(1, byteorder='big')
        byte_str = byte_str * j
        f.write(byte_str)
        print(byte_str)
        length += len(byte_str)

print(f'Length: {length}')