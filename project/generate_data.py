
if False:
    length = 0
    with open('test_files/a-j_increasing.bin', 'wb') as f:
        for i, j in zip(range(ord('a'), ord('k')), range(0, 256)):
            j = 2 ** j
            byte_str = i.to_bytes(1, byteorder='big')
            byte_str = byte_str * j
            f.write(byte_str)
            print(byte_str)
            length += len(byte_str)

    print(f'Length: {length}')

if False:
    length = 0
    with open('test_files/A-z_increasing.txt', 'wb') as f:
        for i, j in zip(range(ord('A'), ord('z')+1), range(1, 257)):
            byte_str = i.to_bytes(1, byteorder='big')
            byte_str = byte_str * j
            f.write(byte_str)
            print(byte_str)
            length += len(byte_str)

    print(f'Length: {length}')

if False:
    length = 0
    with open('test_files/all_same.bin', 'wb') as f:
        for _ in range(0, 2):
            for i in range(0, 256):
                byte_str = i.to_bytes(1, byteorder='big')
                f.write(byte_str)
                print(byte_str)
                length += len(byte_str)

    print(f'Length: {length}')

if False:
    length = 0
    RANGE_LOW = ord('A') - 1
    RANGE_HIGH = ord('Z') + 5
    with open('../all_A-Z_pow2.bin', 'wb') as f:
        for i in range(0, 256):
            if i < RANGE_LOW or i > RANGE_HIGH:
                byte_str = i.to_bytes(1, byteorder='big')
                f.write(byte_str)
                print(byte_str)
                length += len(byte_str)

        for i in range(RANGE_LOW, RANGE_HIGH + 1):
            print(2 ** (i - RANGE_LOW))
            byte_str = i.to_bytes(1, byteorder='big')
            byte_str = byte_str * (2 ** (i - RANGE_LOW))
            f.write(byte_str)
            length += len(byte_str)
        
        byte_str = b'\x00' * 31
        length += len(byte_str)
        f.write(byte_str)

    print(f'Length: {length}')

if False:
    length = 0
    RANGE_LOW = ord('A') - 1
    RANGE_HIGH = ord('Z') + 5
    with open('test_files/4G-1.bin', 'wb') as f:
        for i in range(RANGE_LOW, RANGE_HIGH + 1):
            print(2 ** (i - RANGE_LOW))
            byte_str = i.to_bytes(1, byteorder='big')
            byte_str = byte_str * (2 ** (i - RANGE_LOW))
            f.write(byte_str)
            length += len(byte_str)
        
        for i in range(0, 256):
            if (i < RANGE_LOW or i > RANGE_HIGH) and length < 2 ** 32:
                byte_str = i.to_bytes(1, byteorder='big')
                print("Adding: ", byte_str)
                f.write(byte_str)
                print(byte_str)
                length += len(byte_str)

    print(f'Length: {length}')

if True:
    def fibonaci(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
            yield a
    
    RANGE_LOW = ord('A')
    RANGE_HIGH = ord('Z') + 1
    with open('test_files/fibonaci_A-Z.txt', 'wb') as f:
        for i, char in zip(fibonaci(RANGE_HIGH - RANGE_LOW), range(RANGE_LOW, RANGE_HIGH)):
            byte_str = char.to_bytes(1, byteorder='big')
            byte_str = byte_str * i
            f.write(byte_str)