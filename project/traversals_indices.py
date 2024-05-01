import numpy as np

def horizontal_zig_zag_traversal():
    memory = np.linspace(0, 255, 256)
    memory = memory.astype(np.uint8)
    memory = memory.reshape(16, 16)

    reverse = False
    traversal = []
    for i in range(16):
        if reverse:
            row = memory[i][::-1]
        else:
            row = memory[i]

        traversal += row.tolist()
        reverse = not reverse
    
    return traversal

def vertical_zig_zag_traversal():
    memory = np.linspace(0, 255, 256)
    memory = memory.astype(np.uint8)
    memory = memory.reshape(16, 16)
    memory = memory.T

    reverse = False
    traversal = []
    for i in range(16):
        if reverse:
            column = memory[i][::-1]
        else:
            column = memory[i]

        traversal += column.tolist()
        reverse = not reverse
    
    return traversal

def major_diagonal_zig_zag_traversal():
    memory = np.linspace(0, 255, 256)
    memory = memory.astype(np.uint8)
    memory = memory.reshape(16, 16)

    reverse = True
    traversal = []
    for i in range(31):
        if reverse:
            diagonal = memory.diagonal(offset=i - 15, axis1=1, axis2=0)[::-1]
        else:
            diagonal = memory.diagonal(offset=i - 15, axis1=1, axis2=0)

        traversal += diagonal.tolist()
        reverse = not reverse
    
    return traversal

def minor_diagonal_zig_zag_traversal():
    memory = np.linspace(0, 255, 256)
    memory = memory.astype(np.uint8)
    memory = memory.reshape(16, 16)
    memory = np.fliplr(memory)

    reverse = True
    traversal = []
    for i in range(31):
        if reverse:
            diagonal = memory.diagonal(offset=i - 15, axis1=1, axis2=0)[::-1]
        else:
            diagonal = memory.diagonal(offset=i - 15, axis1=1, axis2=0)

        traversal += diagonal.tolist()
        reverse = not reverse
    
    return traversal

def create_indices(traversal):
    rows = []
    columns = []
    for i in range(len(traversal)):
        row = traversal[i] // 16
        column = traversal[i] % 16
        rows.append(str(row))
        columns.append(str(column))
    
    return rows, columns

if __name__ == "__main__":
    traversal = horizontal_zig_zag_traversal()
    rows, columns = create_indices(traversal)
    print(f"constexpr static uint8_t HORIZONTAL_ZIG_ZAG_ROW_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {{{', '.join(rows)}}};")
    print(f"constexpr static uint8_t HORIZONTAL_ZIG_ZAG_COL_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {{{', '.join(columns)}}};")
    print()

    traversal = vertical_zig_zag_traversal()
    rows, columns = create_indices(traversal)
    print(f"constexpr static uint8_t VERTICAL_ZIG_ZAG_ROW_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {{{', '.join(rows)}}};")
    print(f"constexpr static uint8_t VERTICAL_ZIG_ZAG_COL_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {{{', '.join(columns)}}};")
    print()

    traversal = major_diagonal_zig_zag_traversal()
    rows, columns = create_indices(traversal)
    print(f"constexpr static uint8_t MAJOR_DIAGONAL_ZIG_ZAG_ROW_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {{{', '.join(rows)}}};")
    print(f"constexpr static uint8_t MAJOR_DIAGONAL_ZIG_ZAG_COL_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {{{', '.join(columns)}}};")
    print()

    traversal = minor_diagonal_zig_zag_traversal()
    rows, columns = create_indices(traversal)
    print(f"constexpr static uint8_t MINOR_DIAGONAL_ZIG_ZAG_ROW_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {{{', '.join(rows)}}};")
    print(f"constexpr static uint8_t MINOR_DIAGONAL_ZIG_ZAG_COL_INDICES[BLOCK_SIZE * BLOCK_SIZE] = {{{', '.join(columns)}}};")
    print()
