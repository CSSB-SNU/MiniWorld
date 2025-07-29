def get_seq_group(L):
    GROUP_LENGTHS = [32 * 32, 64 * 64, 128 * 128, 256 * 256]
    for length in GROUP_LENGTHS:
        if L <= length:
            return length
    return GROUP_LENGTHS[-1]
