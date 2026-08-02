import numpy as np

PAD_IDX = 4  # N统一用4表示

def read_dna_sequences(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            # Treat RNA base U as DNA base T.
            sequence = list(line.strip().upper().replace('U', 'T'))
            sequences.append(sequence)
    return sequences


def map_to_indices(sequences):
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': PAD_IDX}
    mapped_sequences = []
    for sequence in sequences:
        # .get(base, PAD_IDX) 遇到其他未知字符也当N处理
        mapped_sequence = [base_to_index.get(base, PAD_IDX) for base in sequence]
        mapped_sequences.append(mapped_sequence)
    return np.array(mapped_sequences)
