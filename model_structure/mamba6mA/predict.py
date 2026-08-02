import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from dna_map import read_dna_sequences, map_to_indices, PAD_IDX
from net.mamba import Mamba
from net.model_args import ModelArgs
import numpy as np
import torch

batch_size = 64

def test_main(test_path, model_path, output_excel_path):

    test_file_path = test_path
    test_sequences = read_dna_sequences(test_file_path)

    print("Encoding test sequences", flush=True)
    test_encoded = map_to_indices(test_sequences)

    X_test = np.array(test_encoded).astype(np.int64)
    X_test_tensor = torch.tensor(X_test, dtype=torch.long)

    test_dataset = TensorDataset(X_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_args = ModelArgs()
    model_args.__post_init__()
    mamba_model = Mamba(model_args)


    mamba_model.load_state_dict(torch.load(model_path))
    mamba_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mamba_model.to(device)

    logits_list = []

    with torch.no_grad():
        for inputs in tqdm(test_dataloader, desc='Testing'):
            inputs = inputs[0].to(device)
            padding_mask = (inputs == PAD_IDX)
            try:
                batch_logits = mamba_model(inputs, padding_mask=padding_mask)
            except TypeError:
                batch_logits = mamba_model(inputs)
            logits_list.extend(batch_logits.cpu().numpy())


    logits_df = pd.DataFrame(logits_list, columns=['Logits'])
    logits_df.to_excel(output_excel_path, index=False)
    print(f'Logits saved to {output_excel_path}')

if __name__ == "__main__":

    test_main(
        '',  #预测数据
        './model/C.elegans.pth',
        './result/output_logits.xlsx'
    )
