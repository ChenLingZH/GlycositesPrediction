from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from dna_map import read_dna_sequences, map_to_indices, PAD_IDX
from net.mamba import Mamba
from net.model_args import ModelArgs
import numpy as np
import torch
import pandas as pd




batch_size = 64
learning_rate = 0.00005
num_epochs = 200
def train_main(neg_path, pos_path, val_neg_path, val_pos_path):
    def load_sequences(path):
        # Support both: txt(one sequence per line) and csv(column name 'window')
        if str(path).lower().endswith('.csv'):
            df = pd.read_csv(path)
            col = 'window' if 'window' in df.columns else df.columns[0]
            seqs = df[col].astype(str).tolist()
            # Convert RNA U -> T and keep as list of characters.
            return [list(s.strip().upper().replace('U', 'T')) for s in seqs if s and s.strip()]
        return read_dna_sequences(path)

    negative_sequences = load_sequences(neg_path)
    positive_sequences = load_sequences(pos_path)


    negative_encoded = map_to_indices(negative_sequences)
    positive_encoded = map_to_indices(positive_sequences)

    X = np.concatenate([negative_encoded, positive_encoded], axis=0).astype(np.int64)
    y = np.concatenate([np.zeros(len(negative_encoded)), np.ones(len(positive_encoded))]).astype(np.int64)

    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.long)

    val_negative_sequences = load_sequences(val_neg_path)
    val_positive_sequences = load_sequences(val_pos_path)


    val_negative_encoded = map_to_indices(val_negative_sequences)
    val_positive_encoded = map_to_indices(val_positive_sequences)

    X_val = np.concatenate([val_negative_encoded, val_positive_encoded], axis=0)
    y_val = np.concatenate([np.zeros(len(val_negative_encoded)), np.ones(len(val_positive_encoded))])

    X_val_tensor = torch.tensor(X_val, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model_args = ModelArgs()
    model_args.__post_init__()
    mamba_model = Mamba(model_args)

    optimizer = optim.Adam(mamba_model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    mamba_model.to(device)
    best_val_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = 30

    for epoch in range(num_epochs):
        mamba_model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device).float()
            padding_mask = (inputs == PAD_IDX)
            optimizer.zero_grad()
            try:
                logits = mamba_model(inputs, padding_mask=padding_mask)
            except TypeError:
                logits = mamba_model(inputs)
            logits = logits.squeeze(-1).view(-1)
            labels = labels.view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = torch.round(probs)
            all_preds.extend(preds.detach().cpu().numpy().astype(np.int64))
            all_labels.extend(labels.detach().cpu().numpy().astype(np.int64))

        mamba_model.eval()
        with torch.no_grad():
            val_preds = []
            val_probs = []
            for inputs, labels in tqdm(val_dataloader, desc='Validating'):
                inputs = inputs.to(device)
                padding_mask = (inputs == PAD_IDX)
                try:
                    batch_logits = mamba_model(inputs, padding_mask=padding_mask)
                except TypeError:
                    batch_logits = mamba_model(inputs)
                batch_logits = batch_logits.squeeze(-1)
                batch_probs = torch.sigmoid(batch_logits)
                batch_preds = torch.round(batch_probs).detach().cpu().numpy().astype(np.int64)
                val_preds.extend(batch_preds.reshape(-1))
                val_probs.extend(batch_probs.detach().cpu().numpy().reshape(-1))

            y_val = y_val_tensor.cpu().numpy().reshape(-1).astype(np.int64)
            val_preds_arr = np.array(val_preds, dtype=np.int64).reshape(-1)
            val_probs_arr = np.array(val_probs, dtype=np.float64).reshape(-1)
            val_accuracy = accuracy_score(y_val, val_preds_arr)
            val_auc = roc_auc_score(y_val, val_probs_arr)
            val_mcc = matthews_corrcoef(y_val, val_preds_arr)
            tn, fp, fn, tp = confusion_matrix(y_val, val_preds_arr).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)

            print(f'Validation Metrics: Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}, MCC: {val_mcc:.4f}, '
                  f'Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}')

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(mamba_model.state_dict(), 'best_mamba_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                break

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)

        print(f'Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')


    print(f'best epoch {best_epoch} ')


if __name__ == "__main__":
     train_main('./negative_train_cdhit80.csv',
                './positive_train_cdhit80.csv',
                './negative_test_cdhit80.csv',
                './positive_test_cdhit80.csv')
