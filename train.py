import os
import time
import shutil
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from utils_muti import *


setup_seed(999)

dataset = MyDataset(r'dataset.h5')

train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiInputModel(nn.Module):
    def __init__(self, time_layers, fa_layers, fa_cnn_layers, time_cnn_layers):
        super(MultiInputModel, self).__init__()

        self.time_block = nn.Sequential(
            nn.Conv1d(9, 16, 2, 2, 1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 2, 2, 1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 2, 1, 1), nn.ReLU(),
            nn.Conv1d(16, 32, 2, 1, 1), nn.ReLU(),
            *self._create_intermediate_conv_layers(32, time_cnn_layers),
            nn.Conv1d(32, 64, 2, 2, 1), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.time_lstm = nn.LSTM(64, 2, time_layers, batch_first=True)

        self.fa_block = nn.Sequential(
            nn.Conv1d(9, 16, 2, 1, 1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 2, 2, 1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 2, 1, 1), nn.ReLU(),
            *self._create_intermediate_conv_layers(32, fa_cnn_layers),
            nn.Conv1d(32, 64, 2, 1, 1), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.fa_lstm = nn.LSTM(64, 2, fa_layers, batch_first=True)

        self.fc_table = nn.Sequential(
            nn.Linear(108, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 8)
        )

        self.fc_final = nn.Sequential(
            nn.Linear(96 + 26 + 8, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(32, 2)
        )

    def _create_intermediate_conv_layers(self, channels, layers):
        intermediate_layers = []
        for _ in range(layers):
            intermediate_layers.append(nn.Conv1d(channels, channels, 2, 1, 1))
            intermediate_layers.append(nn.ReLU())
        return intermediate_layers

    def forward(self, wave, fa, tabular):
        seq1_out = self.time_block(wave).permute(0, 2, 1)
        seq1_out, _ = self.time_lstm(seq1_out)
        seq1_out = seq1_out.flatten(start_dim=1)

        seq2_out = self.fa_block(fa).permute(0, 2, 1)
        seq2_out, _ = self.fa_lstm(seq2_out)
        seq2_out = seq2_out.flatten(start_dim=1)

        tabular = tabular.squeeze(axis=1)
        tabular = self.fc_table(tabular)

        combined = torch.cat((seq1_out, seq2_out, tabular), dim=1)
        return self.fc_final(combined)


params = {
    'time_layers': 2,
    'fa_layers': 2,
    'time_cnn_layers': 2,
    'fa_cnn_layers': 2,
    'epochs': 100,
    'lr': 0.001
}

model = MultiInputModel(
    **{k: params[k] for k in ['time_layers', 'fa_layers', 'fa_cnn_layers', 'time_cnn_layers']}
).to(device)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // 40))

best_val_loss = float('inf')
patience_counter = 0
patience = 10
train_losses, vali_losses, train_accuracies, vali_accuracies = [], [], [], []




save_dir = './runs'
checkpoint_dir = os.path.join(save_dir, 'checkpoints')
best_model_dir = os.path.join(save_dir, 'best_model')
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)

start_time_all = time.time()

for epoch in range(params['epochs']):
    train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    vali_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    vali_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{params['epochs']} "
        f"Train Loss: {train_loss:.4f} Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} Val Acc: {val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), f'{checkpoint_dir}/ck_{round(best_val_loss, 4)}.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered. Patience {patience} epochs without improvement.")
            break


best_model_path = f'{checkpoint_dir}/ck_{round(best_val_loss, 4)}.pth'
model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True), strict=False)
best_model = model.to(device)
torch.save(best_model.state_dict(), f'{best_model_dir}/{round(best_val_loss, 4)}.pth')

labels, preds, avg_loss, accuracy, all_outputs = test_with_results(best_model, test_loader, device, criterion)
print('test loss', round(avg_loss, 4), 'test acc', round(accuracy, 4))
shutil.rmtree(checkpoint_dir, ignore_errors=True)


epochs = range(1, epoch + 2)
data = {
    'Epoch': epochs,
    'Training Loss': train_losses,
    'Validation Loss': vali_losses,
    'Training Accuracy': train_accuracies,
    'Validation Accuracy': vali_accuracies
}
pd.DataFrame(data).to_csv(os.path.join(save_dir, 'epoch_loss.csv'), index=False)


