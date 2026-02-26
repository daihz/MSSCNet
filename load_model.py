import torch
from torch.utils.data import DataLoader
from utils import MyDataset, test_with_results
import torch.nn as nn



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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def load_model(checkpoint_path, device):
    model = MultiInputModel(**{k: params[k] for k in ['time_layers','fa_layers','fa_cnn_layers','time_cnn_layers']}).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model("./model.pth", device)

    test_dataset = MyDataset("./data/test.h5")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    labels, preds, loss, acc, outputs = test_with_results(model, test_loader, device, criterion)

    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")


