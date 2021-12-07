import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.NUM_CLASSES = 2

        self.layer1 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, self.NUM_CLASSES)

        self.dropout = nn.Dropout(p = 0.25)

        self.model = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            nn.Flatten(),
            self.dropout,
            self.fc1,
            self.dropout,
            self.fc2
        )

    def forward(self, x):
        x = self.model(x)
        return 