from torch import nn


class ModelArchitectureConvolution_v1(nn.Module):
    def __init__(self, input_channels, x_size, y_size, number_of_classes, dropout_conv=0.25, dropout_linear=0.5):
        super(ModelArchitectureConvolution_v1, self).__init__()


        self.sequential = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_conv),
            nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Dropout2d(dropout_conv),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(dropout_linear),
            nn.Linear(128, number_of_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.sequential(x)

class ModelArchitectureLinear_v1(nn.Module):
    def __init__(self, input_channels, x_size, y_size, number_of_classes, dropout_linear=0.5):
        super(ModelArchitectureLinear_v1, self).__init__()

        self.sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * x_size * y_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_linear),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_linear),
            nn.Linear(128, number_of_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.sequential(x)

