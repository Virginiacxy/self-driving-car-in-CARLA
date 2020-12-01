from torch import nn
from torchvision.models import MobileNetV2
from torchvision.models.utils import load_state_dict_from_url


class MobileNetV2Encoder(MobileNetV2):
    def __init__(self, in_channels, out_classes):
        super().__init__(num_classes=out_classes)
        
        if in_channels != 3:
            self.features[0][0] = nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False)

        self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')
        del state_dict['features.0.0.weight']
        del state_dict['classifier.1.weight']
        del state_dict['classifier.1.bias']
        self.load_state_dict(state_dict, strict=False)


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, out_classes):
        super().__init__()
        self.out_classes = out_classes
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 3, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 3, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512, 3, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 1024, 3, 2, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1024, out_classes, 1)
        )

    def forward(self, x):
        return self.model(x).view(-1, self.out_classes)