from torch import nn
from torchvision.models import MobileNetV2
from torchvision.models.utils import load_state_dict_from_url


class MobileNetV2Encoder(MobileNetV2):
    
    def __init__(self, in_channels, out_classes):
        super().__init__(num_classes=out_classes)
        
        if in_channels != 3:
            self.features[0][0] = nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False)

    def load_pretrained_weights(self):
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')
        del state_dict['features.0.0.weight']
        del state_dict['classifier.1.weight']
        del state_dict['classifier.1.bias']
        self.load_state_dict(state_dict, strict=False)