import torchvision.models as models
from functions import ReverseLayerF
import torchvision
import torch.nn as nn

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        # Load the pretrained ResNet-34 model
        resnet34 = models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)

        # Remove the last classification layer from the ResNet model
        resnet34.fc = nn.Identity()

        # Use the ResNet model as feature extractor
        self.feature = resnet34

        self.class_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(128, 10),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(10, 3),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 2),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 224, 224)

        # Pass input data through Resnet backbone
        features = self.feature(input_data)

        # Process features for classification and domain adaptation
        class_features = features.view(-1, 512)  # Correctly reshape to match the actual size
        reverse_feature = ReverseLayerF.apply(class_features, alpha)
        class_output = self.class_classifier(class_features)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
