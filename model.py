import torchvision.models as models
from functions import ReverseLayerF
import torchvision
import torch.nn as nn


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        # Load the pretrained ResNet-18 model
        resnet18 = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove the last classification layer from the ResNet model
        resnet18.fc = nn.Identity()

        # Use the ResNet model as feature extractor
        self.feature = resnet18
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(512, 128))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout(p=0.3))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(256))
        # self.class_classifier.add_module('c_fc2', nn.Linear(256, 128))
        # # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # # self.class_classifier.add_module('c_drop2', nn.Dropout(p=0.25))
        # # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(128))
        # self.class_classifier.add_module('c_fc3', nn.Linear(128,10))
        # # self.class_classifier.add_module('c_relu3', nn.ReLU(True))
        # # self.class_classifier.add_module('c_drop3', nn.Dropout(p =0.1))
        # # self.class_classifier.add_module('c_bn3', nn.BatchNorm1d(10))
        self.class_classifier.add_module('c_fc4', nn.Linear(128,3))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(512, 128))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_drop1', nn.Dropout(p=0.3))
        # self.domain_classifier.add_module('d_fc2', nn.Linear(256, 64))
        # self.domain_classifier.add_module('d_drop2', nn.Dropout(p=0.25))
        self.domain_classifier.add_module('d_fc3', nn.Linear(128, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))


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
