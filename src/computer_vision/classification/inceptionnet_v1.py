import torch
from typing import Tuple


class InceptionNaiveModule(torch.nn.Module):
    """
    Implementation of GoogleNet's naive inception module.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels_1x1: int,
                 out_channels_3x3: int,
                 out_channels_5x5: int) -> None:
        """
        Method to instantiate object of :class: InceptionNaiveModule

        :param in_channels: Number of channels present in the previous layer
        :param out_channels_1x1: Number of output channels of 1x1 convolution block
        :param out_channels_3x3: Number of output channels of 3x3 convolution block
        :param out_channels_5x5: Number of output channels of 5x5 convolution block

        :returns: Instance of :class:InceptionNaiveModule
        """
        super().__init__()

        self.conv_1x1_layer = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels_1x1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                torch.nn.ReLU())
        self.conv_3x3_layer = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels_3x3, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                                torch.nn.ReLU())
        self.conv_5x5_layer = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels_5x5, kernel_size=5, stride=1, padding=2, dilation=1, bias=False),
                                torch.nn.ReLU())
        self.pooling_layer = torch.nn.Sequential(
                                torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1),
                                torch.nn.ReLU())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to implement forward propagation of input tensor through the current layer

        :param X: input tensor

        :returns: Output of forward propagation of the input tensor
        """
        
        output_1x1 = self.conv_1x1_layer(X)
        output_3x3 = self.conv_3x3_layer(X)
        output_5x5 = self.conv_5x5_layer(X)
        output_pooling_layer = self.pooling_layer(X)

        return torch.concat((output_1x1, output_3x3, output_5x5, output_pooling_layer), dim=1)


class InceptionModule(torch.nn.Module):
    """
    Implementation of GoogleNet's inception module.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels_1x1: int,
                 out_channels_3x3_reduce: int,
                 out_channels_3x3: int,
                 out_channels_5x5_reduce: int,
                 out_channels_5x5: int,
                 out_channels_pool_projection: int) -> None:
        """
        Method to instantiate object of :class: InceptionNaiveModule

        :param in_channels: Number of channels present in the previous layer
        :param out_channels_1x1: Number of output channels of 1x1 convolution block
        :param out_channels_3x3_reduce: Number of output channels of 1x1 convolution layer used to reduce input channels for 3x3 convolution layer
        :param out_channels_3x3: Number of output channels of 3x3 convolution layer
        :param out_channels_5x5_reduce: Number of output channels of 1x1 convolution layer used to reduce input channels for 5x5 convolution layer
        :param out_channels_5x5: Number of output channels of 5x5 convolution layer
        :param out_channels_pool_projection: Number of output channels of 1x1 convolution layer used to reduce the output channels of the 3x3 max pooling layer

        :returns: Instance of :class:InceptionNaiveModule
        """
        super().__init__()

        self.conv_1x1_layer = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels_1x1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                torch.nn.ReLU())
        self.conv_3x3_layer = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels_3x3_reduce, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=out_channels_3x3_reduce, out_channels=out_channels_3x3, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                                torch.nn.ReLU())
        # Implementation of this module in pytorch is an import from tensorflow where there is a bug. The bug is they use a single 3x3 convolution layer instead of a 5x5 layer according to the paper.
        # Refer issue on pytorch's repo https://github.com/pytorch/vision/issues/906
        self.conv_5x5_layer = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels_5x5_reduce, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=out_channels_5x5_reduce, out_channels=out_channels_5x5, kernel_size=5, stride=1, padding=2, dilation=1, bias=False),
                                torch.nn.ReLU())
        self.pooling_layer = torch.nn.Sequential(
                                torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels_pool_projection, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                torch.nn.ReLU())
    

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to implement forward propagation of input tensor through the current layer

        :param X: input tensor

        :returns: Output of forward propagation of the input tensor
        """
        
        output_1x1 = self.conv_1x1_layer(X)
        output_3x3 = self.conv_3x3_layer(X)
        output_5x5 = self.conv_5x5_layer(X)
        output_pooling_layer = self.pooling_layer(X)

        return torch.concat((output_1x1, output_3x3, output_5x5, output_pooling_layer), dim=1)

class InceptionNetV1(torch.nn.Module):
    """
    Class that implements GoogleNet V1 (InceptionNet V1) introduced in the paper
    "Going deeper with convolutions"

    Link to paper:
    https://arxiv.org/pdf/1409.4842

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/inceptionnet_v1.ipynb`
    """

    def __init__(self, num_classes):
        """
        Method to instantiate object of :class: InceptionNetV1

        :param: Number of classes to train the model on. Defaults to 1000 classes

        :returns: Instance of :class: InceptionNetV1
        """
        super().__init__()

        self.conv_layer_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2, dilation=1),
                                torch.nn.ReLU())
        self.max_pooling_1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        self.local_response_norm_1 = torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.conv_layer_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                                torch.nn.ReLU())
        self.local_response_norm_2 = torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.max_pooling_2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        self.inception_3a = InceptionModule(in_channels=192, out_channels_1x1=64, out_channels_3x3_reduce=96, out_channels_3x3=128, out_channels_5x5_reduce=16, out_channels_5x5=32, out_channels_pool_projection=32)
        self.inception_3b = InceptionModule(in_channels=256, out_channels_1x1=128, out_channels_3x3_reduce=128, out_channels_3x3=192, out_channels_5x5_reduce=32, out_channels_5x5=96, out_channels_pool_projection=64)
        self.max_pooling_3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        self.inception_4a = InceptionModule(in_channels=480, out_channels_1x1=192, out_channels_3x3_reduce=96, out_channels_3x3=208, out_channels_5x5_reduce=16, out_channels_5x5=48, out_channels_pool_projection=64)
        self.inception_4b = InceptionModule(in_channels=512, out_channels_1x1=160, out_channels_3x3_reduce=112, out_channels_3x3=224, out_channels_5x5_reduce=24, out_channels_5x5=64, out_channels_pool_projection=64)
        self.inception_4c = InceptionModule(in_channels=512, out_channels_1x1=128, out_channels_3x3_reduce=128, out_channels_3x3=256, out_channels_5x5_reduce=24, out_channels_5x5=64, out_channels_pool_projection=64)
        self.inception_4d = InceptionModule(in_channels=512, out_channels_1x1=112, out_channels_3x3_reduce=144, out_channels_3x3=288, out_channels_5x5_reduce=32, out_channels_5x5=64, out_channels_pool_projection=64)
        self.inception_4e = InceptionModule(in_channels=528, out_channels_1x1=256, out_channels_3x3_reduce=160, out_channels_3x3=320, out_channels_5x5_reduce=32, out_channels_5x5=128, out_channels_pool_projection=128)
        self.max_pooling_4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        self.inception_5a = InceptionModule(in_channels=832, out_channels_1x1=256, out_channels_3x3_reduce=160, out_channels_3x3=320, out_channels_5x5_reduce=32, out_channels_5x5=128, out_channels_pool_projection=128)
        self.inception_5b = InceptionModule(in_channels=832, out_channels_1x1=384, out_channels_3x3_reduce=192, out_channels_3x3=384, out_channels_5x5_reduce=48, out_channels_5x5=128, out_channels_pool_projection=128)
        self.average_pooling = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.auxilary_classifier_1 = torch.nn.Sequential(
                                        torch.nn.AvgPool2d(kernel_size=5, stride=3, padding=0),
                                        torch.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Flatten(),
                                        torch.nn.Linear(in_features=2048, out_features=1024),
                                        torch.nn.Dropout1d(p=0.7),
                                        torch.nn.Linear(in_features=1024,out_features=num_classes))

        self.auxilary_classifier_2 = torch.nn.Sequential(
                                        torch.nn.AvgPool2d(kernel_size=5, stride=3, padding=0),
                                        torch.nn.Conv2d(in_channels=528, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Flatten(),
                                        torch.nn.Linear(in_features=2048, out_features=1024),
                                        torch.nn.Dropout1d(p=0.7),
                                        torch.nn.Linear(in_features=1024,out_features=num_classes))

        self.classifier = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Dropout(p=0.4),
                            torch.nn.Linear(in_features=1024, out_features=num_classes))

    def forward(self, X: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor]:
        """
        Method to implement forward propagation of input tensor through the model

        :param X: input tensor

        :returns: Output of forward propagation of the input tensor
        """

        # Always generate outputs from every classifier
        output, aux_output_1, aux_output_2 = self._forward(X)

        # During inference mode only return the output of the main classifier
        if not self.training:
            return output
        else:
            return  output, aux_output_1, aux_output_2


    def _forward(self, X: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Method to propagate the input tensor to always go through all three classifiers.

        :param X: input tensor

        :returns: A tuple containing outputs of all three classifiers
        """
        
        X = self.conv_layer_1(X)
        X = self.max_pooling_1(X)
        X = self.local_response_norm_1(X)
        X = self.conv_layer_2(X)
        X = self.local_response_norm_2(X)
        X = self.max_pooling_2(X)
        X = self.inception_3a(X)
        X = self.inception_3b(X)
        X = self.max_pooling_3(X)
        X = self.inception_4a(X)
        aux_output_1 = self.auxilary_classifier_1(X)
        X = self.inception_4b(X)
        X = self.inception_4c(X)
        X = self.inception_4d(X)
        aux_output_2 = self.auxilary_classifier_2(X)
        X = self.inception_4e(X)
        X = self.max_pooling_4(X)
        X = self.inception_5a(X)
        X = self.inception_5b(X)

        X = self.average_pooling(X)
        output = self.classifier(X)

        return output, aux_output_1, aux_output_2