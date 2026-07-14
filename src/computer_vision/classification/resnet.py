import torch

# Components of ResNet 18 and 34
class ConvBlockV1(torch.nn.Module):
    """
    Class that implements residual connections across two convolution layer described in the paper
    "Deep Residual Learning for Image Recognition".

    This block does not perform reduction of feature maps 
    """
    def __init__(self, in_channels: int) -> None:
        """
        Method to instantiate  object of :class: ConvBlockV1

        :param in_channels: Number of incoming channels of the input tensor
        
        :returns: An object of :class: ConvBlockV1
        """

        super().__init__()

        self.conv_layer_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1, dilation=1, bias=False),
                                torch.nn.BatchNorm2d(num_features=in_channels),
                                torch.nn.ReLU())

        self.conv_layer_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1, dilation=1, bias=False),
                                torch.nn.BatchNorm2d(num_features=in_channels))
    
        self.relu_layer = torch.nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the ConvBlock.

        :param X: Input tensor

        :retuns: Result of forward propagation through ConvBlock
        """

        output = self.conv_layer_1(X)
        output = self.conv_layer_2(output)

        return self.relu_layer(output + X)


class IdentityReductionV1(torch.nn.Module):
    """
    Class that implements residual connections across two convolution layers with feature map reduction
    and two folds increase of the number of dimension using "identity mapping" technique which uses 
    zero padded feature maps to increase the dimensions as described in the paper 
    "Deep Residual Learning for Image Recognition" 
    """

    def __init__(self, in_channels: int) -> None:
        """
        Method to instantiate  object of :class: IdentityReductionV1

        :param in_channels: Number of incoming channels of the input tensor
        
        :returns: An object of :class: IdentityReductionV1
        """

        super().__init__()

        self.conv_layer_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, padding=1, stride=2, dilation=1, bias=False),
                                torch.nn.BatchNorm2d(num_features= in_channels * 2),
                                torch.nn.ReLU())

        self.conv_layer_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels * 2, kernel_size=3, padding=1, stride=1, dilation=1, bias=False),
                                torch.nn.BatchNorm2d(num_features=in_channels * 2))

        self.relu_layer = torch.nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the IdentityReductionV1.

        :param X: Input tensor

        :retuns: Result of forward propagation through IdentityReductionV1
        """
    
        output = self.conv_layer_1(X)
        output = self.conv_layer_2(output)

        down_sampled_input = X[:, :, ::2, ::2]
        zero_tensor = torch.zeros_like(down_sampled_input)
        identity_output = torch.concat((down_sampled_input, zero_tensor), dim=1)

        return self.relu_layer(output + identity_output)


class PartialProjectionReductionV1(torch.nn.Module):
    """
    Class that implements residual connections across two convolution layers with feature map reduction
    and two folds increase of the number of dimension using "identity mapping" and projecting the reduced 
    input to increase the number of dimension which is termed as partial projection as described in the paper 
    "Deep Residual Learning for Image Recognition" 
    """

    def __init__(self, in_channels: int) -> None:
        """
        Method to instantiate  object of :class: PartialProjectionReductionV1

        :param in_channels: Number of incoming channels of the input tensor
        
        :returns: An object of :class: PartialProjectionReductionV1
        """

        super().__init__()

        self.conv_layer_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, padding=1, stride=2, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=in_channels * 2),
                                torch.nn.ReLU())

        self.conv_layer_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels * 2, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=in_channels * 2))

        self.projection_layer = torch.nn.Sequential(
                                    torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=2, padding=0, dilation=1, bias=True),
                                    torch.nn.BatchNorm2d(num_features=in_channels))

        self.relu_layer = torch.nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the PartialProjectionReductionV1.

        :param X: Input tensor

        :retuns: Result of forward propagation through PartialProjectionReductionV1
        """

        output = self.conv_layer_1(X)
        output = self.conv_layer_2(output)

        down_sampled_input = X[:, :, ::2, ::2]
        down_sampled_projections = self.projection_layer(X)
        projection_output = torch.concat((down_sampled_input, down_sampled_projections), dim=1)

        return self.relu_layer(output + projection_output)
    

class ProjectionReductionV1(torch.nn.Module):
    """
    Class that implements residual connections across two convolution layers with feature map reduction
    and two folds increase of the number of dimension by completely projecting the input into an output
    of twice the number of feature maps using projections as described in the paper 
    "Deep Residual Learning for Image Recognition" 
    """

    def __init__(self, in_channels: int) -> None:
        """
        Method to instantiate  object of :class: ProjectionReductionV1

        :param in_channels: Number of incoming channels of the input tensor
        
        :returns: An object of :class: ProjectionReductionV1
        """

        super().__init__()

        self.conv_layer_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, padding=1, stride=2, dilation=1, bias=False),
                                torch.nn.BatchNorm2d(num_features=in_channels * 2),
                                torch.nn.ReLU())

        self.conv_layer_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels * 2, kernel_size=3, padding=1, stride=1, dilation=1, bias=False),
                                torch.nn.BatchNorm2d(num_features=in_channels * 2))

        self.projection_layer = torch.nn.Sequential(
                                    torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1, stride=2, padding=0, dilation=1, bias=False),
                                    torch.nn.BatchNorm2d(num_features=in_channels * 2))

        self.relu_layer = torch.nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the ProjectionReductionV1.

        :param X: Input tensor

        :retuns: Result of forward propagation through ProjectionReductionV1
        """

        output = self.conv_layer_1(X)
        output = self.conv_layer_2(output)

        projection_output = self.projection_layer(X)

        return self.relu_layer(output + projection_output)


class ResNetBlockIdentityV1(torch.nn.Module):
    """
    Class that represents empherical representation of residual layers with identity mapping technique
    described in the paper "Deep Residual Learning for Image Recognition".

    This block is for ResNet 18 and 34 layer architecture
    """
    def __init__(self, in_channels: int, num_blocks: int, reduce: bool = True):
        """
        Method to instantiate  object of :class: ResNetBlockIdentityV1

        :param in_channels: Number of incoming channels of the input tensor
        :param num_blocks: Total number of ConvBlock and Reduction blocks present in this residual connection layer block
        :param reduce: If reduction of feature maps and increase in dimenions needs to take place. Defaults to True
     
        :returns: An object of :class: ResNetBlockIdentityV1
        """

        super().__init__()

        self.conv_layers = torch.nn.ModuleList()
        if reduce:
            self.conv_layers.append(IdentityReductionV1(in_channels=in_channels))
            in_channels = in_channels * 2
        else:
            self.conv_layers.append(ConvBlockV1(in_channels=in_channels))

        for _ in range(num_blocks - 1):
            self.conv_layers.append(ConvBlockV1(in_channels=in_channels))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the ResNetBlockIdentityV1.

        :param X: Input tensor

        :retuns: Result of forward propagation through ResNetBlockIdentityV1
        """

        for layer in self.conv_layers:
            X = layer(X)
        
        return X


class ResNetBlockPartialProjectionV1(torch.nn.Module):
    """
    Class that represents empherical representation of residual layers with partial projection technique
    described in the paper "Deep Residual Learning for Image Recognition".

    This block is for ResNet 18 and 34 layer architecture
    """
    def __init__(self, in_channels: int, num_blocks: int, reduce: bool = True):
        """
        Method to instantiate  object of :class: ResNetBlockPartialProjectionV1

        :param in_channels: Number of incoming channels of the input tensor
        :param num_blocks: Total number of ConvBlock and Reduction blocks present in this residual connection layer block
        :param reduce: If reduction of feature maps and increase in dimenions needs to take place. Defaults to True
     
        :returns: An object of :class: ResNetBlockPartialProjectionV1
        """

        super().__init__()

        self.conv_layers = torch.nn.ModuleList()
        if reduce:
            self.conv_layers.append(PartialProjectionReductionV1(in_channels=in_channels))
            in_channels = in_channels * 2
        else:
            self.conv_layers.append(ConvBlockV1(in_channels=in_channels))

        for _ in range(num_blocks - 1):
            self.conv_layers.append(ConvBlockV1(in_channels=in_channels))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the ResNetBlockPartialProjectionV1.

        :param X: Input tensor

        :retuns: Result of forward propagation through ResNetBlockPartialProjectionV1
        """

        for layer in self.conv_layers:
            X = layer(X)
        
        return X


class ResNetBlockProjectionV1(torch.nn.Module):
    """
    Class that represents empherical representation of residual layers with fully projection technique
    described in the paper "Deep Residual Learning for Image Recognition".

    this is the layer architecture that lead to the highest performance out of the three approaches described
    in the paper and is used by pytorch in it's implementation of ResNet 18 and 34.
    """
    def __init__(self, in_channels: int, num_blocks: int, reduce: bool = True):
        """
        Method to instantiate  object of :class: ResNetBlockProjectionV1

        :param in_channels: Number of incoming channels of the input tensor
        :param num_blocks: Total number of ConvBlock and Reduction blocks present in this residual connection layer block
        :param reduce: If reduction of feature maps and increase in dimenions needs to take place. Defaults to True
     
        :returns: An object of :class: ResNetBlockProjectionV1
        """

        super().__init__()

        self.conv_layers = torch.nn.ModuleList()
        if reduce:
            self.conv_layers.append(ProjectionReductionV1(in_channels=in_channels))
            in_channels = in_channels * 2
        else:
            self.conv_layers.append(ConvBlockV1(in_channels=in_channels))

        for _ in range(num_blocks - 1):
            self.conv_layers.append(ConvBlockV1(in_channels=in_channels))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the ResNetBlockProjectionV1.

        :param X: Input tensor

        :retuns: Result of forward propagation through ResNetBlockProjectionV1
        """

        for layer in self.conv_layers:
            X = layer(X)
        
        return X

# Components of ResNet 50, 101 and 152
class ConvBlockV2(torch.nn.Module):
    """
    Class that implements residual connections across three convolution layer described in the paper
    "Deep Residual Learning for Image Recognition".

    This block does not perform reduction of feature maps 
    """
    def __init__(self, in_channels: int, continuity_channels: int) -> None:
        """
        Method to instantiate  object of :class: ConvBlockV2

        :param in_channels: Number of incoming channels of the input tensor
        :param continuity_channels: Number of channels that will be required inside the convolution block
    
        :returns: An object of :class: ConvBlockV2
        """

        super().__init__()

        self.in_channels = in_channels
        self.continuity_channels = continuity_channels
        self.conv_layer_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels, out_channels=continuity_channels, kernel_size=1, padding=0, stride=1, dilation=1, bias=False),
                                torch.nn.BatchNorm2d(num_features=continuity_channels),
                                torch.nn.ReLU())

        self.conv_layer_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=continuity_channels, out_channels=continuity_channels, kernel_size=3, padding=1, stride=1, dilation=1, bias=False),
                                torch.nn.BatchNorm2d(num_features=continuity_channels),
                                torch.nn.ReLU())
    
        self.conv_layer_3 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=continuity_channels, out_channels=continuity_channels * 4, kernel_size=1, padding=0, stride=1, dilation=1, bias=False),
                                torch.nn.BatchNorm2d(num_features=continuity_channels * 4))


        if self.in_channels == self.continuity_channels * 4:
            self.projection_layer = torch.nn.Identity()
        else:
            self.projection_layer = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=in_channels, out_channels=continuity_channels * 4, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                            torch.nn.BatchNorm2d(num_features=continuity_channels * 4))
        
        self.relu_layer = torch.nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the ConvBlockV2.

        :param X: Input tensor

        :retuns: Result of forward propagation through ConvBlockV2
        """

        output = self.conv_layer_1(X)
        output = self.conv_layer_2(output)
        output = self.conv_layer_3(output)

        projected_output = self.projection_layer(X)
        return self.relu_layer(output + projected_output)


class ProjectionReductionV2(torch.nn.Module):
    """
    Class that implements residual connections across three convolution layers with feature map reduction
    and two folds increase of the number of dimension using projections to increase the number of dimensions
    described in the paper "Deep Residual Learning for Image Recognition".

    This is the technique used for the deeper ResNets
    """

    def __init__(self, in_channels: int, continuity_channels: int) -> None:
        """
        Method to instantiate  object of :class: ConvBlockV2

        :param in_channels: Number of incoming channels of the input tensor
        :param continuity_channels: Number of channels that will be required inside the convolution block
    
        :returns: An object of :class: ConvBlockV2
        """

        super().__init__()
        self.in_channels = in_channels
        self.continuity_channels = continuity_channels
        self.conv_layer_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=in_channels, out_channels=continuity_channels, kernel_size=1, padding=0, stride=2, dilation=1, bias=False),
                                torch.nn.BatchNorm2d(num_features=continuity_channels),
                                torch.nn.ReLU())

        self.conv_layer_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=continuity_channels, out_channels=continuity_channels, kernel_size=3, padding=1, stride=1, dilation=1, bias=False),
                                torch.nn.BatchNorm2d(num_features=continuity_channels),
                                torch.nn.ReLU())
    
        self.conv_layer_3 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=continuity_channels, out_channels=continuity_channels * 4, kernel_size=1, padding=0, stride=1, dilation=1, bias=False),
                                torch.nn.BatchNorm2d(num_features=continuity_channels * 4))

        if self.in_channels == self.continuity_channels * 4:
            self.projection_layer = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=2, padding=0, dilation=1, bias=False),
                            torch.nn.BatchNorm2d(num_features=in_channels))
        else:
            self.projection_layer = torch.nn.Sequential(
                                        torch.nn.Conv2d(in_channels=in_channels, out_channels=continuity_channels * 4, kernel_size=1, stride=2, padding=0, dilation=1, bias=False),
                                        torch.nn.BatchNorm2d(num_features=continuity_channels * 4))

        self.relu_layer = torch.nn.ReLU()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the PartialProjectionReductionV1.

        :param X: Input tensor

        :retuns: Result of forward propagation through PartialProjectionReductionV1
        """

        output = self.conv_layer_1(X)
        output = self.conv_layer_2(output)
        output = self.conv_layer_3(output)

        down_sampled_projections = self.projection_layer(X)

        return self.relu_layer(output + down_sampled_projections)


class ResNetBlockPartialProjectionV2(torch.nn.Module):
    """
    Class that represents empherical representation of residual layers with partial projection technique
    described in the paper "Deep Residual Learning for Image Recognition".

    This block is for ResNet 50, 101 and 152 layer architecture
    """
    def __init__(self, in_channels: int, continuity_channels: int, num_blocks: int, reduce: bool = True):
        """
        Method to instantiate  object of :class: ResNetBlockPartialProjectionV1

        :param in_channels: Number of incoming channels of the input tensor
        :param continuity_channels: Number of channels that will be required inside the convolution block
        :param num_blocks: Total number of ConvBlock and Reduction blocks present in this residual connection layer block
        :param reduce: If reduction of feature maps and increase in dimenions needs to take place. Defaults to True
     
        :returns: An object of :class: ResNetBlockPartialProjectionV1
        """

        super().__init__()

        self.conv_layers = torch.nn.ModuleList()
        if reduce:
            self.conv_layers.append(ProjectionReductionV2(in_channels=in_channels, continuity_channels=continuity_channels))
        else:
            self.conv_layers.append(ConvBlockV2(in_channels=in_channels, continuity_channels=continuity_channels))

        for _ in range(num_blocks - 1):
            self.conv_layers.append(ConvBlockV2(in_channels=continuity_channels * 4, continuity_channels=continuity_channels))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the ResNetBlockPartialProjectionV1.

        :param X: Input tensor

        :retuns: Result of forward propagation through ResNetBlockPartialProjectionV1
        """

        for layer in self.conv_layers:
            X = layer(X)
        
        return X


class ResNetBase(torch.nn.Module):
    """
    Base class that all ResNet models inherit. This class contains the layers that are same
    across all 5 ResNet models described in the paper "Deep Residual Learning for Image Recognition"
    """

    def __init__(self) -> None:
        """
        Method to instantiate  object of :class: ResNetBase
     
        :returns: An object of :class: ResNetBase
        """
        super().__init__()

        # Input: (N, 3, 224, 224) -> Output: (N, 64, 56, 56)
        self.conv_layer_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, dilation=1, bias=False),
                                torch.nn.BatchNorm2d(num_features=64),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1))


class ResNet18(ResNetBase):
    """
    Implementation of ResNet 18 image classification model described in the paper "Deep Residual Learning for Image Recognition"
    """

    def __init__(self, num_classes: int) -> None:
        """
        Method to instantiate  object of :class: ResNet18

        :param num_classes: Number of classes to train the model on.
     
        :returns: An object of :class: ResNet18
        """

        super().__init__()

        # Input: (N, 64, 56, 56) -> Output: (N, 64, 56, 56)
        self.resnet_layer_1 = ResNetBlockProjectionV1(in_channels=64, num_blocks=2, reduce=False)
        # Input: (N, 64, 56, 56) -> Output: (N, 128, 28, 28)
        self.resnet_layer_2 = ResNetBlockProjectionV1(in_channels=64, num_blocks=2, reduce=True)
        # Input: (N, 128, 28, 28) -> Output: (N, 256, 14, 14)
        self.resnet_layer_3 = ResNetBlockProjectionV1(in_channels=128, num_blocks=2, reduce=True)
        # Input: (N, 256, 14, 14) -> Output: (N, 512, 7, 7)
        self.resnet_layer_4 = ResNetBlockProjectionV1(in_channels=256, num_blocks=2, reduce=True)

        self.average_pooling_layer = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.classifier = torch.nn.Sequential(torch.nn.Flatten(),
                                              torch.nn.Linear(in_features=512, out_features=num_classes))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the ResNet18.

        :param X: Input tensor

        :retuns: Result of forward propagation through ResNet18
        """

        X = self.conv_layer_1(X)
        X = self.resnet_layer_1(X)
        X = self.resnet_layer_2(X)
        X = self.resnet_layer_3(X)
        X = self.resnet_layer_4(X)
        X = self.average_pooling_layer(X)
        
        return self.classifier(X)


class ResNet34(ResNetBase):
    """
    Implementation of ResNet 34 layer image classification model described in the paper "Deep Residual Learning for Image Recognition"
    """
    def __init__(self, num_classes: int) -> None:
        """
        Method to instantiate  object of :class: ResNet34

        :param num_classes: Number of classes to train the model on.
     
        :returns: An object of :class: ResNet34
        """

        super().__init__()

        # Input: (N, 64, 56, 56) -> Output: (N, 64, 56, 56)
        self.resnet_layer_1 = ResNetBlockIdentityV1(in_channels=64, num_blocks=3, reduce=False)
        # Input: (N, 64, 56, 56) -> Output: (N, 128, 28, 28)
        self.resnet_layer_2 = ResNetBlockIdentityV1(in_channels=64, num_blocks=4, reduce=True)
        # Input: (N, 128, 28, 28) -> Output: (N, 256, 14, 14)
        self.resnet_layer_3 = ResNetBlockIdentityV1(in_channels=128, num_blocks=6, reduce=True)
        # Input: (N, 256, 14, 14) -> Output: (N, 512, 7, 7)
        self.resnet_layer_4 = ResNetBlockIdentityV1(in_channels=256, num_blocks=3, reduce=True)

        self.average_pooling_layer = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.classifier = torch.nn.Sequential(torch.nn.Flatten(),
                                              torch.nn.Linear(in_features=512, out_features=num_classes))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the ResNet34.

        :param X: Input tensor

        :retuns: Result of forward propagation through ResNet34
        """

        X = self.conv_layer_1(X)
        X = self.resnet_layer_1(X)
        X = self.resnet_layer_2(X)
        X = self.resnet_layer_3(X)
        X = self.resnet_layer_4(X)
        X = self.average_pooling_layer(X)
        
        return self.classifier(X)


class ResNet50(ResNetBase):
    """
    Implementation of ResNet 54 layer image classification model described in the paper "Deep Residual Learning for Image Recognition"
    """

    def __init__(self, num_classes: int) -> None:
        """
        Method to instantiate  object of :class: ResNet50

        :param num_classes: Number of classes to train the model on.
     
        :returns: An object of :class: ResNet50
        """

        super().__init__()

        # Input: (N, 64, 56, 56) -> Output: (N, 64, 56, 56)
        self.resnet_layer_1 = ResNetBlockPartialProjectionV2(in_channels=64, continuity_channels=64, num_blocks=3, reduce=False)
        # Input: (N, 64, 56, 56) -> Output: (N, 128, 28, 28)
        self.resnet_layer_2 = ResNetBlockPartialProjectionV2(in_channels=64 * 4, continuity_channels=128, num_blocks=4, reduce=True)
        # Input: (N, 128, 28, 28) -> Output: (N, 256, 14, 14)
        self.resnet_layer_3 = ResNetBlockPartialProjectionV2(in_channels=128 * 4, continuity_channels=256, num_blocks=6, reduce=True)
        # Input: (N, 256, 14, 14) -> Output: (N, 512, 7, 7)
        self.resnet_layer_4 = ResNetBlockPartialProjectionV2(in_channels=256 * 4, continuity_channels=512, num_blocks=3, reduce=True)

        self.average_pooling_layer = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.classifier = torch.nn.Sequential(torch.nn.Flatten(),
                                              torch.nn.Linear(in_features=2048, out_features=num_classes))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the ResNet50.

        :param X: Input tensor

        :retuns: Result of forward propagation through ResNet50
        """

        X = self.conv_layer_1(X)
        X = self.resnet_layer_1(X)
        X = self.resnet_layer_2(X)
        X = self.resnet_layer_3(X)
        X = self.resnet_layer_4(X)
        X = self.average_pooling_layer(X)
        
        return self.classifier(X)


class ResNet101(ResNetBase):
    """
    Implementation of ResNet 101 layer image classification model described in the paper "Deep Residual Learning for Image Recognition"
    """

    def __init__(self, num_classes: int) -> None:
        """
        Method to instantiate  object of :class: ResNet101

        :param num_classes: Number of classes to train the model on.
     
        :returns: An object of :class: ResNet101
        """

        super().__init__()

        # Input: (N, 64, 56, 56) -> Output: (N, 64, 56, 56)
        self.resnet_layer_1 = ResNetBlockPartialProjectionV2(in_channels=64, continuity_channels=64, num_blocks=3, reduce=False)
        # Input: (N, 64, 56, 56) -> Output: (N, 128, 28, 28)
        self.resnet_layer_2 = ResNetBlockPartialProjectionV2(in_channels=64 * 4, continuity_channels=128, num_blocks=4, reduce=True)
        # Input: (N, 128, 28, 28) -> Output: (N, 256, 14, 14)
        self.resnet_layer_3 = ResNetBlockPartialProjectionV2(in_channels=128 * 4, continuity_channels=256, num_blocks=23, reduce=True)
        # Input: (N, 256, 14, 14) -> Output: (N, 512, 7, 7)
        self.resnet_layer_4 = ResNetBlockPartialProjectionV2(in_channels=256 * 4, continuity_channels=512, num_blocks=3, reduce=True)

        self.average_pooling_layer = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.classifier = torch.nn.Sequential(torch.nn.Flatten(),
                                              torch.nn.Linear(in_features=2048, out_features=num_classes))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the ResNet101.

        :param X: Input tensor

        :retuns: Result of forward propagation through ResNet101
        """

        X = self.conv_layer_1(X)
        X = self.resnet_layer_1(X)
        X = self.resnet_layer_2(X)
        X = self.resnet_layer_3(X)
        X = self.resnet_layer_4(X)
        X = self.average_pooling_layer(X)
        
        return self.classifier(X)


class ResNet152(ResNetBase):
    """
    Implementation of ResNet 152 layer image classification model described in the paper "Deep Residual Learning for Image Recognition"
    """

    def __init__(self, num_classes: int) -> None:
        """
        Method to instantiate  object of :class: ResNet152

        :param num_classes: Number of classes to train the model on.
     
        :returns: An object of :class: ResNet152
        """

        super().__init__()

        # Input: (N, 64, 56, 56) -> Output: (N, 64, 56, 56)
        self.resnet_layer_1 = ResNetBlockPartialProjectionV2(in_channels=64, continuity_channels=64, num_blocks=3, reduce=False)
        # Input: (N, 64, 56, 56) -> Output: (N, 128, 28, 28)
        self.resnet_layer_2 = ResNetBlockPartialProjectionV2(in_channels=64 * 4, continuity_channels=128, num_blocks=8, reduce=True)
        # Input: (N, 128, 28, 28) -> Output: (N, 256, 14, 14)
        self.resnet_layer_3 = ResNetBlockPartialProjectionV2(in_channels=128 * 4, continuity_channels=256, num_blocks=36, reduce=True)
        # Input: (N, 256, 14, 14) -> Output: (N, 512, 7, 7)
        self.resnet_layer_4 = ResNetBlockPartialProjectionV2(in_channels=256 * 4, continuity_channels=512, num_blocks=3, reduce=True)

        self.average_pooling_layer = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.classifier = torch.nn.Sequential(torch.nn.Flatten(),
                                              torch.nn.Linear(in_features=2048, out_features=num_classes))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate input tensor through the ResNet152.

        :param X: Input tensor

        :retuns: Result of forward propagation through ResNet152
        """

        X = self.conv_layer_1(X)
        X = self.resnet_layer_1(X)
        X = self.resnet_layer_2(X)
        X = self.resnet_layer_3(X)
        X = self.resnet_layer_4(X)
        X = self.average_pooling_layer(X)
        
        return self.classifier(X)
