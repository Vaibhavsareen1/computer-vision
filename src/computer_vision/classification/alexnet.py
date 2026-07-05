import torch


class AlexNet(torch.nn.Module):
    """
    Class that implements an image classification model called `AlexNet` introduced in the paper 
    "ImageNet Classification with Deep Convolutional Neural Networks"

    Link to paper:
    https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/alexnet.ipynb`
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """
        Method to instantiate object of :class: AlexNet

        :param: Number of classes to train the model on

        :returns: Instance of :class: AlexNet
        """

        super().__init__()

        self.classes = num_classes
        self.relu = torch.nn.ReLU()

        # First two convolution, pooling and local response normalization block spread across two gpus
        # (N, 3, 224, 224) -> (N, 128, 13, 13) per layer
        self.gpu_1_convolution_layer_1_2 = torch.nn.Sequential(
                                                torch.nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2, dilation=1, bias=True),
                                                torch.nn.ReLU(),
                                                torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                                                torch.nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, padding=0),
                                            
                                                torch.nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2, stride=1, dilation=1, bias=True),
                                                torch.nn.ReLU(),
                                                torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                                                torch.nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, padding=0))

        self.gpu_2_convolution_layer_1_2 = torch.nn.Sequential(
                                                torch.nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2, dilation=1, bias=True),
                                                torch.nn.ReLU(),
                                                torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                                                torch.nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, padding=0),

                                                torch.nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2, stride=1, dilation=1, bias=True),
                                                torch.nn.ReLU(),
                                                torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                                                torch.nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, padding=0))

        # (N, 128, 13, 13) -> (N, 192, 13, 13) per layer
        self.gpu_1_convolution_layer_3_1 = torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, padding=1, stride=1, dilation=1, bias=True)
        self.gpu_1_convolution_layer_3_2 = torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, padding=1, stride=1, dilation=1, bias=True)
        self.gpu_2_convolution_layer_3_1 = torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, padding=1, stride=1, dilation=1, bias=True)
        self.gpu_2_convolution_layer_3_2 = torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, padding=1, stride=1, dilation=1, bias=True)

        # (N, 192, 13, 13) -> (N, 128, 13, 13) per layer
        self.gpu_1_convolution_layer_4_5 = torch.nn.Sequential(
                                                torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
                                                torch.nn.ReLU(),

                                                torch.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
                                                torch.nn.ReLU(),

                                                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1))
    
        self.gpu_2_convolution_layer_4_5 = torch.nn.Sequential(
                                                torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
                                                torch.nn.ReLU(),

                                                torch.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
                                                torch.nn.ReLU(),

                                                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1))

        self.gpu_1_fully_connected_layer_1 = torch.nn.Sequential(
                                                    torch.nn.Flatten(),
                                                    torch.nn.Dropout1d(p=0.5),
                                                    torch.nn.Linear(in_features=9216, out_features=2048, bias=True),
                                                    torch.nn.ReLU())

        self.gpu_2_fully_connected_layer_1 = torch.nn.Sequential(
                                                    torch.nn.Flatten(),
                                                    torch.nn.Linear(in_features=9216, out_features=2048, bias=True),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Dropout1d(p=0.5))

        self.gpu_1_fully_connected_layer_2 = torch.nn.Sequential(
                                                    torch.nn.Dropout1d(p=0.5),
                                                    torch.nn.Linear(in_features=4096, out_features=2048, bias=True),
                                                    torch.nn.ReLU())
        self.gpu_2_fully_connected_layer_2 = torch.nn.Sequential(
                                                    torch.nn.Dropout1d(p=0.5),
                                                    torch.nn.Linear(in_features=4096, out_features=2048, bias=True),
                                                    torch.nn.ReLU())

        self.fully_connected_layer_3 = torch.nn.Linear(in_features=4096, out_features=self.classes, bias=True)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through AlexNet's neural network layers

        :param X: input tensor

        :returns: Output tensor containing `num_classes` number of values
        """

        # Splitting data over 2 GPUs
        # Propagate input through the first two convolution layers spread across two GPUs
        gpu_1_output = self.gpu_1_convolution_layer_1_2(X)
        gpu_2_output = self.gpu_2_convolution_layer_1_2(X)

        # Both outputs of convolution layer 1 and 2 is considred by convolution layer 3 so combine the output from each gpu layer
        # into one to propagate through convolution layer 3
        convolution_layer_3_input = gpu_1_output + gpu_2_output
        
        gpu_1_output_1 = self.gpu_1_convolution_layer_3_1(convolution_layer_3_input)
        gpu_1_output_2 = self.gpu_1_convolution_layer_3_2(convolution_layer_3_input)
        gpu_2_output_1 = self.gpu_2_convolution_layer_3_1(convolution_layer_3_input)
        gpu_2_output_2 = self.gpu_2_convolution_layer_3_2(convolution_layer_3_input)

        gpu_1_output_1 = self.relu(gpu_1_output_1)
        gpu_1_output_2 = self.relu(gpu_1_output_2)
        gpu_2_output_1 = self.relu(gpu_2_output_1)
        gpu_2_output_2 = self.relu(gpu_2_output_2)

        gpu_1_output = torch.cat((gpu_1_output_1, gpu_2_output_2), dim=1)
        gpu_2_output = torch.cat((gpu_2_output_1, gpu_1_output_2), dim=1)

        gpu_1_output = self.gpu_1_convolution_layer_4_5(gpu_1_output)
        gpu_2_output = self.gpu_2_convolution_layer_4_5(gpu_2_output)  
          
        # Concatenate the feature maps obtained across the 5th convolution layer on the channel's dimension
        # 
        concatenated_output = torch.cat((gpu_1_output, gpu_2_output), dim=1)

        # Propagate the convolution output through the fully connected layers
        # after each layer propagated combine the outputs into one
        gpu_1_output = self.gpu_1_fully_connected_layer_1(concatenated_output)
        gpu_2_output = self.gpu_2_fully_connected_layer_1(concatenated_output)

        concatenated_output = torch.cat((gpu_1_output, gpu_2_output), dim=1)

        gpu_1_output = self.gpu_1_fully_connected_layer_2(concatenated_output)
        gpu_2_output = self.gpu_2_fully_connected_layer_2(concatenated_output)

        concatenated_output = torch.cat((gpu_1_output, gpu_2_output), dim=1)

        output = self.fully_connected_layer_3(concatenated_output)

        return output
    

class AlexNetV2(torch.nn.Module):
    """
    Class that implements AlexNet's version two described in the paper 
    "One weird trick for parallelizing convolutional neural networks"

    This model is the one implemented in pytorch which is more efficient version according to Alex Krizhevsky

    Link to paper:
    https://arxiv.org/pdf/1404.5997

    Note: 
    There is a discripancy between the config file accompanying the paper and the paper itself.
    In the paper the total number of kernel maps is 64, 192, 384, 384 and 256 across each layer where as in the config file
    it is 64, 192, 384, 256 and 256. Pytorch prefers the config file over the paper and hence here the config file's configuration
    is considered the source of truth

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/alexnet.ipynb`
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """
        Method to instantiate object of :class: AlexNetV2

        :param: Number of classes to train the model on. Defaults to 1000 classes

        :returns: Instance of :class: AlexNetV2
        """

        super().__init__()

        self.feature = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2, dilation=1, bias=True),
                                           torch.nn.ReLU(),
                                           torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                                           torch.nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, padding=0),
                                        
                                           torch.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2, stride=1, dilation=1, bias=True),
                                           torch.nn.ReLU(),
                                           torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                                           torch.nn.MaxPool2d(kernel_size=3, stride=2, dilation=1, padding=0),

                                           torch.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
                                           torch.nn.ReLU(),

                                           torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
                                           torch.nn.ReLU(),

                                           torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, dilation=1, bias=True),
                                           torch.nn.ReLU(),
                                           torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1))

        self.classifier = torch.nn.Sequential(torch.nn.Flatten(),
                                              torch.nn.Dropout1d(p=0.5),
                                              torch.nn.Linear(in_features=9216, out_features=4096, bias=True),
                                              torch.nn.ReLU(),
                                              torch.nn.Dropout1d(p=0.5),
                                              torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(in_features=4096, out_features=num_classes, bias=True))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through AlexNetV2's neural network layers

        :param X: input tensor

        :returns: Output tensor containing `num_classes` number of values
        """

        X = self.feature(X)

        return self.classifier(X)