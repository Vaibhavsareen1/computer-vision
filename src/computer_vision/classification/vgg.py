import torch


class VGGA(torch.nn.Module):
    """
    Class that implements VGG's 11 layer model introduced in the paper
    "Very Deep Convolutional Networks For Large Scale Image Recognition" 

    Link to paper:
    https://arxiv.org/pdf/1409.1556

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/vgg.ipynb`
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """
        Method to instantiate object of :class: VGGA

        :param: Number of classes to train the model on. Defaults to 1000 classes

        :returns: Instance of :class: VGGA
        """

        super().__init__()
        
        self.classes = num_classes

        self.conv_block_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_3 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_4 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_5 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))

        self.classifier = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=4096, out_features=self.classes, bias=True)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through VGGA's neural network layers

        :param X: input tensor

        :returns: Output tensor containing `num_classes` number of values
        """

        X = self.conv_block_1(X)
        X = self.conv_block_2(X)
        X = self.conv_block_3(X)
        X = self.conv_block_4(X)
        X = self.conv_block_5(X)

        return self.classifier(X)


class VGGALRN(torch.nn.Module):
    """
    Class that implements VGG's 11 layer model with Local Response Norm (LRN) introduced in the paper
    "Very Deep Convolutional Networks For Large Scale Image Recognition" 

    Link to paper:
    https://arxiv.org/pdf/1409.1556

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/vgg.ipynb`
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """
        Method to instantiate object of :class: VGGALRN

        :param: Number of classes to train the model on. Defaults to 1000 classes

        :returns: Instance of :class: VGGALRN
        """

        super().__init__()
        
        self.classes = num_classes

        self.conv_block_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_3 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_4 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_5 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))

        self.classifier = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=4096, out_features=self.classes, bias=True)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through VGGALRN's neural network layers

        :param X: input tensor

        :returns: Output tensor containing `num_classes` number of values
        """

        X = self.conv_block_1(X)
        X = self.conv_block_2(X)
        X = self.conv_block_3(X)
        X = self.conv_block_4(X)
        X = self.conv_block_5(X)

        return self.classifier(X)


class VGGB(torch.nn.Module):
    """
    Class that implements VGG's 13 layer model introduced in the paper
    "Very Deep Convolutional Networks For Large Scale Image Recognition" 

    Link to paper:
    https://arxiv.org/pdf/1409.1556

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/vgg.ipynb`
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """
        Method to instantiate object of :class: VGGB

        :param: Number of classes to train the model on. Defaults to 1000 classes

        :returns: Instance of :class: VGGB
        """

        super().__init__()
        
        self.classes = num_classes

        self.conv_block_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_3 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_4 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_5 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))

        self.classifier = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=4096, out_features=self.classes, bias=True)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through VGGB's neural network layers

        :param X: input tensor

        :returns: Output tensor containing `num_classes` number of values
        """

        X = self.conv_block_1(X)
        X = self.conv_block_2(X)
        X = self.conv_block_3(X)
        X = self.conv_block_4(X)
        X = self.conv_block_5(X)

        return self.classifier(X)


class VGGC(torch.nn.Module):
    """
    Class that implements VGG's 16 layer model with 1x1 convolution layer introduced in convolution block 3, 4, and 5 introduced in the paper
    "Very Deep Convolutional Networks For Large Scale Image Recognition" 

    Link to paper:
    https://arxiv.org/pdf/1409.1556

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/vgg.ipynb`
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """
        Method to instantiate object of :class: VGGC

        :param: Number of classes to train the model on. Defaults to 1000 classes

        :returns: Instance of :class: VGGC
        """

        super().__init__()
        
        self.classes = num_classes

        self.conv_block_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_3 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_4 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_5 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))

        self.classifier = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=4096, out_features=self.classes, bias=True)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through VGGC's neural network layers

        :param X: input tensor

        :returns: Output tensor containing `num_classes` number of values
        """

        X = self.conv_block_1(X)
        X = self.conv_block_2(X)
        X = self.conv_block_3(X)
        X = self.conv_block_4(X)
        X = self.conv_block_5(X)

        return self.classifier(X)


class VGGD(torch.nn.Module):
    """
    Class that implements VGG's 16 layer model with 3x3 convolution layer introduced in convolution block 3, 4, and 5 introduced in the paper
    "Very Deep Convolutional Networks For Large Scale Image Recognition" 

    Link to paper:
    https://arxiv.org/pdf/1409.1556

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/vgg.ipynb`
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """
        Method to instantiate object of :class: VGGD

        :param: Number of classes to train the model on. Defaults to 1000 classes

        :returns: Instance of :class: VGGD
        """

        super().__init__()
        
        self.classes = num_classes

        self.conv_block_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_3 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_4 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_5 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))

        self.classifier = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=4096, out_features=self.classes, bias=True)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through VGGD's neural network layers

        :param X: input tensor

        :returns: Output tensor containing `num_classes` number of values
        """

        X = self.conv_block_1(X)
        X = self.conv_block_2(X)
        X = self.conv_block_3(X)
        X = self.conv_block_4(X)
        X = self.conv_block_5(X)

        return self.classifier(X)
    

class VGGE(torch.nn.Module):
    """
    Class that implements VGG's 19 layer model with 3x3 convolution layer introduced in convolution block 3, 4, and 5 introduced in the paper
    "Very Deep Convolutional Networks For Large Scale Image Recognition" 

    Link to paper:
    https://arxiv.org/pdf/1409.1556

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/vgg.ipynb`
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """
        Method to instantiate object of :class: VGGE

        :param: Number of classes to train the model on. Defaults to 1000 classes

        :returns: Instance of :class: VGGE
        """

        super().__init__()
        
        self.classes = num_classes

        self.conv_block_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_3 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_4 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_5 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.ReLU(), 
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))

        self.classifier = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=4096, out_features=self.classes, bias=True)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through VGGE's neural network layers

        :param X: input tensor

        :returns: Output tensor containing `num_classes` number of values
        """

        X = self.conv_block_1(X)
        X = self.conv_block_2(X)
        X = self.conv_block_3(X)
        X = self.conv_block_4(X)
        X = self.conv_block_5(X)

        return self.classifier(X)


# Batch Normalized Classes
class VGGABN(torch.nn.Module):
    """
    Class that implements VGG's 11 layer model introduced in the paper
    "Very Deep Convolutional Networks For Large Scale Image Recognition" 

    This class comes with Batch Normalization Applied to each convolution layer

    Link to paper:
    https://arxiv.org/pdf/1409.1556

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/vgg.ipynb`
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """
        Method to instantiate object of :class: VGGA

        :param: Number of classes to train the model on. Defaults to 1000 classes

        :returns: Instance of :class: VGGA
        """

        super().__init__()
        
        self.classes = num_classes

        self.conv_block_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=64),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=128),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_3 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=256),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=256),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_4 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_5 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))

        self.classifier = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=4096, out_features=self.classes, bias=True)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through VGGA's neural network layers

        :param X: input tensor

        :returns: Output tensor containing `num_classes` number of values
        """

        X = self.conv_block_1(X)
        X = self.conv_block_2(X)
        X = self.conv_block_3(X)
        X = self.conv_block_4(X)
        X = self.conv_block_5(X)

        return self.classifier(X)


class VGGALRNBN(torch.nn.Module):
    """
    Class that implements VGG's 11 layer model with Local Response Norm (LRN) introduced in the paper
    "Very Deep Convolutional Networks For Large Scale Image Recognition" 

    This class comes with Batch Normalization Applied to each convolution layer

    Link to paper:
    https://arxiv.org/pdf/1409.1556

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/vgg.ipynb`
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """
        Method to instantiate object of :class: VGGALRN

        :param: Number of classes to train the model on. Defaults to 1000 classes

        :returns: Instance of :class: VGGALRN
        """

        super().__init__()
        
        self.classes = num_classes

        self.conv_block_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=64),
                                torch.nn.ReLU(),
                                torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=128),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_3 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=128),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=256),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_4 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_5 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))

        self.classifier = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=4096, out_features=self.classes, bias=True)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through VGGALRN's neural network layers

        :param X: input tensor

        :returns: Output tensor containing `num_classes` number of values
        """

        X = self.conv_block_1(X)
        X = self.conv_block_2(X)
        X = self.conv_block_3(X)
        X = self.conv_block_4(X)
        X = self.conv_block_5(X)

        return self.classifier(X)


class VGGBBN(torch.nn.Module):
    """
    Class that implements VGG's 13 layer model introduced in the paper
    "Very Deep Convolutional Networks For Large Scale Image Recognition" 
    
    This class comes with Batch Normalization Applied to each convolution layer

    Link to paper:
    https://arxiv.org/pdf/1409.1556

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/vgg.ipynb`
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """
        Method to instantiate object of :class: VGGB

        :param: Number of classes to train the model on. Defaults to 1000 classes

        :returns: Instance of :class: VGGB
        """

        super().__init__()
        
        self.classes = num_classes

        self.conv_block_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=64),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=64),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=128),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=128),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_3 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=256),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=256),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_4 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_5 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))

        self.classifier = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=4096, out_features=self.classes, bias=True)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through VGGB's neural network layers

        :param X: input tensor

        :returns: Output tensor containing `num_classes` number of values
        """

        X = self.conv_block_1(X)
        X = self.conv_block_2(X)
        X = self.conv_block_3(X)
        X = self.conv_block_4(X)
        X = self.conv_block_5(X)

        return self.classifier(X)


class VGGCBN(torch.nn.Module):
    """
    Class that implements VGG's 16 layer model with 1x1 convolution layer introduced in convolution block 3, 4, and 5 introduced in the paper
    "Very Deep Convolutional Networks For Large Scale Image Recognition" 

    This class comes with Batch Normalization Applied to each convolution layer

    Link to paper:
    https://arxiv.org/pdf/1409.1556

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/vgg.ipynb`
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """
        Method to instantiate object of :class: VGGC

        :param: Number of classes to train the model on. Defaults to 1000 classes

        :returns: Instance of :class: VGGC
        """

        super().__init__()
        
        self.classes = num_classes

        self.conv_block_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=64),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=64),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=128),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=128),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_3 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=256),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=256),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=256),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_4 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_5 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))

        self.classifier = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=4096, out_features=self.classes, bias=True)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through VGGC's neural network layers

        :param X: input tensor

        :returns: Output tensor containing `num_classes` number of values
        """

        X = self.conv_block_1(X)
        X = self.conv_block_2(X)
        X = self.conv_block_3(X)
        X = self.conv_block_4(X)
        X = self.conv_block_5(X)

        return self.classifier(X)


class VGGDBN(torch.nn.Module):
    """
    Class that implements VGG's 16 layer model with 3x3 convolution layer introduced in convolution block 3, 4, and 5 introduced in the paper
    "Very Deep Convolutional Networks For Large Scale Image Recognition" 

    This class comes with Batch Normalization Applied to each convolution layer

    Link to paper:
    https://arxiv.org/pdf/1409.1556

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/vgg.ipynb`
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """
        Method to instantiate object of :class: VGGD

        :param: Number of classes to train the model on. Defaults to 1000 classes

        :returns: Instance of :class: VGGD
        """

        super().__init__()
        
        self.classes = num_classes

        self.conv_block_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_feature=64),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_feature=64),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_feature=128),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_feature=128),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_3 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_feature=256),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_feature=256),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_feature=256),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_4 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_feature=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_feature=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_feature=512),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_5 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_feature=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_feature=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_feature=512),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))

        self.classifier = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=4096, out_features=self.classes, bias=True)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through VGGD's neural network layers

        :param X: input tensor

        :returns: Output tensor containing `num_classes` number of values
        """

        X = self.conv_block_1(X)
        X = self.conv_block_2(X)
        X = self.conv_block_3(X)
        X = self.conv_block_4(X)
        X = self.conv_block_5(X)

        return self.classifier(X)
    

class VGGEBN(torch.nn.Module):
    """
    Class that implements VGG's 19 layer model with 3x3 convolution layer introduced in convolution block 3, 4, and 5 introduced in the paper
    "Very Deep Convolutional Networks For Large Scale Image Recognition" 

    This class comes with Batch Normalization Applied to each convolution layer    

    Link to paper:
    https://arxiv.org/pdf/1409.1556

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/vgg.ipynb`
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """
        Method to instantiate object of :class: VGGE

        :param: Number of classes to train the model on. Defaults to 1000 classes

        :returns: Instance of :class: VGGE
        """

        super().__init__()
        
        self.classes = num_classes

        self.conv_block_1 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=64),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=64),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_2 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=128),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=128),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_3 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=256),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=256),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=256),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=256),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_4 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))
        
        self.conv_block_5 = torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                                torch.nn.BatchNorm2d(num_features=512),
                                torch.nn.ReLU(), 
                                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1))

        self.classifier = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Dropout1d(p=0.5),
                            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
                            torch.nn.ReLU(),
                            torch.nn.Linear(in_features=4096, out_features=self.classes, bias=True)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through VGGE's neural network layers

        :param X: input tensor

        :returns: Output tensor containing `num_classes` number of values
        """

        X = self.conv_block_1(X)
        X = self.conv_block_2(X)
        X = self.conv_block_3(X)
        X = self.conv_block_4(X)
        X = self.conv_block_5(X)

        return self.classifier(X)