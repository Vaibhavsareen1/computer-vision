import torch


class LeNet(torch.nn.Module):
    """
    Class that implements Yann LeCunn's first model called `LeNet 1` introduced in the paper 
    "Backpropagation Applied to Handwritten Zip Code Recognition"

    Link to paper:
    https://galileo-unbound.blog/wp-content/uploads/2025/02/lecun.neco_.1989.1.4.541.pdf

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/lenet_1.ipynb`
    """

    def __init__(self) -> None:
        """
        Method to instantiate object of :class: LeNet

        :returns: Instance of :class: LeNet
        """

        super().__init__()

        self.tanh = torch.nn.Tanh()

        self.h1_convolution_filter = torch.nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, padding=2, stride=2, dilation=1, bias=False)
        self.h1_bias = torch.nn.Parameter(data=torch.randn(size=(1, 12, 8, 8), dtype=torch.float32))

        # Create 12 convolution layers which get stacked together where each takes in 8 channels and produces the same output shape
        self.h2_convolution_filter_1 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, dilation=1, bias=False)
        self.h2_convolution_filter_2 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, dilation=1, bias=False)
        self.h2_convolution_filter_3 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, dilation=1, bias=False)
        self.h2_convolution_filter_4 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, dilation=1, bias=False)
        self.h2_convolution_filter_5 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, dilation=1, bias=False)
        self.h2_convolution_filter_6 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, dilation=1, bias=False)
        self.h2_convolution_filter_7 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, dilation=1, bias=False)
        self.h2_convolution_filter_8 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, dilation=1, bias=False)
        self.h2_convolution_filter_9 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, dilation=1, bias=False)
        self.h2_convolution_filter_10 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, dilation=1, bias=False)
        self.h2_convolution_filter_11 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, dilation=1, bias=False)
        self.h2_convolution_filter_12 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, dilation=1, bias=False)
        self.h2_bias = torch.nn.Parameter(data=torch.randn(size=(1, 12, 4, 4), dtype=torch.float32))
    
        self.h3_layer = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Linear(in_features=192, out_features=30, bias=True),
                            torch.nn.Tanh(),
                            torch.nn.Linear(in_features=30, out_features=10, bias=True))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through LeNet's neural network layers

        :param X: input tensor

        :returns: Output tensor containing 10 values representing the ten classes for which the model is being trained for
        """

        # Propagate input tensor through the first H1 layer and add bias
        h1_result = self.h1_convolution_filter(X)
        h1_result = h1_result + self.h1_bias
        h1_result = self.tanh(h1_result)

        # Extract 12 sets of 8 receptive fields (the combination for the 8 channels is not disclosed in the original paper)
        # For understanding we are using a technique of group of 8 channels for each feature map inspired by LeNet5
        h2_convolution_filter_1_input = h1_result[:, [0, 1, 2, 3, 4, 5, 6, 7], :, :]
        h2_convolution_filter_2_input = h1_result[:, [1, 2, 3, 4, 5, 6, 7, 8], :, :]
        h2_convolution_filter_3_input = h1_result[:, [2, 3, 4, 5, 6, 7, 8, 9], :, :]
        h2_convolution_filter_4_input = h1_result[:, [3, 4, 5, 6, 7, 8, 9, 10], :, :]
        h2_convolution_filter_5_input = h1_result[:, [4, 5, 6, 7, 8, 9, 10, 11], :, :]
        h2_convolution_filter_6_input = h1_result[:, [5, 6, 7, 8, 9, 10, 11, 0], :, :]
        h2_convolution_filter_7_input = h1_result[:, [6, 7, 8, 9, 10, 11, 0, 1], :, :]
        h2_convolution_filter_8_input = h1_result[:, [7, 8, 9, 10, 11, 0, 1, 2], :, :]
        h2_convolution_filter_9_input = h1_result[:, [8, 9, 10, 11, 0, 1, 2, 3], :, :]
        h2_convolution_filter_10_input = h1_result[:, [9, 10, 11, 0, 1, 2, 3, 4], :, :]
        h2_convolution_filter_11_input = h1_result[:, [10, 11, 0, 1, 2, 3, 4, 5], :, :]
        h2_convolution_filter_12_input = h1_result[:, [11, 0, 1, 2, 3, 4, 5, 6], :, :]

        # Propapagate all of the h2 inputs to get outputs from the filters
        h2_convolution_filter_1_output = self.h2_convolution_filter_1(h2_convolution_filter_1_input)
        h2_convolution_filter_2_output = self.h2_convolution_filter_2(h2_convolution_filter_2_input)
        h2_convolution_filter_3_output = self.h2_convolution_filter_3(h2_convolution_filter_3_input)
        h2_convolution_filter_4_output = self.h2_convolution_filter_4(h2_convolution_filter_4_input)
        h2_convolution_filter_5_output = self.h2_convolution_filter_5(h2_convolution_filter_5_input)
        h2_convolution_filter_6_output = self.h2_convolution_filter_6(h2_convolution_filter_6_input)
        h2_convolution_filter_7_output = self.h2_convolution_filter_7(h2_convolution_filter_7_input)
        h2_convolution_filter_8_output = self.h2_convolution_filter_8(h2_convolution_filter_8_input)
        h2_convolution_filter_9_output = self.h2_convolution_filter_9(h2_convolution_filter_9_input)
        h2_convolution_filter_10_output = self.h2_convolution_filter_10(h2_convolution_filter_10_input)
        h2_convolution_filter_11_output = self.h2_convolution_filter_11(h2_convolution_filter_11_input)
        h2_convolution_filter_12_output = self.h2_convolution_filter_12(h2_convolution_filter_12_input)

        # Horizontally stack the outputs 
        h2_result = torch.cat(tensors=[h2_convolution_filter_1_output,
                                       h2_convolution_filter_2_output,
                                       h2_convolution_filter_3_output,
                                       h2_convolution_filter_4_output,
                                       h2_convolution_filter_5_output,
                                       h2_convolution_filter_6_output,
                                       h2_convolution_filter_7_output,
                                       h2_convolution_filter_8_output,
                                       h2_convolution_filter_9_output,
                                       h2_convolution_filter_10_output,
                                       h2_convolution_filter_11_output,
                                       h2_convolution_filter_12_output],
                            dim=1)
        h2_result = h2_result + self.h2_bias
        h2_result = self.tanh(h2_result)

        # Propagate the output of h2 layer through h3 layer and return it back
        return self.h3_layer(h2_result)