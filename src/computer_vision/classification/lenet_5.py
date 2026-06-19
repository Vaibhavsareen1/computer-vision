import torch


class LeNet5(torch.nn.Module):
    """
    Class that implements Yann LeCunn's fifth iteration of a classification model called `LeNet 5` introduced in the paper 
    "Gradient Based Learning Applied to Document Recognition"

    Link to paper:
    http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

    For detail analysis of the paper have a look at the notebook stored at
    `computer-vision/notebooks/classification/lenet_5.ipynb`
    """

    def __init__(self) -> None:
        """
        Method to instantiate object of :class: LeNet5

        :returns: Instance of :class: LeNet5
        """

        super().__init__()

        self.tanh = torch.nn.Tanh()

        # The paper denotes convolution layers as Cx, sub sampling layers as Sx and fully connected layers as Fx.
        self.C1_layer = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=0, stride=1, dilation=1, bias=True)
        # Pooling (sub sampling) layers in pytorch don't use weights and biases, but in case of LeNet 5 for each feature map a weight and bias needs to be 
        # multiplied  and added respectively with the sum of the sub samping operation to do so we have to set AvgPool2d's `divisor_overide` to 1
        self.S2_layer = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0, divisor_override=1)
        self.S2_weight = torch.nn.Parameter(data=torch.randn(size=(1, 6, 1, 1), dtype=torch.float32))
        self.S2_bias = torch.nn.Parameter(data=torch.randn(size=(1, 6, 1, 1), dtype=torch.float32))
        self.tanh = torch.nn.Tanh()

        # Create 16 convolution layers which get stacked together where first 6 feature maps use 3 channels followed by
        # 9 feature maps each takes in 4 channels and the last feature map uses all input channels produces the same output shape
        self.C3_convolution_filter_1 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_2 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_3 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_4 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_5 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_6 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_7 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_8 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_9 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_10 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_11 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_12 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_13 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_14 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_15 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)
        self.C3_convolution_filter_16 = torch.nn.Conv2d(in_channels=6, out_channels=1, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)

        # Similar to S2 create S4 sampling layer, weights and biases
        self.S4_layer = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0, divisor_override=1)
        self.S4_weight = torch.nn.Parameter(data=torch.randn(size=(1, 16, 1, 1), dtype=torch.float32))
        self.S4_bias = torch.nn.Parameter(data=torch.randn(size=(1, 16, 1, 1), dtype=torch.float32))
    
        self.C5_layer = torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0, dilation=1, bias=True)

        self.F6_layer = torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Linear(in_features=120, out_features=84, bias=True),
                            torch.nn.Tanh(),
                            torch.nn.Linear(in_features=84, out_features=10, bias=True)
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to propagate incoming tensor through LeNet-5's neural network layers

        :param X: input tensor

        :returns: Output tensor containing 10 values representing the ten classes for which the model is being trained for
        """

        c1_output = self.C1_layer(X)
        s2_output = self.S2_layer(c1_output)
        s2_output = s2_output * self.S2_weight + self.S2_bias
        s2_output = self.tanh(s2_output)

        # Extract 16 sets of 8 receptive fields (the combination for the 8 channels is not disclosed in the original paper)
        # For understanding we are using a technique of group of 8 channels for each feature map inspired by LeNet5
        c3_convolution_filter_1_input = s2_output[:, [0, 1, 2], :, :]
        c3_convolution_filter_2_input = s2_output[:, [1, 2, 3], :, :]
        c3_convolution_filter_3_input = s2_output[:, [2, 3, 4], :, :]
        c3_convolution_filter_4_input = s2_output[:, [3, 4, 5], :, :]
        c3_convolution_filter_5_input = s2_output[:, [0, 4, 5], :, :]
        c3_convolution_filter_6_input = s2_output[:, [0, 1, 5], :, :]
        c3_convolution_filter_7_input = s2_output[:, [0, 1, 2, 3], :, :]
        c3_convolution_filter_8_input = s2_output[:, [1, 2, 3, 4], :, :]
        c3_convolution_filter_9_input = s2_output[:, [2, 3, 4, 5], :, :]
        c3_convolution_filter_10_input = s2_output[:, [0, 3, 4, 5], :, :]
        c3_convolution_filter_11_input = s2_output[:, [0, 1, 4, 5], :, :]
        c3_convolution_filter_12_input = s2_output[:, [0, 1, 2, 5], :, :]
        c3_convolution_filter_13_input = s2_output[:, [0, 1, 3, 4], :, :]
        c3_convolution_filter_14_input = s2_output[:, [1, 2, 4, 5], :, :]
        c3_convolution_filter_15_input = s2_output[:, [0, 2, 3, 5], :, :]
        c3_convolution_filter_16_input = s2_output

        # Propapagate all of the c3 inputs to get outputs from the filters
        c3_convolution_filter_1_output = self.C3_convolution_filter_1(c3_convolution_filter_1_input)
        c3_convolution_filter_2_output = self.C3_convolution_filter_2(c3_convolution_filter_2_input)
        c3_convolution_filter_3_output = self.C3_convolution_filter_3(c3_convolution_filter_3_input)
        c3_convolution_filter_4_output = self.C3_convolution_filter_4(c3_convolution_filter_4_input)
        c3_convolution_filter_5_output = self.C3_convolution_filter_5(c3_convolution_filter_5_input)
        c3_convolution_filter_6_output = self.C3_convolution_filter_6(c3_convolution_filter_6_input)
        c3_convolution_filter_7_output = self.C3_convolution_filter_7(c3_convolution_filter_7_input)
        c3_convolution_filter_8_output = self.C3_convolution_filter_8(c3_convolution_filter_8_input)
        c3_convolution_filter_9_output = self.C3_convolution_filter_9(c3_convolution_filter_9_input)
        c3_convolution_filter_10_output = self.C3_convolution_filter_10(c3_convolution_filter_10_input)
        c3_convolution_filter_11_output = self.C3_convolution_filter_11(c3_convolution_filter_11_input)
        c3_convolution_filter_12_output = self.C3_convolution_filter_12(c3_convolution_filter_12_input)
        c3_convolution_filter_13_output = self.C3_convolution_filter_13(c3_convolution_filter_13_input)
        c3_convolution_filter_14_output = self.C3_convolution_filter_14(c3_convolution_filter_14_input)
        c3_convolution_filter_15_output = self.C3_convolution_filter_15(c3_convolution_filter_15_input)
        c3_convolution_filter_16_output = self.C3_convolution_filter_16(c3_convolution_filter_16_input)

        # Horizontally stack the outputs 
        c3_output = torch.cat(tensors=[c3_convolution_filter_1_output,
                                       c3_convolution_filter_2_output,
                                       c3_convolution_filter_3_output,
                                       c3_convolution_filter_4_output,
                                       c3_convolution_filter_5_output,
                                       c3_convolution_filter_6_output,
                                       c3_convolution_filter_7_output,
                                       c3_convolution_filter_8_output,
                                       c3_convolution_filter_9_output,
                                       c3_convolution_filter_10_output,
                                       c3_convolution_filter_11_output,
                                       c3_convolution_filter_12_output,
                                       c3_convolution_filter_13_output,
                                       c3_convolution_filter_14_output,
                                       c3_convolution_filter_15_output,
                                       c3_convolution_filter_16_output],
                                dim=1)

        s4_output = self.S4_layer(c3_output)
        s4_output = s4_output * self.S4_weight + self.S4_bias
        s4_output = self.tanh(s4_output)

        c5_output = self.C5_layer(s4_output)
        c5_output = self.tanh(c5_output)

        f6_output = self.F6_layer(c5_output)


        return f6_output
