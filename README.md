# Cycle_GAN
In this exercise I used two MNIST datasets. Each digit is colored in a different color in  each dataset. The aim of the model is to learn how to take the image of the digit in a specific  color and translate it into a different color.
=

Model Architecture:

The two discriminators are built with 5 Convolutional layers:
3X64>64X128>128X256>256X512>512>1024. The kernel size is 3X3.
Between the layers we use the LeakyRelu activation function and average pooling.
At the end there is a linear layer 1024X1.
Finally a sigmoid function is used on the output.
The two generators are built with 8 Convolutional layers:
3X64>64X128>128X256>256X128>128X64>64X32>32X16>16X3. The kernel size is 3X3
I use stride=2,padding=1 in each layer and between each layer the ReLu activation function. 
Finally tanh is used on the output.
I used the MSE and L1 losses and ran the model for 10 epochs and the batch size is 64
