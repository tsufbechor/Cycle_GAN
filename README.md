# Cycle_GAN

In this exercise I used two MNIST datasets. Each digit is colored in a different color in  each dataset. The aim of the model is to learn how to take the image of the digit in a specific  color and translate it into a different color.

Model Architecture:
=

The two discriminators are built with 5 Convolutional layers:
3X64>64X128>128X256>256X512>512>1024. The kernel size is 3X3.
I used LeakyRelu activation function and average pooling.
At the end there is a linear layer 1024X1.
Finally a sigmoid function is used on the output.
The two generators are built with 8 Convolutional layers:
3X64>64X128>128X256>256X128>128X64>64X32>32X16>16X3. The kernel size is 3X3
I use stride=2,padding=1 in each layer and between each layer the ReLu activation function. 
Finally tanh is used on the output.
I used the MSE and L1 losses and ran the model for 10 epochs and the batch size is 64

Training Process:
=

In each iteration we sample a real photo and a fake generated photo. We create the label 1 
for the real photo and the label 0 for generated fake photo. We train the discriminator of 
the first dataset, then train the generator from the second dataset on the first dataset, then 
the discriminator of the second dataset and finally the generator of the first dataset on the 
second dataset. 
The discriminator is trained by getting an input from the first dataset and a generated fake 
photo. Then the loss is calculated for these two photos. Then the weights are updated 
accordingly.
Let’s take a look at how effective the Generators are at different stages of the training 
process:
5 epochs:
=

![image](https://user-images.githubusercontent.com/81694762/235474563-2637608e-e259-4b12-97da-b72d2cc5b524.png)

![image](https://user-images.githubusercontent.com/81694762/235474711-85c98e81-5102-4e71-a930-a553d7ef2a4a.png)


