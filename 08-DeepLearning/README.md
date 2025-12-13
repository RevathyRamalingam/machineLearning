# Neural Networks in AI

Neural Network in AI is a term inspired by the biological network of neurons in the human brain. It works just like how neurons in
the human brain receive inputs, process, and transmit information. Artificial neurons/nodes in AI learn from data and perform tasks 
such as pattern recognition, natural language processing, image classification, etc.

## Types of Neural Networks

1. **Feed Forward NN**  
   The simplest type of NN where information flows in one direction from input nodes through hidden layers to output nodes.
   
2. **Recurrent NN**  
   Used to process sequential data and is ideal for time-series prediction. These networks have a feedback loop to retain information from previous inputs, enabling them to process sequential data.

3. **Convolutional NN (CNN)**  
   Designed with convolutional layers used for image classification and image recognition.

4. **Generative Adversarial Network (GAN)**  
   Uses a generator to generate new data and a discriminator to identify whether the image is real or fake.

5. **Transformer Neural Network**  
   For example, ChatGPT uses a Generative Pretrained Transformer that understands and produces human-like text.

## Popular Deep Learning Frameworks

- **Keras**: A high-level framework built on top of TensorFlow.
- **PyTorch**: A lower-level framework that is more flexible and programmers can map pytorch to C, while Keras can be compared to C++.

## Tensor Basics

A tensor is a mathematical term used to represent a numerical value or scalar. Tensors can have multiple dimensions:

- **1D Tensor**: A scalar, any numerical value.
- **2D Tensor**: A matrix with rows and columns.
- **3D Tensor**: A matrix with 3 dimensions (e.g., a cuboid with length, breadth, and height).
- **4D Tensor**: A batch of color images (e.g., [batch size, r, g, b]).
- **5D Tensor**: A stack of videos.
- **6D, 7D**: Higher-dimensional tensors are used in specialized tasks.

## CNN Architecture

CNN consists of two primary layers:

1. **Convolutional Layer**  
2. **Dense (Fully Connected) Layer**

- **Convolutional Filters**: Filters (3) slide over the image and generate feature maps (1, 2, 3) for each filter.
- **Pooling**: Downsamples images to reduce their size (e.g., from 200x200 to 100x100).
  
### Activation Functions

- **Softmax**: Generalization of the sigmoid function for multiple classes.
- **ReLU**: Used in hidden layers to introduce non-linearity.

### Static Vs Dynamic Computational Graph

- Pytorch creates a dynamic graph when forward pass is initiated (model is created by passing input images), and it offers the flexibility to debug the layers at every stage of convolutions. This makes Pytorch the most preferred one for image classification. It is useful for prototyping, research activities and debugging.
- TensorFlow(Keras) uses static computation graph which must be created during training step using @tf.function annotation. As the graph is preloaded, it is faster, memory efficient and faster to deploy in TensorFlowLite. Keras is ideal for production environments and largescale deployments
  

## Steps to Build a CNN in Keras

1. **Create a Model**  
   - **Sequential API**: For linear architecture.
   - **Functional API**: For more complex models with multiple inputs/outputs.
   - **Custom Model**: By subclassing the `Model` class.

2. **Compile**: Specify the loss function, optimizer, and evaluation metrics.

3. **Fit**: Train the model  
   ```python
   history = model.fit(train_ds, epochs=10, validation_data=val_ds)
4. Fine-tune: Adjust the learning rate for optimal validation accuracy.

5. Add Dense Layer: Include a dense layer before the output layer and fine-tune the number of neurons.

6.  Initialization: Each neuron is initialized using algorithms like Random Initialization, Xavier Initialization, He Initialization, etc.

7.  Use Dropout to prevent overfitting. Example:
inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
drop = keras.layers.Dropout(droprate)(inner)
outputs = keras.layers.Dense(10)(drop)
8. Data Augmentation:
Use ImageDataGenerator for rotation, flipping, zooming, etc.

9. Save the Model:
Create a ModelCheckpoint object to save the best model.

Steps to Build a CNN in PyTorch

1. Create Custom Dataset:
Subclass the Dataset class and override the __getitem__() method to load images and labels.

2. Transformations:
Use transforms.Compose() to apply sequential transformations like resizing, cropping, and normalization.

3. DataLoader:
Create a DataLoader instance to handle batching and shuffling of data.

4. Model Creation:
   
4.1 Subclass the nn.Module class:
create a model from base class nn.module. Implement init method to initialize parameters for convolution layers, FC layer and 
activation.

Implement forward method (forward pass) as the images pass through the convolutional layers.
When module(x) is invoked it calls forward method from the model and applies the convolutions specified there.

The output channels can be computed using the below formula
output channels = ( W − K + 2 P )/ S + 1
eg image of size 200*200 height=200, width=200 default stride=1, padding =0, kernel size=(3*3)=size of the filter ,output channel = ((200 -3+(2*0)/1)+1 = 198

4.2 create the model and apply  it to device

4.3 define optimizer and loss function appropriate for the model.Use the torch.optim.SGD optimizer to update model parameters.
optimizer uses stochastic gradient descend (SGD)algorithm with momentum to optimize model parameters. It creates an optimizer that updates the model parameters for optimum performance.

5 Train the model for several epochs
An epoch is one complete cycle of training of all images in trainset. If batch size is 10 and there are 1000 images in 
training dataset then it completes the epoch in 100 iterations (1000/10 = 100).

	5.1 fetch image and label from trainloader
	5.2 unsqueeze labels to include batchsize as images have batchsize
	5.3 optimizer.zero_grad() is used to clear the gradients to 0 inorder to do backward propagation
 BACKPROPAGATION  - is a key concept in neural network learning. After making forward pass and making weight and feature computation, it predicts and
		finds out the loss(error). This error is corrected by updating the weights in backpropagation.
	5.4 optimizer.step() updates the model parameters

Forward Pass:

First, the input data is passed through the neural network to make a prediction. This is known as the forward pass.
The network applies a series of transformations (weighted sums and activation functions) to the input data to compute an output.

Compute Loss:

Once the prediction is made, we compare it to the actual target value (the ground truth) and calculate the loss (or error). This is typically done using a loss function (like Mean Squared Error for regression tasks or Cross-Entropy for classification tasks).

Backward Pass:

Backpropagation comes into play here. The loss needs to be "propagated backward" through the network to update the weights.
The algorithm uses gradient descent to adjust the weights. Essentially, it computes the gradient of the loss with respect to each weight in the network. The gradient tells us how much each weight 
contributed to the error.


While processing data in training Loop , we unsqueeze the input images to add batch_size dimension to the input data.
Tensor Operations:
unsqueeze(input, dim) adds a new dimension to the tensor. This operation is used to add batch_size dimension to the input images.
Example:

y = x.unsqueeze(x, 0)

Insights from DeepLearning Module:
* Keras vs. PyTorch:
 explored the differences between Keras and PyTorch. While Keras provides a higher-level, user-friendly API for rapid prototyping, PyTorch offers more flexibility and control, making it suitable for research and complex models.
* Training Images with Convolutional Neural Networks (CNNs):
 learnt how CNNs work to process images by applying filters (kernels) that slide over the image to extract patterns, forming a feature map. This allows the model to capture important features like edges, textures, and shapes.
* Impact of Stride, Kernel Size, and Padding on Feature Output:
The stride, kernel (filter) size, and padding significantly affect the output dimensions of the feature map. Stride controls how much the filter moves, kernel size affects the receptive field, and padding ensures the spatial dimensions of the image are preserved.
* ReLU Activation for Feature Maps:
 discovered how the ReLU activation function is applied to feature maps, introducing non-linearity and helping the network learn complex patterns.
* Backpropagation: It is the foundation of neural networks, where gradients of the loss are calculated, and the model's weights are adjusted accordingly to minimize the error.
* Loss Functions: 
learnt about various loss functions such as Mean Squared Error (MSE), Cross Entropy, and Binary Cross-Entropy (BCE). These functions measure how well the model’s predictions align with the true values.
* Softmax Function:
The Softmax function is used to convert raw scores (logits) into probabilities, making it essential for multi-class classification tasks.
* Optimizers - Stochastic Gradient Descent (SGD):
 gained an understanding of SGD and how it helps in optimizing the model parameters by adjusting the weights in the direction that minimizes the loss.
* Torchsummary:
explored torchsummary, which provides a convenient way to visualize the architecture of a CNN model, including the number of parameters and output shapes at each layer.
* Pretrained Models vs. Building model from Scratch
* Gradient: The gradient tells you how much to adjust each model parameter to reduce the loss, making it a key concept in the learning process.
* Unsqueeze: used to add dimensions to the input tensor. For example, converting a 2D tensor into a 3D tensor by adding a batch size dimension.


