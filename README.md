# Natural-Language-Processing-NN-
The NN in this repository predicts the fourth word in sequence given the preceding tri- gram.

The following archituecture has been used to design this NN. 
![arch](https://user-images.githubusercontent.com/48417171/76659278-08273900-6587-11ea-9356-f6ce35a56a07.png)

The input layer has 3 neurons corresponding to the trigram entries. An embedding matrix R (250xD) is used to linearly map each single word onto a vector representation of length D. The same embedding matrix is used for each input word in the trigram, without considering the sequence order. The hidden layer uses a sigmoidal activation function on each of P hidden-layer neurons. The output layer predicts a separate response zi for each of 250 vocabulary words, and the probability of each word is estimated via a soft-max operation.

For this part, I have trained a neural network with one output, one hidden and a word embedding matrix. I have initialized all the weights including the word embedding matrix with 0 mean and 0.01 standard deviation. Furthermore, I have used 200 mini-batch for training and used a 0.15 learning rate with a 0.85 momentum rate. For training, I have not exceeded 50 epochs.
At the end of each epoch, I have found the cross-entropy error for the validation set and the training set, meanwhile, I have also recorded the weights. I have plotted both error curves and decided to select the proper weights by looking at the validation error curve. The weights that gives the lowest validation error has been selected for the final weights because if the validation error curve starts to increase whereas the training error curve is still decreasing, I understood that the training error decreases because the training starts to overfit and taking these weights as final will be not proper in terms of generalization. By using the neural network that gives the lowest validation error, I have then tested the performance by finding cross-entropy error on the test set and also by looking whether my network gives reasonable words as an output for the fourth word of to the input 3 words. Also, the reason why I didn’t stop the training when I realized that the validation error increasing is I just wanted to plot the whole learning curves.
For training rule, I have used batch gradient descent rule where the gradients of the neural network have found as follows,

![1](https://user-images.githubusercontent.com/48417171/76690384-7d127580-6650-11ea-9a60-6f28383848c1.png)
![2](https://user-images.githubusercontent.com/48417171/76690390-8865a100-6650-11ea-9b0c-db6741189c15.png)


The learning curves for the Neural Network with 64 neurons in the hidden layer and 8 dimensions for the word embedding matrix is as follows,
![3](https://user-images.githubusercontent.com/48417171/76690391-8b609180-6650-11ea-8570-c025ef95eb6e.png)

By looking at this graph, we can see that our Neural Network learns the set while both error curves decrease in each epoch. However, after a point the errors start to decrease slower, even the validation error decreases slower than the training error. After some more training epochs, we can say that these errors will not decrease and will converge. For this training, the lowest validation error is seen in the last epoch. Therefore, I have taken the last network's weights and used it to show the performance on the test set. Also, with these parameters, the lowest training error is 3.2 and the validation error is 3.3. Both errors are close to each other whereas this is correlated with the model’s complexity. The complexity comes from the dimensions of the embedded matrix and the neuron number in the hidden layer. An increase in one of these will result in a more complex model which will make our network enable us to learn new features from the data set that will finally result in a lower training error with a more obvious overfit at the final.


The learning curves for the Neural Network with 128 neurons in the hidden layer and 16 dimensions for the word embedding matrix is as follows,
![4](https://user-images.githubusercontent.com/48417171/76690393-8bf92800-6650-11ea-9c8d-620a191e4a25.png)

This time we can see that our validation error almost converges (almost parallel) where the training error keeps decreasing. As I mentioned before increasing the dimensions of the word embedding matrix and the number of the neurons in the hidden layer has resulted in a more complex network. This network is able to learn more from the data set which we can see that the training error is 2.86 and the validation error is 3.0. A second result of the increasing complexity is that while the validation error is almost converging the training error is keep decreasing which results in a 0.2 difference between validation and training errors at the 50th epoch. That means this network started to overfit after a point where the validation error does not decrease significantly but the training error gets smaller which means that the generalization of this network will decrease. However, still the complexity of this network is not high that after a point the validation error does not start to increase. For the testing performance of this set. The model that gives the lowest validation error will be chosen to show the performance on the test set.


The learning curves for the Neural Network with 256 neurons in the hidden layer and 32 dimensions for the word embedding matrix is as follows,
![5](https://user-images.githubusercontent.com/48417171/76690394-8c91be80-6650-11ea-8bb2-5fc2b9a81ba4.png)

This is the third training model and this the most complex one. Therefore, this model is able to learn more complex patterns from the data set. This result can be seen in both validation and the training error. Our network is able to learn complex patterns in the words which result in smaller errors in both validation and training sets. The final error in the validation set is 2.9 where the final error in training is 2.74. Same with the previous network after a point the validation error does not decrease significantly whereas the training error keeps decreasing. For this model network was able to learn even more about the training set but this learning has not result with a significant boost in validation set performance. This is an overfit example where our network does not increase its performance in terms of generalization but gets better and better at the training set. Also, the complexity is not enough again to make the validation set’s error increase. For this scenario again the network with the smallest validation error will be taken to show the performance on the test set.
Overall, the biggest dimension and neuron numbered network, i.e. the most complex network was able to decrease the training and the validation errors most due to its complexity. We can say that increasing the number of representations of a word by increasing the embedded matrix’s dimension increases the performance as well the increasing neuron number also increases the performance while these two also increases the learning speed slightly.

**The Result**

For every 3 network 3 input trigrams have been selected where their 4th word has been shown in the parentheses, then the network’s first 10 output have been listed below. Lastly, the accuracy of the networks has been calculated whether the 4th word has been guessed correctly. The result has been shown in the table.

![6](https://user-images.githubusercontent.com/48417171/76690395-8d2a5500-6650-11ea-9e9a-77b8e355604e.png)

