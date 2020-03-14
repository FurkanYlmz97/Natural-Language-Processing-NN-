# Natural-Language-Processing-NN-
The NN in this repository predicts the fourth word in sequence given the preceding tri- gram.

The following archituecture has been used to design this NN. 
![arch](https://user-images.githubusercontent.com/48417171/76659278-08273900-6587-11ea-9356-f6ce35a56a07.png)

The input layer has 3 neurons corresponding to the trigram entries. An embedding matrix R (250xD) is used to linearly map each single word onto a vector representation of length D. The same embedding matrix is used for each input word in the trigram, without considering the sequence order. The hidden layer uses a sigmoidal activation function on each of P hidden-layer neurons. The output layer predicts a separate response zi for each of 250 vocabulary words, and the probability of each word is estimated via a soft-max operation.
