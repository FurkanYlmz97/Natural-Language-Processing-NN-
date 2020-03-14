__author__ = "Furkan Yilmaz"
import h5py as h5
import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_of_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def question_3_2_1(d, p, epoch):

    # Load files
    file = h5.File("assign2_data2.h5", "r")
    traind = np.array(file["traind"]) - 1  # 372500x1 values -> 0-249
    trainx = np.array(file["trainx"]) - 1  # 372500x3 values -> 0-249
    vald = np.array(file["vald"]) - 1  # 46500x1 values -> 0-249
    valx = np.array(file["valx"]) - 1  # 46500x3 values -> 0-249
    classes = 250

    # initialize variables
    who = np.random.normal(0, 0.01, ((250, p + 1)))
    weh = np.random.normal(0, 0.01, (p, d + 1))
    word_embed = np.random.normal(0, 0.1, (250, d))
    gradient_who = np.zeros((250, p + 1))
    gradient_weh = np.zeros((p, d + 1))
    gradient_word_embed = np.zeros((250, d))
    momentum_weh = 0
    momentum_word = 0
    momentum_who = 0
    desired = np.zeros(250)
    o = np.zeros(250)
    ce_cum = []
    val_ce_cum = []
    all_weights = []

    # Need to find the best valdiation error weights and stop the training
    stop = False
    last_weights = []
    last_error = 9999999

    # epoch starts maximum of 50
    for e in range(epoch):
        ce = []
        val_ce = []

        # If the validaiton does not start to increase continue
        if not stop:
            for i in range(372500):

                # take the input by one hot coding and the desired output
                inputs = np.eye(classes)[trainx[i]]
                desired = np.eye(classes)[traind[i]]

                # use the embedded matrix to find an input that will be given to hidden layer
                embedded_input = np.matmul(inputs, word_embed)
                mean_embedded_input = ((embedded_input[0] + embedded_input[1] + embedded_input[2]) / 3)
                mean_embedded_input = np.append(mean_embedded_input, 1)
                # mean_embedded_input = mean_embedded_input - np.mean(mean_embedded_input)
                # mean_embedded_input = mean_embedded_input / np.std(mean_embedded_input)

                # Hidden layer equations
                hidden_v = np.matmul(mean_embedded_input, weh.transpose())
                f_hidden = np.diag(derivative_of_sigmoid(hidden_v))
                hidden_o = sigmoid(hidden_v)
                hidden_o = np.append(hidden_o, 1)

                # Output layer equtions
                pre_o = np.matmul(hidden_o, who.transpose())
                o = np.exp(pre_o) / np.sum(np.exp(pre_o))

                # find error
                e = -1 * np.dot(desired, np.log(o))
                ce.append(e)

                # gradient of the output layer weights
                delta_o = o - desired
                gradient_who = gradient_who + np.outer(delta_o, hidden_o)

                # gradient of the hidden layer weights
                delta_h = np.matmul(np.dot(delta_o.transpose(), who[0:, 0:p]), f_hidden)
                gradient_weh = gradient_weh + np.outer(delta_h, mean_embedded_input.transpose())

                # gradient of the embedding matrix weights
                Ay = (inputs[0] + inputs[2] + inputs[1]) / 3
                gradient_word_embed = gradient_word_embed + np.matmul(np.outer(Ay, delta_h.transpose()), weh[0:, 0:d])

                # if 200 samples have been collected update weights
                if (i + 1) % 200 == 0:
                    # who = who - 0.15 * gradient_who / 200 - 0.85 * momentum_who
                    # weh = weh - 0.15 * gradient_weh / 200 - 0.85 * momentum_weh
                    # word_embed = word_embed - 0.15 * gradient_word_embed / 200 - 0.85 * momentum_word
                    #
                    # momentum_weh = 0.15 * gradient_weh / 200
                    # momentum_word = 0.15 * momentum_word / 200
                    # momentum_who = 0.15 * gradient_who / 200

                    who = who - 0.15 * (gradient_who / 200 + 0.85 * momentum_who)
                    weh = weh - 0.15 * (gradient_weh / 200 + 0.85 * momentum_weh)
                    word_embed = word_embed - 0.15 * (gradient_word_embed / 200 + 0.85 * momentum_word)

                    momentum_weh = gradient_weh / 200 + 0.85 * momentum_weh
                    momentum_word = gradient_word_embed / 200 + 0.85 * momentum_word
                    momentum_who = gradient_who / 200 + 0.85 * momentum_who

                    gradient_who = np.zeros((250, p + 1))
                    gradient_weh = np.zeros((p, d + 1))
                    gradient_word_embed = np.zeros((250, d))

            # calculate the mean epoch error
            ce_cum.append(np.mean(ce))

            # use the trained network in validation set
            for i in range(46500):

                # take input
                inputs = np.eye(classes)[valx[i]]
                desired = np.eye(classes)[vald[i]]

                # find the outpur and error
                embedded_input = np.matmul(inputs, word_embed)
                mean_embedded_input = ((embedded_input[0] + embedded_input[1] + embedded_input[2]) / 3)
                mean_embedded_input = np.append(mean_embedded_input, 1)
                # mean_embedded_input = mean_embedded_input - np.mean(mean_embedded_input)
                # mean_embedded_input = mean_embedded_input / np.std(mean_embedded_input)
                hidden_v = np.matmul(mean_embedded_input, weh.transpose())
                hidden_o = sigmoid(hidden_v)
                hidden_o = np.append(hidden_o, 1)
                pre_o = np.matmul(hidden_o, who.transpose())
                o = np.exp(pre_o) / np.sum(np.exp(pre_o))
                e = -1 * np.dot(desired, np.log(o))
                val_ce.append(e)

            # find the epoch error
            epoch_error = np.mean(val_ce)
            val_ce_cum.append(epoch_error)
            # if epoch error starts to increase stop training and return last weights
            # if epoch_error > last_error:
            #     stop = True
            # else:
            #     last_weights = []
            #     last_weights.append((weh, word_embed, who))
            #     last_error = epoch_error

            # find lowest validation error's weights
            if epoch_error <= min(val_ce_cum):
                last_weights = []
                last_weights.append((weh, word_embed, who))

    plt.figure()
    plt.plot(ce_cum, label="Training CE")
    plt.plot(val_ce_cum, label="Validation CE")
    plt.title("Cross Entropy Error")
    plt.xlabel("Epoch Number")
    plt.ylabel("Cross Entropy Error")
    plt.legend()

    return last_weights


if __name__ == '__main__':

    # Load Files
    file = h5.File("assign2_data2.h5", "r")
    testd = np.array(file["testd"]) - 1  # 46500x1 values -> 0-249
    testx = np.array(file["testx"]) - 1  # 46500x3 values -> 0-249
    words = np.array(file["words"])  # 250

    # Start training with 3 different dimension and neuron numbers
    #############################################

    # get the weights of the trained NN
    a = question_3_2_1(8, 64, 50)
    weh, word_embed, who = a[0][0], a[0][1], a[0][2]
    classes = 250
    te = []
    acc = []

    # select three test examples to show the 4th word
    rand = np.random.randint(0, 46499)

    for i in range(46500):
        # take the input
        inputs = np.eye(classes)[testx[i]]
        desired = np.eye(classes)[testd[i]]

        # find the output and the Test error
        embedded_input = np.matmul(inputs, word_embed)
        mean_embedded_input = ((embedded_input[0] + embedded_input[1] + embedded_input[2]) / 3)
        mean_embedded_input = np.append(mean_embedded_input, 1)
        hidden_v = np.matmul(mean_embedded_input, weh.transpose())
        hidden_o = sigmoid(hidden_v)
        hidden_o = np.append(hidden_o, 1)
        pre_o = np.matmul(hidden_o, who.transpose())
        o = np.exp(pre_o) / np.sum(np.exp(pre_o))
        e = -1 * np.dot(desired, np.log(o))
        te.append(e)

        # Show the guessed 4th word of 3 triagrams
        if i == rand or i == rand + 1 or i == rand + 2:
            print("Sentence: ", end="")
            print(words[testx[i][0]].decode('UTF-8'), end=" ")
            print(words[testx[i][1]].decode('UTF-8'), end=" ")
            print(words[testx[i][2]].decode('UTF-8'), end=" (")
            print(words[testd[i]].decode('UTF-8'), end=")")
            print()
            np.sort(o)
            for m in range(10):
                if m == 0:
                    if words[testd[i]].decode('UTF-8') == words[np.argmax(o)].decode('UTF-8'):
                        acc.append(1)
                    else:
                        acc.append(0)
                print(words[np.argmax(o)].decode('UTF-8'), end=" -> with probability ")
                print(np.around(np.max(o) * 100, decimals=1))
                o[np.argmax(o)] = 0

    # print test errors
    print("Test error for (D,P) = (8, 64): ", end="")
    print(np.mean(te))
    print("Accuracy for (D,P) = (8, 64): %", end="")
    print(np.mean(acc) * 100)
    print()
    print()
    #############################################

    #############################################

    # get the weights of the trained NN
    a = question_3_2_1(16, 128, 50)
    weh, word_embed, who = a[0][0], a[0][1], a[0][2]
    classes = 250
    te = []
    acc = []
    rand = np.random.randint(0, 46499)

    for i in range(46500):

        # take the input
        inputs = np.eye(classes)[testx[i]]
        desired = np.eye(classes)[testd[i]]

        # find the output and the Test error
        embedded_input = np.matmul(inputs, word_embed)
        mean_embedded_input = ((embedded_input[0] + embedded_input[1] + embedded_input[2]) / 3)
        mean_embedded_input = np.append(mean_embedded_input, 1)
        hidden_v = np.matmul(mean_embedded_input, weh.transpose())
        hidden_o = sigmoid(hidden_v)
        hidden_o = np.append(hidden_o, 1)
        pre_o = np.matmul(hidden_o, who.transpose())
        o = np.exp(pre_o) / np.sum(np.exp(pre_o))
        e = -1 * np.dot(desired, np.log(o))
        te.append(e)

        # Show the guessed 4th word of 3 triagrams
        if i == rand or i == rand + 1 or i == rand + 2:
            print("Sentence: ", end="")
            print(words[testx[i][0]].decode('UTF-8'), end=" ")
            print(words[testx[i][1]].decode('UTF-8'), end=" ")
            print(words[testx[i][2]].decode('UTF-8'), end=" (")
            print(words[testd[i]].decode('UTF-8'), end=")")
            print()
            np.sort(o)
            for m in range(10):
                if m == 0:
                    if words[testd[i]].decode('UTF-8') == words[np.argmax(o)].decode('UTF-8'):
                        acc.append(1)
                    else:
                        acc.append(0)
                print(words[np.argmax(o)].decode('UTF-8'), end=" -> with probability ")
                print(np.around(np.max(o) * 100, decimals=1))
                o[np.argmax(o)] = 0

    # print test errors
    print("Test error for (D,P) = (16, 128): ", end="")
    print(np.mean(te))
    print("Accuracy for (D,P) = (16, 128): %", end="")
    print(np.mean(acc) * 100)
    print()
    print()
    #############################################

    #############################################

    # get the weights of the trained NN
    a = question_3_2_1(32, 256, 50)
    weh, word_embed, who = a[0][0], a[0][1], a[0][2]
    classes = 250
    te = []
    acc = []
    rand = np.random.randint(0, 46499)

    for i in range(46500):

        # take the input
        inputs = np.eye(classes)[testx[i]]
        desired = np.eye(classes)[testd[i]]

        # find the output and the Test error
        embedded_input = np.matmul(inputs, word_embed)
        mean_embedded_input = ((embedded_input[0] + embedded_input[1] + embedded_input[2]) / 3)
        mean_embedded_input = np.append(mean_embedded_input, 1)
        hidden_v = np.matmul(mean_embedded_input, weh.transpose())
        hidden_o = sigmoid(hidden_v)
        hidden_o = np.append(hidden_o, 1)
        pre_o = np.matmul(hidden_o, who.transpose())
        o = np.exp(pre_o) / np.sum(np.exp(pre_o))
        e = -1 * np.dot(desired, np.log(o))
        te.append(e)

        # Show the guessed 4th word of 3 triagrams
        if i == rand or i == rand + 1 or i == rand + 2:
            print("Sentence: ", end="")
            print(words[testx[i][0]].decode('UTF-8'), end=" ")
            print(words[testx[i][1]].decode('UTF-8'), end=" ")
            print(words[testx[i][2]].decode('UTF-8'), end="( ")
            print(words[testd[i]].decode('UTF-8'), end=")")
            print()
            np.sort(o)
            for m in range(10):
                if m == 0:
                    if words[testd[i]].decode('UTF-8') == words[np.argmax(o)].decode('UTF-8'):
                        acc.append(1)
                    else:
                        acc.append(0)
                print(words[np.argmax(o)].decode('UTF-8'), end=" -> with probability ")
                print(np.around(np.max(o) * 100, decimals=1))
                o[np.argmax(o)] = 0

    # print test errors
    print("Test error for (D,P) = (32, 256): ", end="")
    print(np.mean(te))
    print("Accuracy for (D,P) = (32, 256): %", end="")
    print(np.mean(acc) * 100)
    print()
    print()
    #############################################
    plt.show()
