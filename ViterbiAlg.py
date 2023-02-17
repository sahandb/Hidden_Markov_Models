import numpy as np
import pandas as pd


# pre Process
def readData(Path):
    with open(Path, encoding='unicode_escape') as file:
        lines = file.readlines()

        allLines = np.array(lines)
        chunkedData = ''.join(allLines)

        # return list of sentence
        sentenceList = chunkedData.split('###/###')

        LIST_DATA = []
        for index_sentences in range(len(sentenceList)):
            sentences = sentenceList[index_sentences]

            list_words = sentences.split('\n')

            for word in list_words:
                if word is None:
                    continue
                if len(word.split('/')) == 2:
                    LIST_DATA.append({'word': word.split('/')[0],
                                      'classLabel': word.split('/')[1],
                                      'sentenceNumber': index_sentences})

        df = pd.DataFrame(LIST_DATA)
        return df


def getPi(df):
    list_tags = df.loc[:, 'classLabel'].unique()
    list_words = df.loc[:, 'word'].unique()
    len_sentences = df['sentenceNumber'].max()

    listOfPi = []
    for classLabel in list_tags:
        len_current_tag = len(df.loc[df['classLabel'] == classLabel])
        pi = (len_current_tag + 1) / (len_sentences + len(list_tags))
        listOfPi.append(pi)

    return listOfPi, list_tags, list_words, len_sentences


def getA(lst, lsw, df):
    matrixA = np.zeros(shape=(len(lst), len(lst)))

    for tagIdx in range(len(lst)):
        for idxTagNext in range(len(lst)):
            tagIdxs = lst[tagIdx]
            nextTagIdx = lst[idxTagNext]
            sequenceTagCounter = 0
            for idxs in range(len(lst)):
                if lst[idxs] is tagIdxs:
                    if idxs < len(lst) - 1:
                        if lst[idxs + 1] is nextTagIdx:
                            sequenceTagCounter += 1

            iCounter = len(df.loc[df['classLabel'] == tagIdxs])
            matrixA[tagIdx, idxTagNext] = (sequenceTagCounter + 1) / (iCounter + len(lsw))
    return matrixA


def getB(lst, lsw, df):
    matrixB = np.zeros(shape=(len(lst), len(lsw)))

    for jTagIdx in range(len(lst)):
        for idxK in range(len(lsw)):
            word = lsw[idxK]
            classLabel = lst[jTagIdx]

            obCounter = len(df.loc[(df['word'] == word) & (df['classLabel'] == classLabel)])
            jCounter = len(df.loc[df['classLabel'] == classLabel])

            matrixB[jTagIdx, idxK] = (obCounter + 1) / (jCounter + len(lsw))
    return matrixB


def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]

    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]

    return alpha


def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))

    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])

    return beta


def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(V)

    for n in range(n_iter):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return (a, b)


def viterbi(V, a, b, initial_distribution):
    T = V.shape[0]
    M = a.shape[0]

    omega = np.zeros((T, M))
    omega[0, :] = np.log(initial_distribution * b[:, V[0]])

    prev = np.zeros((T - 1, M))

    for t in range(1, T):
        for j in range(M):
            # Same as Forward Probability
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])

            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)

            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)

    # Path Array
    S = np.zeros(T)

    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])

    S[0] = last_state

    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    # Flip the path array since we were backtracking
    S = np.flip(S, axis=0)

    # Convert numeric values to actual hidden states
    result = []
    for s in S:
        if s == 0:
            result.append("A")
        else:
            result.append("B")

    return result


cztrain = readData('Dataset/cztrain.txt')
cztest = readData('Dataset/cztest.txt')

# Probabilities for the initial distribution
piii, listClassLabel, listWord, len_Sen = getPi(cztrain)

# Transition Probabilities
matA = getA(listClassLabel, listWord, cztrain)

# Emission Probabilities
matB = getB(listClassLabel, listWord, cztrain)

a, b = baum_welch(cztest, matA, matB, piii, n_iter=100)

print(viterbi(cztest, a, b, piii))
