import numpy
import csv

def matrix_factorization(P, Q, R, f, m, BU, BI, iters = 50, g = 0.015, l0 = 0.001, l1 = 0.001, bias = False, filename = 'ua.txt'):
    """Do one iteration of stochastic gradient descent

    Keyword arguments:
    P -- the user's matrix
    Q -- the item's matrix
    R -- the rating matrix
    f -- the number of features
    m -- the overall average rating
    BU -- the user's bias vector
    BI -- the item's bias vector
    g -- the learning rate
    l -- the regularization term
    bias -- the bias option
    filename -- the name of the log file
    """
    Q = Q.T
    e_step = 0
    e_prev = 1000000
    output = open(filename, "w+")

    for it in xrange(iters):
        for u in xrange(len(R)):
            for i in xrange(len(R[u])):
                if R[u][i] > 0:
                    if bias:
                        # update the error
                        e_ui = R[u][i] - prediction(P[u,:], Q[:,i], m, BU[u], BI[i])
                    else:
                        e_ui = R[u][i] - numpy.dot(P[u,:], Q[:,i])
                
                    if bias:
                        # update the biases
                        BU[u] += g * e_ui - l0 * BU[u]
                        BI[i] += g * e_ui - l0 * BI[i]
                
                    # update the feature vectors
                    for k in xrange(f):
                        P[u][k] += g * (e_ui * Q[k][i] - l1 * P[u][k])
                        Q[k][i] += g * (e_ui * P[u][k] - l1 * Q[k][i])

        if bias:
            e_step = cost(P, Q, R, f, m, BU, BI, l0, l1, bias)
        else:
            e_step = cost(P, Q, R, f, m, BU, BI, l0, l1, bias)

        improvement = e_prev - e_step
        output.write(str(it) + " " + str(improvement) + "\n")
        e_prev = e_step
        if (e_step < 0.001) | (improvement < 0):
            break
        print("step " + str(it+1))
    output.write("steps: " + str(iters) + "\n")
    output.write("learning rate: " + str(g) + "\n")

    output.close()
    return

def cost(P, Q, R, f, m, BU, BI, l0 = 0.001, l1 = 0.001, bias = False):
    """Return the regularized square error
    
    Keyword arguments:
    P -- the users' matrix
    Q -- the items' matrix
    R -- the rating matrix
    f -- the number of features
    m -- the overall average rating
    BU -- the user's bias vector
    BI -- the item's bias vector
    l -- the regularization term
    bias -- the bias option
    """
    error = 0

    for u in xrange(len(R)):
        for i in xrange(len(R[u])):
            if R[u][i] > 0:
                if bias:
                    error += pow(R[u][i] - prediction(P[u,:], Q[:,i], m, BU[u], BI[i]), 2)
                else:
                    error += pow(R[u][i] - numpy.dot(P[u,:], Q[:,i]), 2)
                for k in xrange(f):
                    error += l1 * (pow(P[u][k], 2) + pow(Q[k][i], 2))

    if bias:
        error += l0 * (numpy.sum(pow(BU, 2)) + numpy.sum(pow(BI, 2)))

    return error
        
def prediction(Pu, Qi, m, bu, bi):
    """Predict the rate for (u, i)
    
    Keyword arguments:
    Pu -- user u's feature vector
    Qi -- item i's feature vector
    m -- the overall average rating
    bu -- user u's bias
    bi -- item i's bias
    """
    rating = 0
    rating += m + bu + bi
    rating += numpy.dot(Pu, Qi)
    return rating

def rmse(P, Q, R, m, BU, BI, bias = False):
    """Compute the root mean square error
    
    Keyword arguments:
    P -- the users' matrix
    Q -- the items' matrix
    R -- the rating matrix
    m -- the overall average rating
    BU -- the user's bias vector
    BI -- the item's bias vector
    """
    Q = Q.T
    error = 0
    n = 0

    for u in xrange(len(R)):
        for i in xrange(len(R[u])):
            if R[u][i] > 0:
                n += 1

                if bias:
                    error += pow(R[u][i] - prediction(P[u,:], Q[:,i], m, BU[u], BI[i]), 2)
                else:
                    error += pow(R[u][i] - numpy.dot(P[u,:], Q[:,i]), 2)

    error = numpy.sqrt(error/n)
    return error
        
def load(filename, path = '../data/ml-100k/'):
    """Load datasets
    
    Keyword arguments:
    filename -- the name of the file to be loaded
    """
    R = numpy.zeros((943, 1682))
    data_file = open(path + filename)
    data_frame = csv.reader(data_file, delimiter = '\t')
    
    for idx, row in enumerate(data_frame):
        u = int(row[0])
        i = int(row[1])
        r = int(row[2])
        R[u-1, i-1] = r
    
    data_file.close()
    return R

############################################################################################
    
if __name__ == "__main__":
    R = load('ua.base')
    T = load('ua.test')
    
    N = len(R)
    M = len(R[0])
    f = 30
    P = numpy.float64(numpy.random.rand(N, f))
    Q = numpy.float64(numpy.random.rand(M, f))

    m = numpy.mean(R)
    BU = m - numpy.mean(R, axis = 1)
    BI = m - numpy.mean(R, axis = 0)

    print("Learning...")
    
    matrix_factorization(P, Q, R, f, m, BU, BI, 50, 0.015, 0.0001, 0.1, False)
    
    output = open('ua.txt', 'a')
    output.write("Root mean square error for the training set: "),
    output.write(str(rmse(P, Q, R, m, BU, BI, False)) + "\n")
    output.write("Root mean square error for the testing set: "),
    output.write(str(rmse(P, Q, T, m, BU, BI, False)) + "\n")
    output.close()

    print("The learning is terminated.")