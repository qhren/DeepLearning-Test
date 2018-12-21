from sigmoid import sigmoid


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

