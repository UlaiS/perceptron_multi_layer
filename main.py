import numpy as np
# guardar las activaciones y derivadas
# implementacion de la propagacion hacia atras (backpropagation)
# implementacion de un gradiante descendente (gradient descent)
# implementacion de entrenamiento
# entreneamiento de red con algunos agunos datos dummy
# hacer buenas predicciones

class MLP:
    """
    Clase de  un perceptron multicapa
    """

    def __init__(self, num_inputs, num_hidden, num_outputs):
        # def __init__(self, num_inputs=2, num_hidden=[4, 2], num_outputs=3):
        """
        Constructor para el MLP. Obtiene el numero de entradas,
        una variable de numero de capas ocultas y un numero de salidas
        :param num_inputs: numero de entradas
        :param num_hidden: una lista de enteros para las capas ocultas
        :param num_outputs:  numero de salidas
        """

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # creacion una representacion generia de las capas
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # inicializacion random weights para las capas
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forwar_propagate(self, inputs):
        """ Calculo propagacion hacia adelante  de la red basada en la senal de entrdas

        :param inputs: senal de entradas
        :return: valores de salida
        """

        # la capa de entrada de activacion es basado en las señales de entrada en sí
        activations = inputs
        self.activations[0] = inputs

        # iteracion a traves de las capas de la red
        for (i, w) in enumerate(self.weights):
            # calculando entradas netas para una determinada capa
            net_imputs = np.dot(activations, w)  # realiza la multiplicacion entre weights o pesos y activaciones
            # calculando las activaciones
            # se aplica la funcion sigmoid
            activations = self._sigmoid(net_imputs)
            self.activations[i + 1] = activations
        # a_3 = s(h_3)

        return activations

    def back_propagate(self, error, verbose=False):

        # dE/dW_i = ((y - a_[i+1]) s'(h_[i+1])) a_i
        # s' (h_[i+1]) = s(h_[i+1]) (1 - s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1]

        # dE/dw_[i-1] = ((y - a_[i+1]) s'(h_[i+1])) W_i s'(h_i) a_[i-1]

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivates for W{}: {}".format(i, self.derivatives[i]))

        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivates = self.derivatives[i]
            weights += derivates * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):
                # realiza la propagacion hacia adelante
                output = self.forwar_propagate(input)
                # calcula el error
                error = target - output
                # realiza la propagacion hacia atras
                self.back_propagate(error)
                # gradiante descendente
                self.gradient_descent(learning_rate)
                sum_error += self._mse(target, output)
            # error
            print("Error: {} en la iteracion/epoca {}".format(sum_error / len(inputs), i))

    def _mse(self, target, output):
        return np.average((target - output) ** 2)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    inputs = np.random.randint(8, size=(1000, 2))
    targets = np.array([[np.linalg.norm(np.array(i[0], i[1]))] for i in inputs])
    # creacion de MLP
    mlp = MLP(2, [5], 1)
    # entrenamiento
    mlp.train(inputs, targets, 50, 0.1)

    input = np.array([7, 5])
    target = np.array([1])

    output = mlp.forwar_propagate(input)
    print()
    print()
    print(" eje 'x' {} y eje 'y' {} dentro del radio {}".format(input[0], input[1], output[0]))