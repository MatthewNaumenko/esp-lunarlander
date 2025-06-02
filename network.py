import numpy as np

class FeedforwardNetwork:
    """
    # Однослойная прямораспространённая сеть (input -> hidden -> output)
    """

    def __init__(self, input_size, hidden_size, output_size,
                 weights_input_hidden=None, weights_hidden_output=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Инициализация весов если не переданы
        if weights_input_hidden is None:
            self.weights_input_hidden = np.random.randn(hidden_size, input_size) * 0.1
        else:
            self.weights_input_hidden = weights_input_hidden

        if weights_hidden_output is None:
            self.weights_hidden_output = np.random.randn(output_size, hidden_size) * 0.1
        else:
            self.weights_hidden_output = weights_hidden_output

    @staticmethod
    def from_weights(data):
        return FeedforwardNetwork(
            data['input_size'],
            data['hidden_size'],
            data['output_size'],
            data['weights_input_hidden'],
            data['weights_hidden_output']
        )
