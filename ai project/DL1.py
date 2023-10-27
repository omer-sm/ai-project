import numpy as np
import matplotlib.pyplot as plt

class DLLayer:
    def __init__ (self, name, num_units, input_shape: tuple, activation="relu", W_initialization="random", alpha=0.01, optimization=None):
        if activation not in ["sigmoid", "tanh", "relu", "leaky_relu", "trim_sigmoid", "trim_tanh"]: 
            raise Exception(f"invalid value: activation must be either 'sigmoid', 'trim_sigmoid', 'tanh', 'trim_tanh', 'relu' or 'leaky_relu'. (is currently {activation})")
        if W_initialization not in ["random", "zeros"]:
            raise Exception(f"invalid value: W_initialization must be either 'random' or 'zeros'. (is currently {W_initialization})")
        if not (optimization is None) and optimization != "adaptive":
            raise Exception(f"invalid value: optimization must be either None or 'adaptive'. (is currently {optimization})")
        self.name = name
        self._num_units = num_units
        self._activation = activation
        self.W_initialization = W_initialization
        self.alpha = alpha
        self._optimization = optimization
        self.random_scale = 0.01
        self._input_shape = input_shape
        if activation == "leaky_relu":
            self.leaky_relu_d = 0.1
        for func in [self._sigmoid, self._trim_sigmoid, self._tanh, self._trim_tanh, self._relu, self._leaky_relu]:
            if func.__name__[1:] == activation:
                self.activation_forward = func
                break
        for func in [self._sigmoid_backward, self._tanh_backward, self._relu_backward, self._leaky_relu_backward]:
            if func.__name__[1:-9] == activation:
                self.activation_backward = func
                break
        if optimization == "adaptive":
            self._adaptive_alpha_b = np.full((self._num_units, 1), self.alpha)
            self._adaptive_alpha_W = np.full((self._num_units, *(self._input_shape)),self.alpha)
            self.adaptive_cont = 1.1
            self.adaptive_switch = -0.5
        self.init_weights(W_initialization)

    def init_weights(self, W_initialization):
        self.b = np.zeros((self._num_units,1), dtype=float)
        if W_initialization == "zeros":
            self.W = np.zeros((self._num_units, *(self._input_shape)), dtype=float)
        else:
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * self.random_scale

    def __str__(self):
        s = self.name + " Layer:\n"
        s += "\tnum_units: " + str(self._num_units) + "\n"
        s += "\tactivation: " + self._activation + "\n"
        if self._activation == "leaky_relu":
            s += "\t\tleaky relu parameters:\n"
            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d)+"\n"
        s += "\tinput_shape: " + str(self._input_shape) + "\n"
        s += "\tlearning_rate (alpha): " + str(self.alpha) + "\n"
        #optimization
        if self._optimization == "adaptive":
            s += "\t\tadaptive parameters:\n"
            s += "\t\t\tcont: " + str(self.adaptive_cont)+"\n"
            s += "\t\t\tswitch: " + str(self.adaptive_switch)+"\n"
        # parameters
        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"
        s += "\t\tshape weights: " + str(self.W.shape)+"\n"
        #plt.hist(self.W.reshape(-1))
        #plt.title("W histogram")
        #plt.show()
        return s
        
    def _sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def _tanh(self, Z):
        return np.tanh(Z)

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _leaky_relu(self, Z):
        return np.where(Z > 0, Z, Z * self.leaky_relu_d)

    def _trim_sigmoid(self,Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                A = 1/(1+np.exp(-Z))
            except FloatingPointError:
                Z = np.where(Z < -100, -100,Z)
                A = A = 1/(1+np.exp(-Z))
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < TRIM,TRIM,A)
            A = np.where(A > 1-TRIM,1-TRIM, A)
        return A

    def _trim_tanh(self,Z):
        A = np.tanh(Z)
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < -1+TRIM,TRIM,A)
            A = np.where(A > 1-TRIM,1-TRIM, A)
        return A

    def forward_propagation(self, A_prev, is_predict):
        self._A_prev = np.array(A_prev, copy=True)
        self._Z = np.dot(self.W, A_prev) + self.b
        A = self.activation_forward(self._Z)
        return A

    def _sigmoid_backward(self, dA):
        A = self._sigmoid(self._Z)
        dZ = dA * A * (1 - A)
        return dZ

    def _tanh_backward(self, dA):
        dZ = (1 - np.tanh(self._Z)**2) * dA
        return dZ

    def _relu_backward(self,dA):
        dZ = np.where(self._Z <= 0, 0, dA)
        return dZ

    def _leaky_relu_backward(self, dA):
        dZ = np.where(self._Z <= 0, dA * self.leaky_relu_d, dA)
        return dZ

    def backward_propagation(self, dA):
        dZ = self.activation_backward(dA)
        m = dZ.shape[1]
        self.db = np.sum(dZ , axis=1, keepdims=True)/m
        self.dW = (dZ @ (self._A_prev.T))/m
        dA_Prev = self.W.T @ dZ
        return dA_Prev

    def update_parameters(self):
        if self._optimization is None:
            self.W -= self.dW * self.alpha
            self.b -= self.db * self.alpha
        elif self._optimization == "adaptive":
            self._adaptive_alpha_W *= np.where(self._adaptive_alpha_W * self.dW > 0, self.adaptive_cont, self.adaptive_switch)
            self._adaptive_alpha_b *= np.where(self._adaptive_alpha_b * self.db > 0, self.adaptive_cont, self.adaptive_switch)
            self.W -= self._adaptive_alpha_W
            self.b -= self._adaptive_alpha_b