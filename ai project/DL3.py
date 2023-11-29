import numpy as np
import matplotlib.pyplot as plt
import os 
import h5py
from sklearn.metrics import classification_report, confusion_matrix

class DLModel:
    def __init__(self, name="Model"):
        self.name = name
        self.layers = [None]
        self._is_compiled = False

    def __str__(self):
        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers)-1) +"\n"
        if self._is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) +"\n"
            s += "\t\tloss function: " + self.loss + "\n\n"
        for i in range(1,len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"
        return s

    def add(self, layer):
        self.layers.append(layer)

    def squared_means(self, AL, Y):
        return (Y-AL)**2

    def squared_means_backward(self, AL, Y):
        return -2*(Y-AL)

    def cross_entropy(self, AL, Y):
        return np.where(Y == 0, -np.log(1-AL), -np.log(AL))

    def cross_entropy_backward(self, AL, Y):
        return np.where(Y == 0, 1/(1-AL), -1/AL)

    def categorical_cross_entropy(self, AL, Y):
        return np.where(Y == 1, -np.log(AL), 0)

    def categorical_cross_entropy_backward(self, AL, Y):
        return AL - Y

    def compile(self, loss, threshold=0.5):
        if loss not in ["cross_entropy", "squared_means", "categorical_cross_entropy"]:
            raise Exception(f"invalid value: loss must be either 'cross_entropy', 'categorical_cross_entropy' or 'squared_means'. (is currently {loss})")
        self.loss = loss
        self.threshold = threshold
        for func in [self.squared_means, self.cross_entropy, self.categorical_cross_entropy]:
            if func.__name__ == loss:
                self.loss_forward = func
        for func in [self.squared_means_backward, self.cross_entropy_backward, self.categorical_cross_entropy_backward]:
            if func.__name__[:-9] == loss:
                self.loss_backward = func
        self._is_compiled = True

    def compute_cost(self, AL, Y):
        m = AL.shape[1]
        costs = self.loss_forward(AL, Y)
        return np.sum(costs)/m

    def train(self, X, Y, num_iterations):
        print_ind = max(num_iterations // 100, 1)
        L = len(self.layers)
        costs = []
        for i in range(num_iterations):
            #forward propagation
            Al = X
            for l in range(1, L):
                Al = self.layers[l].forward_propagation(Al, False)
            #backward propagation
            dAl = self.loss_backward(Al, Y)
            for l in reversed(range(1, L)):
                dAl = self.layers[l].backward_propagation(dAl)
                #update parameters
                self.layers[l].update_parameters()
            #record progress
            if i > 0 and i % print_ind == 0:
                J = self.compute_cost(Al, Y)
                costs.append(J)
                print(f"Iteration: {i}, cost: {J}")
        return costs

    def predict(self, X):
        Al = X
        for l in range(1, len(self.layers)):
                Al = self.layers[l].forward_propagation(Al, True)
        if self.loss == self.categorical_cross_entropy:
            return np.where(Al==Al.max(axis=0),1,0)
        return np.where(Al > self.threshold, 1, 0)

    def save_weights(self, path):
        for l in range(1, len(self.layers)):
            self.layers[l].save_weights(path, "Layer{i}".format(i = l))

    def check_backward_propagation(self, X, Y, epsilon=1e-7):
        L = len(self.layers)
        #forward and backward prop to update dw and db in layers
        Al = X
        for l in range(1, L):
            Al = self.layers[l].forward_propagation(Al, False)
        dAl = self.loss_backward(Al, Y)
        for l in reversed(range(1, L)):
            dAl = self.layers[l].backward_propagation(dAl)
        #make return variables
        avg_diff = 0
        total_check = True
        problem_layer = L
        #calc the approx grad for each layer
        for l in reversed(range(1, L)):
            params_vec = self.layers[l].params_to_vec()
            grad_vec = self.layers[l].gradients_to_vec()
            n = len(params_vec)
            approx = np.zeros((n,), dtype=float)
            for i in range(n):
                v = np.copy(params_vec)
                v[i] += epsilon
                self.layers[l].vec_to_params(v)
                Al = X
                for layer in range(1, L):
                    Al = self.layers[layer].forward_propagation(Al, False)
                j_plus = self.compute_cost(Al, Y)
                v[i] -= 2*epsilon 
                self.layers[l].vec_to_params(v)
                Al = X
                for layer in range(1, L):
                    Al = self.layers[layer].forward_propagation(Al, False)
                j_minus = self.compute_cost(Al, Y) 
                approx[i] = j_plus-j_minus
            approx /= (2*epsilon) 
            diff = (np.linalg.norm(grad_vec-approx))/((np.linalg.norm(grad_vec))+(np.linalg.norm(approx)))
            check = (diff < epsilon)
            self.layers[l].vec_to_params(params_vec)
            print(diff)
            avg_diff += diff 
            if not check:
                total_check = False
                problem_layer = l
        avg_diff /= (L - 1)
        return total_check, avg_diff, problem_layer

    @staticmethod
    def to_one_hot(num_categories, Y):
        m = Y.shape[0]
        Y = Y.reshape(1, m)
        Y_new = np.eye(num_categories)[Y.astype('int32')]
        Y_new = Y_new.T
        Y_new = Y_new.reshape(num_categories, m)
        return Y_new

    def confusion_matrix(self, X, Y):
        prediction = self.predict(X)
        prediction_index = np.argmax(prediction, axis=0)
        Y_index = np.argmax(Y, axis=0)
        right = np.sum(prediction_index == Y_index)
        print("accuracy: ",str(right/len(Y[0])))
        print(confusion_matrix(prediction_index, Y_index))

class DLLayer:
    def __init__ (self, name, num_units, input_shape: tuple, activation="relu", W_initialization="random", alpha=0.01, optimization=None):
        if activation not in ["sigmoid", "tanh", "relu", "leaky_relu", "softmax", "trim_sigmoid", "trim_tanh", "trim_softmax"]: 
            raise Exception(f"invalid value: activation must be either 'sigmoid', 'trim_sigmoid', 'tanh', 'trim_tanh', 'relu', 'leaky_relu', 'softmax' or 'trim_softmax'. (is currently {activation})")
        if W_initialization not in ["random", "zeros", "He", "Xavier"] and W_initialization[-3:] != ".h5":
            print(f"Warning: W_initialization is currently {W_initialization}, which may be incorrect. it should be either 'random', 'He', 'Xavier', 'zeros' or a path to a .h5 file.")
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
        if activation[:4] == "trim":
            self.activation_trim = 1e-10
        for func in [self._sigmoid, self._trim_sigmoid, self._tanh, self._trim_tanh, self._relu, self._leaky_relu,
                    self._softmax, self._trim_softmax]:
            if func.__name__[1:] == activation:
                self.activation_forward = func
                break
        for func in [self._sigmoid_backward, self._tanh_backward, self._relu_backward, 
                     self._leaky_relu_backward, self._trim_tanh_backward, self._trim_sigmoid_backward,
                    self._softmax_backward, self._trim_softmax_backward]:
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
        elif W_initialization == "He":
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * np.sqrt(2/self._input_shape[0])
        elif W_initialization == "Xavier":
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * np.sqrt(1/self._input_shape[0])
        elif W_initialization == "random":
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * self.random_scale
        else: 
            try:
                with h5py.File(W_initialization, 'r') as hf:
                    self.W = hf['W'][:]
                    self.b = hf['b'][:]
            except (FileNotFoundError):
                raise NotImplementedError("Unrecognized initialization:", W_initialization)

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
        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()
        return s
        
    def _sigmoid(self, Z):
        return 1/(1+np.exp(-Z)) 

    def _tanh(self, Z):
        return np.tanh(Z)

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _leaky_relu(self, Z):
        return np.where(Z > 0, Z, Z * self.leaky_relu_d)

    def _softmax(self, Z):
        eZ = np.exp(Z)
        return eZ/np.sum(eZ, 0)

    def _trim_softmax(self, Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                eZ = np.exp(Z)
            except FloatingPointError:
                Z = np.where(Z > 100, 100,Z)
                eZ = np.exp(Z)
        A = eZ/np.sum(eZ, axis=0)
        return A

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
    
    def _trim_sigmoid_backward(self,dA):
        A = self._trim_sigmoid(self._Z)
        dZ = dA * A * (1-A)
        return dZ

    def _trim_tanh_backward(self,dA):
        A = self._trim_tanh(self._Z)
        dZ = dA * (1-A**2)
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
    
    def _softmax_backward(self, dZ):
        return dZ

    def _trim_softmax_backward(self, dZ):
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

    def save_weights(self, path, file_name):
        if not os.path.exists(path):
            os.makedirs(path)
        with h5py.File(path+"/"+file_name+'.h5', 'w') as hf:
            hf.create_dataset("W", data=self.W)
            hf.create_dataset("b", data=self.b)

    def params_to_vec(self):
        return np.concatenate((np.reshape(self.W,(-1,)),np.reshape(self.b, (-1,))), axis=0)

    def vec_to_params(self, vec):
        self.W = vec[0:self.W.size].reshape(self.W.shape)
        self.b = vec[self.W.size:].reshape(self.b.shape)

    def gradients_to_vec(self):
        return np.concatenate((np.reshape(self.dW,(-1,)),np.reshape(self.db, (-1,))), axis=0)

