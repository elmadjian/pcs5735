#Author: Carlos Eduardo Leão Elmadjian

import numpy as np

#Class to model a single neuron
#------------------------------
class Neuron():
    def __init__(self, idx, eta, inputs):
        self.idx = idx
        self.eta = eta
        self.weight  = [0.0 for i in range(inputs+1)]
        self.input   = [1.0 for i in range(inputs+1)]
        self.f_links = []
        self.b_links = []
        self.output  = None
        self.delta   = None

    def set_input(self, idx, x):
        self.input[idx] = x

    def set_weight(self, idx, w):
        self.weight[idx] = w

    def set_weight_list(self, new_list):
        self.weight = new_list

    def set_b_links(self, p):
        self.b_links.append(p)

    def set_f_links(self, p):
        self.f_links.append(p)

    def get_param(self):
        sum = 0
        for f in self.f_links:
            sum += f.weight[self.idx] * f.delta
        return sum

    def calculate_delta(self, param):
        self.delta = self.output * (1.0 - self.output) * param

    def calculate_output(self):
        y = np.dot(self.weight, self.input)
        self.output = 1/(1 + np.exp(y))

    def propagate_output(self):
        for l in self.f_links:
            l.set_input(self.idx, self.output)

    def update_weights(self):
        for i in range(len(self.weight)):
            self.weight[i] += self.eta * self.delta * self.input[i]


#Calculates the BPG algorithm for exercise 01
#--------------------------------------------
def backpropagation(training_set, n_in, n_out, n_hidden, eta=0.05, epochs=1):
    P_hidden = [Neuron(i+1, eta, n_in) for i in range(n_hidden)]
    P_out = [Neuron(i+1, eta, n_hidden) for i in range(n_out)]

    #setting links
    for pi in P_hidden:
        for pj in P_out:
            pj.set_b_links(pi)
            pi.set_f_links(pj)

    #manually setting weights
    P_out[0].set_weight_list([-0.1, -0.4, 0.1, 0.6])
    P_out[1].set_weight_list([0.6, 0.2, -0.1, -0.2])
    P_hidden[0].set_weight_list([0.1, -0.2, 0.0, 0.2])
    P_hidden[1].set_weight_list([0.2, -0.2, 0.1, 0.3])
    P_hidden[2].set_weight_list([0.5, 0.3, -0.4, 0.2])

    #training
    for i in range(epochs):
        for sample in training_set:

            #forwarding the input
            X = [1.0] + sample[0]
            Y = sample[1]
            for p in P_hidden:
                p.input = X
                p.calculate_output()
                p.propagate_output()
            for p in P_out:
                p.calculate_output()

            #backwards propagation
            for i in range(len(P_out)):
                P_out[i].calculate_delta(Y[i] - P_out[i].output)
            for p in P_hidden:
                p.calculate_delta(p.get_param())

            #updating network weights
            [p.update_weights() for p in P_out]
            [p.update_weights() for p in P_hidden]


    print("printing weights after %d epochs:" % epochs)
    print("\nP_hidden:")
    [print(['%.4f' % w for w in p.weight]) for p in P_hidden]
    print("\nP_out:")
    [print(['%.4f' % w for w in p.weight]) for p in P_out]

#initial setup
#--------------
def main():
    training_set = [[[0.6, 0.1, 0.2], [1, 0]], [[0.1, 0.5, 0.6], [0, 1]]]
    backpropagation(training_set, 3, 2, 3, epochs=1)

#-----------------------
if __name__ == "__main__":
    main()
