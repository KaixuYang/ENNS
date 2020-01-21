from LastProject.dnp import DeepNet, NeuralNet
import numpy as np
import torch
from torch import nn


class TwoStepNet(DeepNet):

    def feature_selection_(
            self, x, y, num_bagging: int = 20, sample_prop: float = 0.8, appear_prop: float = 0.5, verbosity: int = 0):
        """
        Feature selection using DNP with bagging
        @param x: the training data
        @param y: the training label
        @param num_bagging: number of bagging to do feature selection
        @param sample_prop: sampling proportion
        @param appear_prop: the proportion threshold for a feature to be selected
        @param verbosity: 0-print everything; 1-print result only; 2-print nothing
        @return: the feature set
        """
        if sample_prop > 1 or sample_prop <= 0:
            raise ValueError("Sampling proportion must be > 0 and <= 1.")
        size = x.shape[0]
        selection = {}
        result = []
        for i in range(num_bagging):
            index = np.random.choice(list(range(size)), size=int(size * sample_prop), replace=False)
            x_train = x[index, :]
            y_train = y[index, :]
            self.train(x_train, y_train, verbosity=verbosity)
            if verbosity <= 1:
                print(f"Features selected {self.S}")
            for id in self.S:
                if id in selection.keys():
                    selection[id] += 1
                else:
                    selection[id] = 1
        if verbosity <= 1:
            print(f"The proportions of appearance are: {[i / num_bagging for i in list(selection.values())]}")
        for key in selection.keys():
            if selection[key] / num_bagging > appear_prop:
                result.append(key)
        return sorted(result)

    def find_index(self, new_added: list, result: list):
        """
        find the true index of new added variable
        @param new_added: list of indices of newly added variables
        @param result: list of indices of already included variables
        @return: list of indices of newly added variables, adjusted
        """
        indices = []
        full = list(range(self.p + 1))
        for i in result[1:][::-1]:
            del full[i]
        print(f"full is: {full}")
        for i in new_added:
            indices.append(full[i])
        return sorted(indices)

    def feature_selection(
            self, x, y, num_bagging: int = 20, sample_prop: float = 0.8, appear_prop: float = 0.5, verbosity: int = 0):
        """
        perform feature selection recursively until no new variable is added
        @param x: the training data
        @param y: the training label
        @param num_bagging: number of bagging to do feature selection
        @param sample_prop: sampling proportion
        @param appear_prop: the proportion threshold for a feature to be selected
        @param verbosity: 0-print everything; 1-print result only; 2-print nothing
        @return: the feature set
        """
        result = [0]
        iters = 0
        max_feature = self.max_feature
        while self.max_feature > 0 and iters < 10:
            iters += 1
            x_sub = np.delete(x, result[1:], axis=1)
            print(f"The dimension of x is: {x_sub.shape}")
            new_added = self.feature_selection_(x_sub, y, num_bagging, sample_prop, appear_prop, verbosity=verbosity)
            new_added = self.find_index(new_added, result)
            print(f"Newly added variables are: {new_added}")
            result += new_added
            result = list(set(result))
            print(f"Selection results are: {result}")
            self.max_feature = max_feature - len(result) + 1
            print(f"Max feature is {self.max_feature}")
        print(f"Final selection is: {result}")
        self.S = result
        return result

    def estimation(self, x, y):
        """
        estimates the neural network model
        @param x: training data
        @param y: training label
        @return:
        """
        input_size = x.shape[1]
        hidden_size = self.hidden_size
        output_size = y.shape[1]
        self.model = NeuralNet(input_size, hidden_size, output_size)