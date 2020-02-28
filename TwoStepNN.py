from LastProject.dnp import DeepNet, NeuralNet
import numpy as np
import torch
import copy
from typing import Union, List


class TwoStepNet(DeepNet):

    def __init__(self, max_feature: int, num_classes: int = 2, hidden_size: list = None,
                 q: int = 2, num_dropout: int = 50, regression: bool = False):
        """
        init function
        """
        super(TwoStepNet, self).__init__(max_feature, num_classes, hidden_size, q, num_dropout, regression)
        self.final_model = None
        self.final_model_l1 = None

    def feature_selection_(
            self, x, y, num_bagging: int = 20, sample_prop: float = 0.8, appear_prop: float = 0.5, verbosity: int = 0,
            learning_rate: float = 0.005, epochs: int = 50):
        """
        Feature selection using DNP with bagging
        @param x: the training data
        @param y: the training label
        @param num_bagging: number of bagging to do feature selection
        @param sample_prop: sampling proportion
        @param appear_prop: the proportion threshold for a feature to be selected
        @param verbosity: 0-print everything; 1-print result only; 2-print nothing
        @param learning_rate: learning rate
        @param epochs: number of epochs
        @return: the feature set
        """
        if sample_prop > 1 or sample_prop <= 0:
            raise ValueError("Sampling proportion must be > 0 and <= 1.")
        size = x.shape[0]
        selection = {}
        for i in range(num_bagging):
            index = np.random.choice(list(range(size)), size=int(size * sample_prop), replace=False)
            x_train = x[index, :]
            y_train = y[index, :]
            self.train(x_train, y_train, verbosity=verbosity, learning_rate=learning_rate, epochs=epochs)
            if verbosity == 0:
                print(f"Features selected {self.S}")
            for idx in self.S:
                if idx in selection.keys():
                    selection[idx] += 1
                else:
                    selection[idx] = 1
        if verbosity <= 1:
            print(f"The proportions of appearance are: {[i / num_bagging for i in list(selection.values())]}")
        keys = []
        values = []
        for key, value in selection.items():
            keys.append(key)
            values.append(value)
        res = list(zip(keys, values))
        res = sorted(res, key=lambda va: va[1], reverse=True)
        res = [i[0] for i in res if i[1] / num_bagging >= appear_prop]
        if 0 in res:
            res.remove(0)
        if len(res) <= self.max_feature:
            return res
        else:
            return res[:self.max_feature]

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
        for i in new_added:
            indices.append(full[i])
        return sorted(indices)

    def feature_selection(
            self, x, y, num_bagging: int = 20, sample_prop: float = 0.8, appear_prop: float = 0.5, verbosity: int = 0,
            learning_rate: float = 0.005, epochs: int = 50):
        """
        perform feature selection recursively until no new variable is added
        @param x: the training data
        @param y: the training label
        @param num_bagging: number of bagging to do feature selection
        @param sample_prop: sampling proportion
        @param appear_prop: the proportion threshold for a feature to be selected
        @param verbosity: 0-print everything; 1-print result only; 2-print nothing
        @param learning_rate: learning rate
        @param epochs: number of epochs
        @return: the feature set
        """
        result = [0]
        iters = 0
        max_feature = self.max_feature
        while self.max_feature > 0 and iters < 5:
            x_sub = np.delete(x, [i - 1 for i in result[1:]], axis=1)
            print(f"The dimension of x is: {x_sub.shape}")
            new_added = self.feature_selection_(x_sub, y, num_bagging, sample_prop, appear_prop, verbosity=verbosity,
                                                learning_rate=learning_rate, epochs=epochs)
            new_added = self.find_index(new_added, result)
            if len(new_added) == 0:
                iters += 1
            print(f"Newly added variables are: {new_added}")
            result += new_added
            result = list(set(result))
            print(f"Selection results are: {result}")
            self.max_feature = max_feature - len(result) + 1
            print(f"Max feature is {self.max_feature}")
        print(f"Final selection is: {result}")
        self.S = result
        return result

    @staticmethod
    def l1_regularizer(model, lambda_l1=0.01):
        lossl1 = 0
        for model_param_name, model_param_value in model.named_parameters():
            if model_param_name.endswith('weight'):
                lossl1 += lambda_l1 * model_param_value.abs().sum()
        return lossl1

    @staticmethod
    def orth_regularizer(model, lambda_orth=0.01):
        lossorth = 0
        for model_param_name, model_param_value in model.named_parameters():
            if model_param_name.endswith('weight'):
                param_flat = model_param_value.view(model_param_value.shape[0], -1)
                sym = torch.mm(param_flat, torch.t(param_flat))
                sym -= torch.eye(param_flat.shape[0])
                lossorth += lambda_orth * sym.abs().sum()
        return lossorth

    def estimate(self, x, y, l1: bool = False, lam: float = None, l1_cv: bool = False, lam_cv: list = None,
                 selection: list = None, verbosity: int = 0, learning_rate: float = 0.1, batch_size: int = 32,
                 epochs: int = 50, val: list = None, xavier: bool = False, early_stopping_rounds: int = 100):
        """
        estimates the neural network model
        @param x: training data
        @param y: training label
        @param l1: whether to use l1 norm regularization
        @param lam: tuning parameter
        @param l1_cv: whether to use cross validation to select the best lam
        @param lam_cv: a list of tuning parameters
        @param selection: a list of selected indices
        @param verbosity: 0, 1, 2 print message
        @param learning_rate: learning rate
        @param batch_size: batch size
        @param epochs: number of epochs
        @param val: a list of x_val and y_val
        @param xavier: whether to use xavier initialization
        @param early_stopping_rounds: patience to early stop
        @return: null
        """
        if selection is not None:
            x = x[:, selection]
        else:
            x = x[:, [i - 1 for i in self.S if i != 0]]
        input_size = x.shape[1]
        hidden_size = self.hidden_size
        output_size = self.num_classes
        model = NeuralNet(input_size, hidden_size, output_size)
        if xavier:
            torch.nn.init.xavier_normal_(model.input.weight, gain=torch.nn.init.calculate_gain('relu'))
            for i in range(len(hidden_size)-1):
                torch.nn.init.xavier_normal_(model.hiddens[i].weight, gain=torch.nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_normal_(model.output.weight, gain=torch.nn.init.calculate_gain('relu'))
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        val_list = []
        best_model = None
        best_acc = -np.inf
        trainset = []
        for i in range(x.shape[0]):
            trainset.append([x[i, :], y[i]])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        for e in range(epochs):
            running_loss = 0
            for data, label in trainloader:
                input_0 = data.view(data.shape[0], -1)
                optimizer.zero_grad()
                output = model(input_0.float())
                loss = torch.nn.CrossEntropyLoss()(output, label.squeeze(1))
                if l1:
                    if lam is None:
                        raise ValueError("lam needs to be specified when l1 is True.")
                    else:
                        loss = loss + self.l1_regularizer(model, lam)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if val is None:
                if verbosity == 0:
                    print(f"Epoch: {e + 1}\nTraining loss: {running_loss/len(trainloader)}")
            else:
                x_v = self.numpy_to_torch(val[0])
                if selection is not None:
                    x_v = x_v[:, selection]
                else:
                    x_v = x_v[:, [i-1 for i in self.S if i != 0]]
                y_v = self.numpy_to_torch(val[1])
                pred = model(x_v.float())
                pred = torch.argmax(pred, 1)
                accuracy_1 = 1 - torch.mean(abs(pred.squeeze() - y_v.squeeze()).float()).item()
                val_list.append(accuracy_1)
                if accuracy_1 > best_acc:
                    best_acc = accuracy_1
                    best_model = copy.deepcopy(model)
                if verbosity == 0:
                    print(f"Epoch: {e + 1}\nTraining loss: {running_loss/len(trainloader)}. Validation accuracy "
                          f"is {accuracy_1}")
            if val is not None and self.early_stopping(val_list, early_stopping_rounds):
                self.final_model = copy.deepcopy(best_model)
                print(f"Early stopped with patience {early_stopping_rounds}.")
                break
        if val is not None:
            self.final_model = copy.deepcopy(best_model)
        else:
            self.final_model = model

    def l1_thres(self, model: torch.nn.Module, quantile: Union[float, List[float]] = 0.5, lam: float = None):
        """
        perform the l1 thresholding function to neural network. Smaller weights are set to zero.
        @param model: the neural network model
        @param quantile: the tuning parameter
        @param lam: tuning parameter, specifying lam will override quantile
        @return: the new neural network model
        """
        if lam is None:
            if type(quantile) is float:
                quantile = [quantile] * (len(self.hidden_size) + 1)
            if type(quantile) is list:
                if len(quantile) >= len(self.hidden_size) + 1:
                    quantile = quantile[:(len(self.hidden_size) + 1)]
                else:
                    quantile = quantile + [0.5] * (len(self.hidden_size) + 1 - len(quantile))
            qt = np.percentile(model.input.weight.data.abs(), quantile[0])
            model.input.weight.data[model.input.weight.data.abs() < qt] = 0.0
            model.input.weight.data[model.input.weight.data > 0] -= qt
            model.input.weight.data[model.input.weight.data < 0] += qt
            for i in range(len(self.hidden_size) - 1):
                qt = np.percentile(model.hiddens[i].weight.data.abs(), quantile[i+1])
                model.hiddens[i].weight.data[model.hiddens[i].weight.data.abs() < qt] = 0.0
                model.hiddens[i].weight.data[model.hiddens[i].weight.data.abs() > 0] -= qt
                model.hiddens[i].weight.data[model.hiddens[i].weight.data.abs() < 0] += qt
            qt = np.percentile(model.output.weight.data.abs(), quantile[-1])
            model.output.weight.data[model.output.weight.data.abs() < qt] = 0.0
            model.output.weight.data[model.output.weight.data.abs() > 0] -= qt
            model.output.weight.data[model.output.weight.data.abs() < 0] += qt
        else:
            model.input.weight.data[model.input.weight.data.abs() < lam] = 0.0
            model.input.weight.data[model.input.weight.data > 0] -= lam
            model.input.weight.data[model.input.weight.data < 0] += lam
            for i in range(len(self.hidden_size) - 1):
                model.hiddens[i].weight.data[model.hiddens[i].weight.data.abs() < lam] = 0.0
                model.hiddens[i].weight.data[model.hiddens[i].weight.data.abs() > 0] -= lam
                model.hiddens[i].weight.data[model.hiddens[i].weight.data.abs() < 0] += lam
            model.output.weight.data[model.output.weight.data.abs() < lam] = 0.0
            model.output.weight.data[model.output.weight.data.abs() > 0] -= lam
            model.output.weight.data[model.output.weight.data.abs() < 0] += lam
        return model

    @staticmethod
    def early_stopping(val_list: list, rounds: int) -> bool:
        """
        input a list of validation metrics and number of early stopping rounds and return whether to stop
        @param val_list: list of validation metrics
        @param rounds: number of early stopping rounds
        @return: whether to early stop
        """
        best_idx = val_list.index(max(val_list))
        if len(val_list) - best_idx > rounds:
            return True
        else:
            return False

    def estimate_l1thres(self, x, y, quantile: Union[float, List[float]] = None, lam: float = None,
                         selection: list = None, verbosity: int = 0, learning_rate: float = 0.1, batch_size: int = 32,
                         epochs: int = 50, val: list = None, xavier: bool = False, early_stopping_rounds: int = 30):
        """
        estimates the neural network model with sparsity
        @param x: training data
        @param y: training label
        @param quantile: between 0 and 1, tuning parameter
        @param lam: tuning parameter, specifying lam will override quantile
        @param selection: a list of selected indices
        @param verbosity: 0, 1, 2 print message
        @param learning_rate: learning rate
        @param batch_size: batch size
        @param epochs: number of epochs
        @param val: a list of x_val and y_val
        @param xavier: whether to use xavier initialization
        @param early_stopping_rounds: patience to stop
        @return: null
        """
        if selection is not None:
            x = x[:, selection]
        else:
            x = x[:, [i - 1 for i in self.S if i != 0]]
        input_size = x.shape[1]
        hidden_size = self.hidden_size
        output_size = self.num_classes
        model = NeuralNet(input_size, hidden_size, output_size)
        if xavier:
            torch.nn.init.xavier_normal_(model.input.weight, gain=torch.nn.init.calculate_gain('relu'))
            for i in range(len(hidden_size)-1):
                torch.nn.init.xavier_normal_(model.hiddens[i].weight, gain=torch.nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_normal_(model.output.weight, gain=torch.nn.init.calculate_gain('relu'))
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        trainset = []
        for i in range(x.shape[0]):
            trainset.append([x[i, :], y[i]])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        val_list = []
        best_model = None
        best_acc = -np.inf
        for e in range(epochs):
            running_loss = 0
            for data, label in trainloader:
                input_0 = data.view(data.shape[0], -1)
                optimizer.zero_grad()
                output = model(input_0.float())
                loss = torch.nn.CrossEntropyLoss()(output, label.squeeze(1))
                loss.backward()
                optimizer.step()
                model = self.l1_thres(model, quantile, lam)
                output = model(input_0.float())
                loss = torch.nn.CrossEntropyLoss()(output, label.squeeze(1))
                running_loss += loss.item()
            if val is None:
                if verbosity == 0:
                    print(f"Epoch: {e + 1}\nTraining loss: {running_loss/len(trainloader)}")
            else:
                x_v = self.numpy_to_torch(val[0])
                if selection is not None:
                    x_v = x_v[:, selection]
                else:
                    x_v = x_v[:, [i-1 for i in self.S if i != 0]]
                y_v = self.numpy_to_torch(val[1])
                pred = model(x_v.float())
                pred = torch.argmax(pred, 1)
                accuracy_1 = 1 - torch.mean(abs(pred.squeeze() - y_v.squeeze()).float()).item()
                val_list.append(accuracy_1)
                if accuracy_1 > best_acc:
                    best_acc = accuracy_1
                    best_model = copy.deepcopy(model)
                if verbosity == 0:
                    print(f"Epoch: {e + 1}\nTraining loss: {running_loss/len(trainloader)}. Validation accuracy "
                          f"is {accuracy_1}")
            if val is not None and self.early_stopping(val_list, early_stopping_rounds):
                self.final_model = copy.deepcopy(best_model)
                print(f"Early stopped with patience {early_stopping_rounds}.")
                break
        if val is not None:
            self.final_model = copy.deepcopy(best_model)
        else:
            self.final_model = model

    def predict_normal(self, x, selection: list = None, prob: bool = False) -> torch.tensor:
        """
        making prediction of x using the neural network model after variable selection
        @param x: testing set
        @param selection: list of indices selected
        @param prob: True to return probabilities
        @return: the prediction of x
        """
        if self.final_model is None:
            print(f"Model not trained, use estimate first.")
        else:
            if selection is None:
                x = x[:, [i - 1 for i in self.S if i != 0]]
            else:
                x = x[:, selection]
            x = self.numpy_to_torch(x)
            y_pred = self.final_model(x.float())
            if prob:
                return y_pred
            y_pred = torch.argmax(y_pred, dim=1)
            return y_pred

    def predict_bagging(self, x, selection: list = None, num_bagging: int = 100, drop_prop: list = None):
        self.nnmodel = self.final_model
        y_pred = None
        if selection is None:
            x = x[:, [i - 1 for i in self.S if i != 0]]
        else:
            x = x[:, selection]
        x = self.numpy_to_torch(x)
        if drop_prop is None:
            drop_prop = [0.5] * len(self.hidden_size)
        self.dropout_prop = drop_prop
        for i in range(num_bagging):
            model = self.dropout()
            if y_pred is None:
                y_pred = torch.argmax(model(x.float()), dim=1)
            else:
                y_pred += torch.argmax(model(x.float()), dim=1)
        y_pred = y_pred.float() / num_bagging
        y_pred = torch.where(y_pred < 0.5, torch.tensor(0), torch.tensor(1))
        return y_pred

