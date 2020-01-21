from DeepNet.simulation import generate_data
from LastProject.TwoStepNN import TwoStepNet
import numpy as np
from DeepNet.dnp import DeepNet
from sklearn.linear_model import LogisticRegression
from pyHSICLasso import HSICLasso


def right_select(true, select):
    count = 0
    for i in range(len(true)):
        if true[i] in select:
            count += 1
    return count


def lasso_selection(x, y, num: int) -> list:
    c1 = 1
    c2 = 1
    y = y.ravel()
    model1 = LogisticRegression(penalty='l1', solver='liblinear', C=c1)
    model1.fit(x, y)
    coefs = model1.coef_.ravel()
    number_of_selection = len([i for i in coefs if i != 0])
    if number_of_selection == num:
        return sorted([i+1 for i, j in enumerate(coefs) if j != 0])
    while number_of_selection > num:
        print(f"Number of selected variables is {number_of_selection}. {c1}")
        c1 /= 2
        model1 = LogisticRegression(penalty='l1', solver='liblinear', C=c1)
        model1.fit(x, y)
        coefs = model1.coef_.ravel()
        number_of_selection = len([i for i in coefs if i != 0])
    model1 = LogisticRegression(penalty='l1', solver='liblinear', C=c2)
    model1.fit(x, y)
    coefs = model1.coef_.ravel()
    number_of_selection = len([i for i in coefs if i != 0])
    while number_of_selection < num:
        print(f"Number of selected variables is {number_of_selection}.")
        c2 *= 2
        model1 = LogisticRegression(penalty='l1', solver='liblinear', C=c2)
        model1.fit(x, y)
        coefs = model1.coef_.ravel()
        number_of_selection = len([i for i in coefs if i != 0])
    while number_of_selection != num:
        print(f"Number of selected variables is {number_of_selection}.")
        c = (c1 + c2) / 2
        model1 = LogisticRegression(penalty='l1', solver='liblinear', C=c)
        model1.fit(x, y)
        coefs = model1.coef_.ravel()
        number_of_selection = len([i for i in coefs if i != 0])
        if number_of_selection > num:
            c2 = c
        else:
            c1 = c
    return sorted([i+1 for i, j in enumerate(coefs) if j != 0])





m = 10
selection_results = []
counts = []
for seed in range(10000):
    if len(selection_results) > 20:
        break
    else:
        x_train, x_test, y_train, y_test = generate_data(seed, m, flip=0, n=2000, train_size=1000)
        if y_train.mean() < 0.42 or y_train.mean() > 0.58:
            continue
        else:
            model = TwoStepNet(max_feature=m, hidden_size=[50])
            selection_result = model.feature_selection(x_train, y_train, sample_prop=0.7, num_bagging=10, verbosity=1, appear_prop=0.3)
            selection_results.append(selection_result)
            counts.append(right_select(list(range(1, m+1)), selection_result))
            print(f"Seed: {seed}. Right selection {right_select(list(range(1, m+1)), selection_result) / m}, accuracy {np.mean(counts)}")


print(np.mean(counts), np.std(counts))


selection_results_dnp = []
for seed in range(10000):
    if len(selection_results_dnp) > 20:
        break
    else:
        x_train, x_test, y_train, y_test = generate_data(seed, m, flip=0, n=2000, train_size=1000)
        if y_train.mean() < 0.42 or y_train.mean() > 0.58:
            continue
        else:
            model = DeepNet(max_feature=m, hidden_size=[50, 30, 15, 10])
            model.train(x_train, y_train)
            selection_result_dnp = model.S[1:]
            selection_results_dnp.append(selection_result_dnp)
            print(f"Seed: {seed}. Right selection {right_select(list(range(1, m+1)), selection_result_dnp) / m}")

counts_dnp = []
for i in selection_results_dnp:
    counts_dnp.append(right_select(list(range(1, m+1)), i))

print(np.mean(counts_dnp), np.std(counts_dnp))


m = 2
selection_results_lasso = []
counts_lasso = []
for seed in range(10000):
    if len(selection_results_lasso) > 20:
        break
    else:
        x_train, x_test, y_train, y_test = generate_data(seed, m, flip=0, n=2000, train_size=1000)
        if y_train.mean() < 0.42 or y_train.mean() > 0.58:
            continue
        else:
            selection_result_lasso = lasso_selection(x_train, y_train, m)
            selection_results_lasso.append(selection_result_lasso)
            counts_lasso.append(right_select(list(range(1, m+1)), selection_result_lasso))
            print(f"Selection results {selection_result_lasso}.")
            print(f"Seed: {seed}. Right selection {right_select(list(range(1, m+1)), selection_result_lasso) / m}, "
                  f"accuracy {np.mean(counts_lasso)}")


print(np.mean(counts_lasso), np.std(counts_lasso))




m = 10
selection_results_hslasso = []
counts_hslasso = []
for seed in range(10000):
    if len(selection_results_hslasso) > 20:
        break
    else:
        x_train, x_test, y_train, y_test = generate_data(seed, m, flip=0, n=2000, train_size=1000)
        if y_train.mean() < 0.42 or y_train.mean() > 0.58:
            continue
        else:
            model = HSICLasso()
            model.input(x_train, y_train.ravel())
            model.classification(m)
            index = model.get_index()
            selection_result_hslasso = [i+1 for i in index]
            selection_results_hslasso.append(selection_result_hslasso)
            counts_hslasso.append(right_select(list(range(1, m+1)), selection_result_hslasso))
            print(f"Selection results {selection_result_hslasso}.")
            print(f"Seed: {seed}. Right selection {right_select(list(range(1, m+1)), selection_result_hslasso) / m}, "
                  f"accuracy {np.mean(counts_hslasso)}")


print(np.mean(counts_hslasso), np.std(counts_hslasso))



