import numpy as np
from dnp import DeepNet
from simulation import generate_data


def generateXY(seed: int, struct: str = 'Linear', response: str = 'classification', n: int = 1000, p: int = 10000,
               s: int = 5, mu: float = 0, sigma: float = 0) -> (np.array, np.array):
    """generates data"""
    if struct == 'NeuralNet':
        if response == 'classification':
            x, _, y, _ = generate_data(seed, s, n+1, n, 0, mu, sigma, False)
        elif response == 'regression':
            x, _, y, _ = generate_data(seed, s, n+1, n, 0, mu, sigma, True)
            y = y.ravel() + np.random.randn(n) * 3
        else:
            print("response need to be classification or regression")
            return None, None
    elif struct in ['Linear', 'Additive']:
        x = np.random.rand(n, p).reshape(n, p) * 2 - 1
        if struct == 'Linear':
            beta = np.random.randn(s)
            eta = np.matmul(x[:, :s], beta)
        else:
            eta = np.sin(x[:, 0]) + x[:, 1] + np.exp(x[:, 2]) + np.square(x[:, 3]) + np.log(x[:, 4] + 2) - 2
        if response == 'classification':
            prob = 1 / (1 + np.exp(-eta))
            y = np.array([np.random.binomial(1, prob[i]) for i in range(n)]).reshape(-1, 1)
        elif response == 'regression':
            y = eta + np.random.randn(n) * 3
        else:
            print("response need to be classification or regression")
            return None, None
    return x, y


n = 1000
p = 10000
s = 5
mu = 0
sigma = 1
rep = 100

sums = []
for prior in range(5):
    selections = []
    for seed in range(rep*1000):
        if len(selections) >= rep:
            break
        initial = np.random.choice(5, prior, False)
        initial = [i + 1 for i in initial]
        x, y = generateXY(seed, 'NeuralNet', 'classification', n, p, s, mu, sigma)
        if y.mean() < 0.1 or y.mean() > 0.9:
            continue
        model = DeepNet(max_feature=s, hidden_size=[50, 30, 10], regression=False)
        selection = model.train_return_next(x, y, return_select=True, initial=initial, verbosity=2)
        selections.append(selection[0])
        print(prior, seed, selection, len([i for i in selections if i not in [1, 2, 3, 4, 5]]))
    correct = [1, 2, 3, 4, 5]
    rightselection = [1 if selections[i] in correct else 0 for i in range(len(selections))]
    sums.append(np.sum(rightselection))

print(sums)

