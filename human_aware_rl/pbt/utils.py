import math
import numpy as np
from sklearn.preprocessing import normalize


def normalization(x):
    return normalize([x])[0]


def exp_prob(x, t=1):
    return np.exp(x*t)/np.sum(np.exp(x*t))


def probability(x):
    x = np.array(x)
    return x/(np.sum(x)+1e-13)


def ucb_shapley_value(x, visits, c=1):
    if len(list(visits.values())) == 0:
        sum_visits = 0
    else:
        sum_visits = np.sum(list(visits.values()))
    # print(sum_visits)
    u = []
    for key, value in enumerate(x):
        if str(key) in visits.keys():
            u.append(math.sqrt(sum_visits) / (1 + visits[str(key)]))
        else:
            u.append(math.sqrt(sum_visits) / (1 + 0))
    u = normalization(u)
    x = normalization(x)
    x = exp_prob(x, t=10)
    x = x + c * u
    return x


def inversed_ucb_shapley_value(x, visits, c=1):
    if len(list(visits.values())) == 0:
        sum_visits = 0
    else:
        sum_visits = np.sum(list(visits.values()))
    u = []
    for key, value in enumerate(x):
        if str(key) in visits.keys():
            u.append(math.sqrt(sum_visits) / (1 + visits[str(key)]))
        else:
            u.append(math.sqrt(sum_visits) / (1 + 0))
    u = normalization(u)
    x = normalization(x)
    x = exp_prob(-x, t=10)
    x = x + c * u

    return x


def ucb_eta(eta, visits, c=5):
    if len(list(visits.values())) == 0:
        sum_visits = 0
    else:
        sum_visits = np.sum(list(visits.values()))
    u = []
    for key, value in enumerate(eta):
        if str(key) in visits.keys():
            u.append(math.sqrt(sum_visits) / (1 + visits[str(key)]))
        else:
            u.append(math.sqrt(sum_visits) / (1 + 0))

    u = normalization(u)
    x = eta + c * u
    return x


if __name__ == '__main__':
    value = np.array(
        [0.25045454107215337, 0.24519156322771718, -0.41049298433888337, 0.7097154523661049, -0.45271746891368997]*6)
    norm = normalization(value)
    a = normalization([1, 1, 1, 0.5, 1])
    num = 30
    s = math.sqrt(num)
    b = [s]*30
    b[10]=s*10
    b[11]=s*20

    b = normalization(b)
    c = normalization([s/2,s,s,s/3,s])
    d = normalization([s/2,s/2,s,s/2,s])

    c_p = 5
    print(a,b,c,d)
    print(probability(norm))
    # print(probability(norm + 5 * a), probability(norm + 1 * a), probability(norm + 10 * a))
    # print(probability(norm + 5 * c), probability(norm + 1 * c), probability(norm + 10 * c))
    # print(probability(norm + 5 * d), probability(norm + 1 * d), probability(norm + 10 * d))

    print(probability(norm + 5 * b), probability(norm + 1 * b), probability(norm + 10 * b))