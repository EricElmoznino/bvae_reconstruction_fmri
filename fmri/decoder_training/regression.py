import random
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch import optim

random.seed(27)


def cv_regression(x, y, n_splits, l2=0.0, sgd=False):
    # Get cross-validated mean test set correlation
    cv_rs = {'train': [], 'test': []}
    kf = KFold(n_splits)
    for train_idx, test_idx in kf.split(x, y):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if sgd:
            _, r_train, r_test = regression_sgd(x_train, y_train, x_test, y_test, l2=l2)
        else:
            _, r_train, r_test = regression_analytic(x_train, y_train, x_test, y_test, l2=l2)
        cv_rs['train'].append(r_train)
        cv_rs['test'].append(r_test)

    # Train on all of the data
    if sgd:
        weights = regression_sgd(x, y, None, None, l2=l2, validate=False)
    else:
        weights = regression_analytic(x, y, None, None, l2=l2, validate=False)

    return weights, cv_rs


def regression_analytic(x_train, y_train, x_test, y_test, l2=0.0, validate=True):
    regr = Ridge(alpha=l2, fit_intercept=False)
    regr.fit(x_train, y_train)
    weights = regr.coef_
    if validate:
        y_pred_train = regr.predict(x_train)
        y_pred_test = regr.predict(x_test)
        r_train = correlation(y_pred_train, y_train)
        r_test = correlation(y_pred_test, y_test)
        return weights, r_train, r_test
    else:
        return weights


def regression_sgd(x_train, y_train, x_test, y_test, l2=0.0, validate=True,
                   n_epochs=1000, batch_size=32, lr=1e-3):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x_train = torch.from_numpy(x_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    if validate:
        x_test = torch.from_numpy(x_test).to(device)
        y_test = torch.from_numpy(y_test).to(device)
    model = nn.Linear(x_train.size(1), y_train.size(1), bias=False).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2)

    pbar = tqdm(range(n_epochs))
    indices = list(range(x_train.size(0)))
    for _ in pbar:
        random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            x_batch, y_batch = x_train[batch_indices], y_train[batch_indices]
            y_pred = model(x_batch)
            loss = loss_func(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if validate:
            with torch.no_grad():
                y_pred = model(x_test)
            loss = loss_func(y_pred, y_test).item()
            r = correlation(y_pred, y_test).mean().cpu().item()
            pbar.set_postfix({'mse': loss, 'r': r})

    weights = model.weight.data.numpy()
    if validate:
        with torch.no_grad():
            y_pred_train = model(x_train)
            y_pred_test = model(x_test)
            r_train = correlation(y_pred_train, y_train).cpu().numpy()
            r_test = correlation(y_pred_test, y_test).cpu().numpy()
        return weights, r_train, r_test
    else:
        return weights


def correlation(a, b):
    zs = lambda v: (v - v.mean(0)) / v.std(0)
    r = (zs(a) * zs(b)).mean(0)
    return r
