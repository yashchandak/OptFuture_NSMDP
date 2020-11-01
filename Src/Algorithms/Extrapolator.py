import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
import torch as torch

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

# mpl.style.use('seaborn')  # https://matplotlib.org/users/style_sheets.html

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



class WLS:
    def __init__(self, max_len, delta, basis_type='Linear', k=10):
        self.basis_type = basis_type
        self.k = k  # number of terms in Fourier series

        if self.basis_type == 'Fourier':
            weights = torch.from_numpy(np.arange(k).reshape([1, k]))             # 1xk
        elif self.basis_type == 'Linear':
            weights = torch.from_numpy(np.ones([1, 2]))             # 1xk
        elif self.basis_type == 'Poly':
            weights = torch.from_numpy(np.arange(k))             # 1xk
        else:
            return ValueError

        self.basis_weights = weights.type(torch.FloatTensor).requires_grad_(False)#.to(self.config.device)

    def get_basis(self, x):

        if self.basis_type == 'Fourier':
            basis = x * self.basis_weights  # Broadcast multiplication B*1 x 1*k => Bxk
            basis = torch.cos(basis * np.pi)

        elif self.basis_type == 'Linear':
            basis = x * self.basis_weights  # Broadcast multiplication B*1 x 1*2 => Bx2
            basis[:, 1] = 1                 # set the second column to be 1

        elif self.basis_type == 'Poly':
            basis = x ** self.basis_weights  # Broadcast power B*1 xx k => Bxk

        return basis


    def forward(self, x, y, weights, max_len, delta):
        const = 1.0 * (max_len + delta + int(0.05 * max_len))           # For normalizing to 0-1
        x = x.float().view(-1, 1) / const                               # Bx1
        x_pred = torch.from_numpy(np.arange(max_len, max_len + delta).reshape([-1, 1])/const).float()  # delta x1

        phi_x = self.get_basis(x)                                       # Bxk
        phi_x_pred = torch.mean(self.get_basis(x_pred), keepdim=True, dim=0)  # deltaxk -> 1xk


        # TODO: might want to add a small noise to avoid inversion of singular matrix
        diag = torch.diag(weights.view(-1))                             # BxB

        phi_xt = phi_x.transpose(1, 0)                                  # kxB
        inv = torch.inverse(phi_xt.mm(diag).mm(phi_x))                  # (kxB x BxB x Bxk)^{-1} -> kxk
        w = inv.mm(phi_xt).mm(diag).mm(y)                               # kxk x kxB x BxB x Bx1 -> kx1

        hat_y = phi_x_pred.mm(w)                                        # 1xk x kx1 -> 1x1

        return hat_y


class OLS:
    def __init__(self, max_len, delta, basis_type='Linear', k=5):
        self.basis_type = basis_type
        self.k = k  # number of terms in Fourier series
        self.update(max_len, delta)

    def update(self, max_len, delta):
        self.delta = delta
        self.max_len = max_len
        self.extra = int(0.05 * max_len)  # Fourier changes drastically near the edges, hence widen to never hit end-point.

        self.all_derivatives = np.zeros(max_len)
        self.cache_derivative()


    def get_basis(self, x):
        x = x / (self.max_len + self.delta + self.extra)        # For normalizing to 0-1
        if self.basis_type == 'Fourier':
            basis = np.arange(self.k)
            phi = np.outer(x, basis)  # Nxk
            phi = np.cos(phi * np.pi)

        elif self.basis_type == 'Linear':
            phi = np.ones((x.size, 2))
            phi[:, 0] = x

        elif self.basis_type == 'Poly':
            phi = np.ones((x.size, self.k+1))
            for idx in range(self.k+1):
                phi[:, idx] = x**idx
        else:
            return ValueError

        return phi

    def cache_derivative(self):
        x = np.arange(self.max_len)
        phi = self.get_basis(x)                         # Nxk
        temp1 = np.linalg.inv(np.dot(phi.T, phi))       # kxk
        temp2 = np.dot(temp1, phi.T)                    # kxk x kxN -> kxN

        # Derivative for the average future
        phi = self.get_basis(np.arange(self.max_len, self.max_len + self.delta))    # delta x k
        phi = np.mean(phi, axis=0, keepdims=True)                                       # 1xk

        self.all_derivatives = np.dot(phi, temp2).T     # 1xk x kxN -> 1xN -> Nx1


    def derivatives(self, ids):
        # Retrieve from cache for this future pos
        return self.all_derivatives[ids]                # Nx1 (B) -> Bx1

    def fit(self, x, y):
        phi = self.get_basis(x)

        self.temp1 = np.linalg.inv(np.dot(phi.T, phi))      # kxk
        self.temp2 = np.dot(self.temp1, phi.T)              # kxk x kxN -> kxN
        self.w = np.dot(self.temp2, y)                      # kxN x Nx1 -> kx1


    def predict(self, x):
        phi = self.get_basis(x)
        hat_y = np.dot(phi, self.w)                         # Nxk x kx1 -> Nx1

        return hat_y

    def forward(self, x):
        phi = self.get_basis(x)
        hat_y = np.dot(phi, self.w)                         # 1xk x kx1 -> Nx1
        grad_y = np.dot(phi, self.temp2)                    # 1xk x kxN  -> 1xN

        return hat_y, grad_y


def get_data(N = 100):
    x = np.arange(N)
    # y = x + 10*np.sin(x) + 3*np.random.randn(N) + 20
    # y = (x - 20)**2 + 100*np.random.randn(N) + 20
    y = x + 10*np.random.randn(N) + 20
    # y = -np.exp(x *0.1) + 10*np.random.randn(N) + 20
    # y = 10 + 100*np.random.randn(N) + 20
    return x, y


def comparison_plots():
    x, y = get_data(N=100)
    delta = 1

    # fn = Fourier(k=5, max=np.max(x))
    Constant = OLS(max_len=np.max(x), delta=delta, basis_type='Fourier', k=1)
    Linear = OLS(max_len=np.max(x), delta=delta, basis_type='Linear', k=5)
    Fourier = OLS(max_len=np.max(x), delta=delta, basis_type='Fourier', k=5)

    Fourier.fit(x[:-delta], y[:-delta])
    F_y = Fourier.predict(x)
    F_pred, F_grad = Fourier.forward(x[-5:])

    Constant.fit(x[:-delta], y[:-delta])
    C_y = Constant.predict(x)
    C_pred, C_grad = Constant.forward(x[-5:])

    Linear.fit(x[:-delta], y[:-delta])
    L_y = Linear.predict(x)
    L_pred, L_grad = Linear.forward(x[-5:])


    plt.figure()
    plt.title("Function")
    plt.plot(x, F_y)
    plt.plot(x, L_y)
    plt.plot(x, C_y)
    plt.plot(x, y)

    plt.figure()
    plt.title("Weights")
    plt.xlabel("Episode")
    plt.ylabel("Value")

    plt.plot(x[:-delta], F_grad[0], linewidth=2, label='Fourier')
    plt.plot(x[:-delta], L_grad[0], linewidth=2, label='Identity')
    plt.plot(x[:-delta], C_grad[0], linewidth=2, label='Constant')
    plt.plot(x[:-delta], np.exp(12000*(x[:-delta]/np.sum(x[:-delta]))**3) - 1, linewidth=2, label='Exponential')
    plt.plot(x[:-delta], np.zeros(x[:-delta].size), '--', linewidth=1, color='k')

    plt.legend()
    plt.show()

    print("Sum of gradients: ", np.sum(F_grad[0]))


def plot():
    x, y = get_data(N=100)
    delta = 1

    # fn = Fourier(k=5, max=np.max(x))
    # fn = OLS(max_len=np.max(x), delta=delta, basis_type='Linear', k=5)
    fn = OLS(max_len=np.max(x), delta=delta, basis_type='Fourier', k=5)
    # fn = OLS(max_len=np.max(x), delta=delta, basis_type='Poly', k=5)

    fn.fit(x[:-delta], y[:-delta])
    hat_y = fn.predict(x)
    pred, grad = fn.forward(x[-5:])

    plt.figure()
    plt.title("Function")
    plt.plot(x, y)
    plt.plot(x, hat_y)

    plt.figure()
    plt.title("Fourier Gradient")
    plt.xlabel("Time")
    plt.ylabel("Gradient")
    plt.plot(x[:-delta], grad[0])
    plt.plot(x[:-delta], np.zeros(x[:-delta].size), '--', linewidth=1, color='k')
    plt.show()

    print("Sum of gradients: ", np.sum(grad[0]))


if __name__ == "__main__":
    # plot()
    comparison_plots()