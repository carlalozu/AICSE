
"""Class to find differential equations based on data"""
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from matplotlib.animation import FuncAnimation
from copy import deepcopy
import torch
from sklearn.linear_model import Ridge

class PDE_FIND():
    """Differential equation for 2D data"""

    def __init__(self, name):
        self.data, self.vars = self.open_data_file(name)
        print(self.data.shape)
        self.unpack_data()
        self.classify = {}

    @staticmethod
    def open_data_file(name):
        """Read npz file and save to torch"""
        data = np.load(f'data/{name}.npz')

        # create dict of variables and positions
        variables = {var: i for i, var in enumerate(data.files)}
        print(variables)

        arrays = []
        for var in variables:
            # convert to torch tensor
            arrays.append(torch.tensor(data[var]))

        return torch.stack(arrays), variables

    def unpack_data(self):
        """Get principal variables and components"""
        self.x = self.data[1, :, 0]
        self.t = self.data[2, 0, :]

        self.u = self.data[0, :].reshape(1, len(self.x), len(self.t))

        axis = {}
        axis['t'] = (self.t[1] - self.t[0]).item()
        axis['x'] = (self.x[1] - self.x[0]).item()

        self.axis = axis

    def set_classify(self, classify):
        """Set classify dictionary"""
        self.classify = deepcopy(classify)

    def create_list_of_possible_terms(self, classify):
        """Create list of possible terms for the PDE"""
        # list derivatives
        classify['derivatives'] = []
        for d in classify['dep']:
            for i in classify['indep']:
                # first derivative
                classify['derivatives'].append(f'{d}_{i}')
                for j in classify['indep']:
                    # second derivative
                    if 't' not in i+j and f'{d}_{j+i}' not in classify['derivatives']:
                        classify['derivatives'].append(f'{d}_{i}{j}')
                    for k in classify['indep']:
                        # third derivative
                        if 't' not in i+j+k and f'{d}_{j+i+k}' not in classify['derivatives']:
                            classify['derivatives'].append(f'{d}_{i}{j}{k}')


        # initialize list of possible terms
        classify['terms'] = classify['dep'].copy() + classify['derivatives'].copy()
        classify['terms'].remove('u_t')

        # add combinations of terms to the list of possible terms
        for i in range(len(classify['terms'])):
            if '_t' not in classify['terms'][i]:
                for j in range(i, len(classify['terms'])):
                    # at most 2 terms
                    if len(classify['terms'][j].split('*')) < 2:
                        if '_t' not in classify['terms'][j]:
                            classify['terms'].append(
                                f'{classify["terms"][i]}*{classify["terms"][j]}')

    def get_derivatives(self, classify):
        """Calculate partial derivatives using finite differences"""
        derivatives = {}
        for der in classify['derivatives']:
            v = der.split('_')[0]
            ind = der.split('_')[1]
            data = deepcopy(self.u[self.vars[v], :]).numpy()
            for d in ind:
                fd = ps.FiniteDifference(
                    axis=self.vars[d]-len(classify['dep']))
                data = fd(data, self.axis[d])
            derivatives[der] = torch.tensor(data.reshape(1, *data.shape))

        return derivatives

    def create_feature_library(self, classify, derivatives):
        """Use derivatives and functions to create a library"""
        print(len(classify['terms']), 'terms')
        library = []
        for term in classify['terms']:
            components = term.split('*')
            library_object = []
            for component in components:
                if len(component) == 1:
                    library_object.append(self.u)
                else:
                    library_object.append(derivatives[component])

            # TODO: change to torch not numpy
            library.append(torch.tensor(np.prod(library_object, axis=0)))

        assert len(library) == len(classify['terms'])
        classify['library'] = torch.stack(library, dim=1).squeeze()

    def plot(self, data, name):
        """Plot u and u dot in two subplots"""

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].pcolormesh(self.t, self.x, self.u[0, :, :])
        ax[0].set_title(r'$u(x, t)$')
        ax[0].set_xlabel(r'$t$')
        ax[0].set_ylabel(r'$x$')

        # get derivative
        ax[1].pcolormesh(self.t, self.x, data)
        ax[1].set_title(f'${name}(x, t)$')
        ax[1].set_xlabel(r'$t$')
        ax[1].set_ylabel(r'$x$')

        fig.show()
        return fig

    def solve(self, library, y_train):
        """Solve the linear regression problem using Ridge regression"""
        inputs = library.flatten(start_dim=1)
        target = y_train.flatten()

        # Fit the Ridge regression model
        model = Ridge(alpha=1.0, fit_intercept=False, tol=1e-6, solver='cholesky')
        model.fit(inputs.T, target)

        return torch.tensor(model.coef_)

    def test(self, library, weights, no_terms=0):
        """Test the neural net"""
        components = library['library']
        names = library['terms']
        dim = len(components)
        y_pred = torch.sum(weights.view(dim, 1, 1) * components, dim=0)
        used = [f'{weight:.2e}*{term}' for term,
                weight in zip(names, weights.numpy()) if weight != 0]
        
        if no_terms:
            # retain largest terms given by no_terms
            idx = torch.argsort(torch.abs(weights), descending=True)
            idx = idx[:no_terms]
            used = [f'{weights[i]:.2e}*{names[i]}' for i in idx]

        print(' + '.join(used)+' = u_t')
        return y_pred.reshape(1, *y_pred.shape)


class PDE_FIND_3D(PDE_FIND):
    """Differential equation finder for 3D data"""

    def unpack_data(self):
        """Get principal variables and components"""
        self.x = self.data[2, :, 0, 0]
        self.y = self.data[3, 0, :, 0]
        self.t = self.data[4, 0, 0, :]

        self.u = self.data[0:2, :]  # contains u and v

        axis = {}
        axis['t'] = (self.t[1] - self.t[0]).item()
        axis['x'] = (self.x[1] - self.x[0]).item()
        axis['y'] = (self.y[1] - self.y[0]).item()

        self.axis = axis

    # def subsample_data(self, percent=0.6):
    #     """Subsample data to n points"""
    #     # Resample is possible because we are also passing the derivative to the
    #     # solver
    #     train = np.random.choice(
    #         len(self.t), int(len(self.t) * percent), replace=False)
    #     u_ = self.u[:, :, :, train]
    #     u_ = u_.reshape(len(self.x), len(self.y), len(train), 2)
    #     u_t_ = self.u_t[:, :, :, train]
    #     u_t_ = u_t_.reshape(len(self.x), len(self.y), len(train), 2)
    #     return u_, u_t_

    def animate(self, idx):
        """Create animation of the data for a given index over time"""
        fig, ax = plt.subplots(figsize=(3, 3))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Initialize the plot with the first frame
        def init():
            im = ax.imshow(self.data[idx, :, :, 0], animated=True)
            return [im]

        # Update function for each frame
        def update(frame):
            im = ax.imshow(self.data[idx, :, :, frame], animated=True)
            return [im]

        # Create the animation
        anim = FuncAnimation(fig, update, frames=self.data.shape[-1],
                             init_func=init, blit=True)

        return anim
