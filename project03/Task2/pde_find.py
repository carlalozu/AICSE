
"""Class to find differential equations based on data"""
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from finite_differences import FiniteDifference
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LassoCV

import seaborn as sns
sns.set()
plt.rcParams['image.cmap'] = 'plasma'


class PDE_FIND():
    """Differential equation for 2D data"""

    def __init__(self, name):
        self.data, self.vars = self.open_data_file(name)
        print(self.data.shape)
        self.unpack_data()
        self.classify = {}

    @staticmethod
    def open_data_file(name):
        """Read npz file and save to np"""
        data = np.load(f'data/{name}.npz')

        # create dict of variables and positions
        variables = {var: i for i, var in enumerate(data.files)}
        print(variables)

        arrays = []
        for var in variables:
            arrays.append(data[var])

        return np.stack(arrays), variables

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

    def create_list_of_possible_terms(self, classify, order=3, max_terms=100):
        """Create list of possible terms for the PDE
        order: maximum order of derivatives (1,2 or 3)
        """
        # list derivatives
        classify['derivatives'] = []
        for d in classify['dep']:
            for i in classify['indep']:
                # first derivative
                if i != 't':
                    classify['derivatives'].append(f'{d}_{i}')
                else:
                    continue
                if order > 1:
                    for j in classify['indep']:
                        # second derivative
                        if j != 't' and f'{d}_{j+i}' not in classify['derivatives']:
                            classify['derivatives'].append(f'{d}_{i}{j}')
                        else:
                            continue
                        if order > 2:
                            for k in classify['indep']:
                                # third derivative
                                if k != 't' and (f'{d}_{j+i+k}' not in classify['derivatives']
                                                 or f'{d}_{i+j+k}' not in classify['derivatives']):
                                    classify['derivatives'].append(
                                        f'{d}_{i}{j}{k}')
        # I know nested loops are not the best, but it works

        # initialize list of possible terms
        classify['terms'] = classify['dep'].copy() + \
            classify['derivatives'].copy()

        # add combinations of terms to the list of possible terms
        for i in range(len(classify['terms'])):
            if '_t' not in classify['terms'][i]:
                for j in range(i, len(classify['terms'])):
                    # at most 2 terms
                    if len(classify['terms'][j].split('*')) < 2:
                        if '_t' not in classify['terms'][j]:
                            classify['terms'].append(
                                f'{classify["terms"][i]}*{classify["terms"][j]}')
                    if len(classify['terms']) > max_terms:
                        # restrict to around max_terms number of terms,
                        # otherwise the computation will take too long
                        break

        # add multiplication of base vars
        for i in classify['dep']:
            for j in classify['dep']:
                for k in classify['dep']:
                    term = f'{i}*{j}*{k}'
                    sorted_term = '*'.join(sorted(term.split('*')))
                    if sorted_term not in classify['terms']:
                        classify['terms'].append(term)

        # add time derivatives
        for i in classify['dep']:
            classify['derivatives'].append(f'{i}_t')

        print(len(classify['terms']), 'terms')
        print(classify['terms'])

    def get_derivatives(self, classify, periodic=False):
        """Calculate partial derivatives using finite differences"""
        derivatives = {}
        for der in classify['derivatives']:
            v = der.split('_')[0]
            ind = der.split('_')[1]
            data = deepcopy(self.u[self.vars[v], :])
            if ind == 't' and periodic:
                temp_periodic = False
            else:
                temp_periodic = periodic
            for d in ind:
                fd = FiniteDifference(
                    axis=self.vars[d]-len(classify['dep']),
                    periodic=temp_periodic)
                data = fd(data, self.axis[d])
            derivatives[der] = data.reshape(1, *data.shape)

        print(len(derivatives), 'derivatives computed')
        print(derivatives.keys())
        return derivatives

    def create_feature_library(self, classify, derivatives):
        """Use derivatives and functions to create a library"""
        print(len(classify['terms']), 'terms')
        library = []
        for i, term in enumerate(classify['terms']):
            terms = term.split('*')
            components = []
            for term in terms:
                if len(term) == 1:
                    data = self.u[self.vars[term], :]
                    data = data.reshape(1, *data.shape)
                    components.append(data)
                else:
                    components.append(derivatives[term])
            library_object = np.prod(components, axis=0)

            if not i:
                library = library_object
            else:
                library = np.concatenate((library, library_object), axis=0)
            print('term', '*'.join(terms), 'done',
                  f"{i+1}/{len(classify['terms'])}")

        assert len(library) == len(classify['terms'])
        classify['library'] = library.squeeze()

    def plot(self, data, name, idx=0):
        """Plot u and u dot in two subplots"""
        var = name.split('_')[0]
        fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        fig.tight_layout(pad=2.0)
        fig.subplots_adjust(left=0.1)

        ax[0].pcolormesh(self.t, self.x, self.u[idx, :, :])
        ax[0].set_title(f'${var}(x, t)$')
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        fig.colorbar(ax[0].pcolormesh(self.t, self.x, self.u[idx, :, :]), ax=ax[0])

        # get derivative
        ax[1].pcolormesh(self.t, self.x, data)
        ax[1].set_title(f'${name}(x, t)$')
        ax[1].set_xlabel('$t$')
        fig.colorbar(ax[1].pcolormesh(self.t, self.x, data), ax=ax[1])

        return fig

    def plot_results(self, reference, data, var='u'):
        """Plot results and difference in two subplots"""
        fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        fig.tight_layout(pad=2.0)
        fig.subplots_adjust(left=0.1)

        ax[0].pcolormesh(self.t, self.x, data)
        ax[0].set_title(f'${var}_t'+'^{pred}(x, t)$')
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        fig.colorbar(ax[0].pcolormesh(self.t, self.x, data), ax=ax[0])

        # get derivative
        ax[1].pcolormesh(self.t, self.x, data-reference)
        ax[1].set_title(f'$\Delta ({var}_t$'+'$^{pred}-$'+f'${var}_t)(x, t)$')
        ax[1].set_xlabel('$t$')
        # add colorbar
        fig.colorbar(ax[1].pcolormesh(self.t, self.x, data-reference), ax=ax[1])

        print('MSE:', np.mean((data-reference)**2).item())
        return fig


    def solve(self, inputs, target, sparsity_threshold, **kwargs):
        """Solve the linear regression problem using Lasso regression"""

        # flatten but omit the first dimension
        inputs_ = inputs.reshape(inputs.shape[0], -1)
        target_ = target.reshape(-1)

        # turn to float23 due to error in LassoCV
        inputs_ = np.array(inputs_, dtype=np.float32)
        target_ = np.array(target_, dtype=np.float32)

        kwargs['random_state'] = 42

        # Fit the Lasso regression model
        model = LassoCV(**kwargs)
        model.fit(inputs_.T, target_)
        print('Best alpha:', model.alpha_)

        weights = np.array(model.coef_)
        # enforce sparsity
        weights[np.abs(weights) < sparsity_threshold] = 0

        return weights

    def test(self, library, weights, no_terms=0, name='u'):
        """Test the neural net"""
        components = library['library']
        names = library['terms']

        # create array on ones with the same shape as the components
        dim = len(components.shape)-1
        weights_ = weights.reshape(weights.shape[0], *[1]*dim)
        y_pred = np.sum(weights_ * components, axis=0)
        used = [f'{weight:.2}*{term}' for term,
                weight in zip(names, weights) if weight != 0]

        if not no_terms:
            no_terms = len(used)

        idx = np.argsort(-np.abs(weights))
        # retain largest terms given by no_terms
        idx = idx[:no_terms]
        used = [f'{weights[i]:.2}{names[i]}' for i in idx]

        print("Number of terms:", len(used))
        print('PDE:')
        print('\t' + ' + '.join(used) + f' = {name}_t')
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

    def subsample_data(self, derivatives, percent=0.6):
        """Subsample data to n points"""
        # Subsample is possible because we are also passing the derivatives to
        # the solver, used in case of too many data points
        subsample = np.random.choice(
            len(self.t), int(len(self.t) * percent), replace=False)
        self.u = self.u[:, :, :, subsample]
        for key in derivatives:
            derivatives[key] = derivatives[key][:, :, :, subsample]

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

    def plot(self, data, name, idx=0):
        """Plot var and var dot in two subplots"""

        # get dep var
        var = name.split('_')[0]
        fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        fig.tight_layout(pad=2.0)
        # add slight pad to the left
        fig.subplots_adjust(left=0.1)

        ax[0].pcolormesh(self.x, self.y, self.u[self.vars[var], :, :, idx])
        ax[0].set_title(f'${var}(x, t)$')
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        fig.colorbar(ax[0].pcolormesh(self.x, self.y, self.u[self.vars[var], :, :, idx]), ax=ax[0])

        # get derivative
        ax[1].pcolormesh(self.x, self.y, data[:, :, idx])
        ax[1].set_title(f'${name}(x, t)$')
        ax[1].set_xlabel('$t$')
        fig.colorbar(ax[1].pcolormesh(self.x, self.y, data[:, :, idx]), ax=ax[1])

        return fig

    def plot_results(self, reference, data, var='u'):
        """Plot results and difference in two subplots"""
        fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        fig.tight_layout(pad=2.0)
        fig.subplots_adjust(left=0.1)

        ax[0].pcolormesh(self.x, self.y, data)
        ax[0].set_title(f'${var}_t'+'^{pred}(x, t)$')
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        fig.colorbar(ax[0].pcolormesh(self.x, self.y, data), ax=ax[0])

        # get derivative
        ax[1].pcolormesh(self.x, self.y, data-reference)
        ax[1].set_title(f'$\Delta ({var}_t$'+'$^{pred}-$'+f'${var}_t)(x, t)$')
        ax[1].set_xlabel('$t$')
        # add colorbar
        fig.colorbar(ax[1].pcolormesh(self.x, self.y, data-reference), ax=ax[1])

        print('MSE:', np.mean((data-reference)**2).item())
        return fig
