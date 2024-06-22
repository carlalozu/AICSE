
"""Class to find differential equations based on data"""
import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from matplotlib.animation import FuncAnimation

# Ignore matplotlib deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class DifferentialEquation():
    """Try to find the differential equation based on data"""

    def __init__(self, name):
        self.data, self.vars = self.open_data_file(name)
        print(self.data.shape)
        self.unpack_data()
        self.get_derivative()

    @staticmethod
    def open_data_file(name):
        """Read npz file and save to torch"""
        data = np.load(f'data/{name}.npz')

        arrays = []
        for i, var in enumerate(data.files):
            print(f'{i}: {var}')
            arrays.append(data[var])

        return np.stack(arrays), data.files

    def unpack_data(self):
        """Get principal variables and components"""
        self.x = self.data[1, :, 0]
        self.t = self.data[2, 0, :]

        self.u = self.data[0, :]

        self.dt = (self.t[1] - self.t[0]).item()
        self.dx = (self.x[1] - self.x[0]).item()

    def get_derivative(self):
        """Calculate derivative using finite differences"""
        fd = ps.FiniteDifference(axis=1)
        self.u_t = fd._differentiate(self.u, self.dt)

    def perform_computations(self):
        """Find differential equations"""
        # Define PDE library that is quadratic in u,
        # and second-order in spatial derivatives of u.
        library_functions = [lambda x: x, lambda x: x * x]
        library_function_names = [lambda x: x, lambda x: x + x]
        pde_lib = ps.PDELibrary(
            library_functions=library_functions,
            function_names=library_function_names,
            derivative_order=3,
            spatial_grid=self.x,
            is_uniform=True)

        # Fit the model with different optimizers.
        # Using normalize_columns = True to improve performance.
        print('STLSQ model: ')
        optimizer = ps.STLSQ(
            threshold=2,
            alpha=1e-5,
            normalize_columns=True
        )
        model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
        u_ = self.u.reshape(len(self.x), len(self.t), 1)
        model.fit(u_, t=self.dt)
        model.print()

    def plot(self):
        """Plot u and u dot in two subplots"""

        fig, ax = plt.subplots(1, 2)

        ax[0].pcolormesh(self.t, self.x, self.u)
        ax[0].set_title('u')
        ax[0].set_xlabel('t', fontsize=16)
        ax[0].set_ylabel('x', fontsize=16)

        # get derivative
        ax[1].pcolormesh(self.t, self.x, self.u_t)
        ax[1].set_title('u_t')
        ax[1].set_xlabel('t', fontsize=16)
        ax[1].set_ylabel('x', fontsize=16)

        fig.show()
        return fig


class DifEq3(DifferentialEquation):

    def __init__(self, name):
        pass

    @staticmethod
    def animate(data, idx):
        """Create animation of the data for a given index over time"""
        fig, ax = plt.subplots(figsize=(3, 3))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Initialize the plot with the first frame
        def init():
            im = ax.imshow(data[idx, :, :, 0], animated=True)
            return [im]

        # Update function for each frame
        def update(frame):
            im = ax.imshow(data[idx, :, :, frame], animated=True)
            return [im]

        # Create the animation
        anim = FuncAnimation(fig, update, frames=data.shape[-1],
                             init_func=init, blit=True)

        return anim
