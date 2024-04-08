from pinn import Pinns

kwargs = {
    "alpha_f" : 0.005,
    "h_f" : 5,
    "T_hot" : 4,
    "T0" : 1,
    "T_cold" : 1,
}


def test_n_cycles():
    t0 = 0
    tf = 1
    n_int = 128
    n_sb = 64
    n_tb = 64

    pinn = Pinns(n_int, n_sb, n_tb, t0, tf, **kwargs)
    assert pinn.n_cycles == 1

def test_spatial_boundary_points():
    t0 = 0
    tf = 1
    n_int = 128
    n_sb = 64
    n_tb = 64

    pinn = Pinns(n_int, n_sb, n_tb, t0, tf, **kwargs)
    input_sb_, output_sb_ = pinn.add_spatial_boundary_points()

    # times two due to the two boundaries x0 and xL
    assert input_sb_.shape == (n_sb*2, 2)
    assert output_sb_.shape == (n_sb*2, 1)

def test_spatial_boundary_points_two_phases():
    t0 = 0
    tf = 2
    n_int = 128*2
    n_sb = 64*2
    n_tb = 64

    pinn = Pinns(n_int, n_sb, n_tb, t0, tf, **kwargs)
    input_sb_, output_sb_ = pinn.add_spatial_boundary_points()

    # times two due to the two boundaries x0 and xL
    assert input_sb_.shape == (n_sb*2, 2)
    assert output_sb_.shape == (n_sb*2, 1)

def test_fit():
    t0 = 0
    tf = 1
    n_int = 128
    n_sb = 64
    n_tb = 64

    pinn = Pinns(n_int, n_sb, n_tb, t0, tf, **kwargs)
    pinn.fit(1, True, max_iter=10)
