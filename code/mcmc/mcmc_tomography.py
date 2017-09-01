import pymc3 as pm
import numpy as np
import theano
import theano.tensor as t
import odl
import tqdm
from util import MovingAverage, MovingStd, ODLTheanoOp
np.random.seed(0)

# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float64')

geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)
operator = odl.tomo.RayTransform(space, geometry)

# Create sinogram of forward projected phantom with noise
true_phantom = odl.phantom.shepp_logan(space, modified=True)
data = operator(true_phantom)
sigma = np.mean(np.abs(data)) * 0.05
noisy_data = data + odl.phantom.white_noise(operator.range) * sigma
observed = noisy_data.asarray()

ellipses_true = odl.phantom.shepp_logan_ellipsoids(2, True)
rand_ellipses = np.zeros([10, 6])
ellipses = np.r_[ellipses_true, rand_ellipses]

callback = (odl.solvers.CallbackShow('current', step=1) &
            odl.solvers.CallbackShow('mean', step=1) * MovingAverage() &
            odl.solvers.CallbackShow('std', step=1) * MovingStd()) * space.element

@theano.compile.ops.as_op(itypes=[t.dvector, t.dvector, t.dvector, t.dvector, t.dvector, t.dvector],
                          otypes=[t.dmatrix])
def ellipse_phantom(scale, axis_0, axis_1, x_0, y_0, theta):
    ellipses = np.c_[scale, np.abs(axis_0), np.abs(axis_1), x_0, y_0, theta]
    phantom = odl.phantom.ellipsoid_phantom(space, ellipses)
    return phantom

with pm.Model() as model:
    n = len(ellipses)
    scale = pm.Normal('scale', mu=0, sd=0.2, shape=n)
    axis_0 = pm.Normal('axis_0', mu=0, sd=0.2, shape=n)
    axis_1 = pm.Normal('axis_1', mu=0, sd=0.2, shape=n)
    x_0 = pm.Normal('x_0', mu=0, sd=0.5, shape=n)
    y_0 = pm.Normal('y_0', mu=0, sd=0.5, shape=n)
    theta = pm.Uniform('theta', - np.pi, np.pi, shape=n)

    start_vals = list(ellipses.T)
    start = {'scale': start_vals[0],
             'axis_0': start_vals[1],
             'axis_1': start_vals[2],
             'x_0': start_vals[3],
             'y_0': start_vals[4],
             'theta': start_vals[5]}

    phantom = ellipse_phantom(scale, axis_0, axis_1, x_0, y_0, theta)

    mu = ODLTheanoOp(operator)(phantom)

    obs = pm.Normal('obs', mu=mu, sd=sigma, observed=observed)

    # start = pm.find_MAP(start=start)

    step = pm.Slice()
    for trace in tqdm.tqdm(pm.iter_sample(10**6, step, start=start)):
        scale_val = trace['scale', -1:].squeeze()
        axis_0_val = trace['axis_0', -1:].squeeze()
        axis_1_val = trace['axis_1', -1:].squeeze()
        x_0_val = trace['x_0', -1:].squeeze()
        y_0_val = trace['y_0', -1:].squeeze()
        theta_val = trace['theta', -1:].squeeze()

        ellipses = np.c_[scale_val, np.abs(axis_0_val), np.abs(axis_1_val), x_0_val, y_0_val, theta_val]
        phantom_val = odl.phantom.ellipsoid_phantom(space, ellipses)

        callback(phantom_val)