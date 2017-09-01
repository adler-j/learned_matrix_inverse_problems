import pymc3 as pm
import numpy as np
import odl
import tqdm
from util import ODLTheanoOp, MovingAverage, MovingStd
np.random.seed(0)

# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float64')

geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)
operator = odl.tomo.RayTransform(space, geometry)
fbp_op = odl.tomo.fbp_op(operator)

# Create sinogram of forward projected phantom with noise
true_phantom = odl.phantom.shepp_logan(space, modified=True)
data = operator(true_phantom)
sigma = np.mean(np.abs(data)) * 0.05
noisy_data = data + odl.phantom.white_noise(operator.range) * sigma
observed = noisy_data.asarray()
fbp_recon = fbp_op(noisy_data).asarray()

# Wavelet
W = odl.Gradient(space, pad_mode='periodic')

callback = (odl.solvers.CallbackShow('current', step=10) &
            odl.solvers.CallbackShow('mean', step=10) * MovingAverage() &
            odl.solvers.CallbackShow('std', step=10) * MovingStd()) * space.element

with pm.Model() as model:
    phantom = pm.Normal('phantom', mu=fbp_recon, sd=1.0,
                        shape=space.shape)

    mu_reg = ODLTheanoOp(W)(phantom)
    regularizer = pm.Laplace('regularizer', mu=mu_reg, b=0.05,
                             observed=0)

    mu = ODLTheanoOp(operator)(phantom)
    obs = pm.Normal('obs', mu=mu, sd=sigma,
                    observed=observed)

    # start = pm.find_MAP()
    step = pm.NUTS(early_max_treedepth=3, max_treedepth=3)
    start = {'phantom': fbp_recon}
    for trace in tqdm.tqdm(pm.iter_sample(10**5, step, start=start)):
        callback(trace['phantom', -1:][0])
