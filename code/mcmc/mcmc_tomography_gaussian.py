import pymc3 as pm
import numpy as np
import odl
import tqdm
from util import ODLTheanoOp, MovingAverage

# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')

geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)
operator = odl.tomo.RayTransform(space, geometry)

# Create sinogram of forward projected phantom with noise
true_phantom = odl.phantom.shepp_logan(space, modified=True)
data = operator(true_phantom)
sigma = np.mean(np.abs(data)) * 0.05
noisy_data = data + odl.phantom.white_noise(operator.range) * sigma
observed = noisy_data.asarray()

callback = (odl.solvers.CallbackShow('current', step=1) &
            odl.solvers.CallbackShow('mean', clim=[0.1, 0.4], step=1) * MovingAverage())

with pm.Model() as model:
    phantom = pm.Normal('phantom', mu=0, sd=0.1,
                        shape=operator.domain.shape)

    mu = ODLTheanoOp(operator)(phantom)
    obs = pm.Normal('obs', mu=mu, sd=sigma,
                    observed=observed)

    start = {'phantom': np.asarray(space.zero())}
    step = pm.NUTS(early_max_treedepth=3, max_treedepth=3)
    for trace in tqdm.tqdm(pm.iter_sample(10**5, step, start=start)):
        callback(space.element(trace['phantom', -1:]))
