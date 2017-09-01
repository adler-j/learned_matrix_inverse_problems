MCMC
====

In order to compute ![equation](https://latex.codecogs.com/svg.latex?\\mathbb{E}(x|y)) we need to use [Markov Chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo).

This folder contains code for computing the conditional expectation for the computed tomography inverse problem given two types of priors:

* **Continuous priors**: Here we have a "pixelwise" prior, for example pointwise gaussian or a TV style prior.
* **Ellipse priors**: Here the prior is that the images are given as the sum of 20 randomly created ellipses.

Dependencies
============
The code is currently based on the latest version of [ODL](https://github.com/odlgroup/odl/pull/972) and the utility library [adler](https://github.com/adler-j/adler). In order to run the MC code, pymc3 is required, which in turn requires theano.

In short, everything can be installed from the latest sources via:

```bash
pip install https://github.com/odlgroup/odl/archive/master.zip https://github.com/adler-j/adler/archive/master.zip https://github.com/Theano/Theano/archive/master.zip https://github.com/pymc-devs/pymc3/archive/master.zip
```