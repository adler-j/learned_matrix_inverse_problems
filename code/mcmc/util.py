import numpy as np
import theano
import odl


class MovingAverage():
    def __init__(self, init=None):
        self.mean = init
        self.step = 1

    def __call__(self, update):
        if self.mean is None:
            self.mean = update
        else:
            delta = update - self.mean
            self.mean += delta / self.step

        self.step += 1

        return self.mean


class MovingStd():
    def __init__(self):
        self.mean = None
        self.M2 = None
        self.step = 1

    def __call__(self, update):
        if self.mean is None:
            self.mean = update.space.zero()
            self.M2 = update.space.zero()

        delta = update - self.mean
        self.mean += delta / self.step
        delta2 = update - self.mean
        self.M2 += delta * delta2

        self.step += 1

        if self.step < 2:
            return update.space.zero()
        else:
            return np.sqrt(self.M2 / (self.step - 1))


class ODLTheanoOp(theano.Op):
    __props__ = ()

    def __init__(self, operator):
        self.operator = operator

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        out_type = theano.tensor.TensorType(
            self.operator.range.dtype,
            [False] * len(self.operator.range.shape))
        return theano.Apply(self, [x], [out_type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = np.asarray(self.operator(x))

    def infer_shape(self, node, i0_shapes):
        return [self.operator.range.shape]

    def grad(self, inputs, output_grads):
        try:
            dom_weight = self.operator.domain.weighting.const
        except AttributeError:
            dom_weight = 1.0

        try:
            ran_weight = self.operator.range.weighting.const
        except AttributeError:
            ran_weight = 1.0

        scale = dom_weight / ran_weight
        return [ODLTheanoOp(scale * self.operator.adjoint)(output_grads[0])]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
