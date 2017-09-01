import tensorflow as tf
import numpy as np
from adler.tensorflow import cosine_decay, leaky_relu

sess = tf.InteractiveSession()


n, m = 2, 2

eps = 0.001
A = np.array([[1, 1],
              [1, 1 + eps]])

eta = 1e-10
gamma = eta ** 2 * np.eye(m)
gamma_inv = np.linalg.inv(gamma)

Ai = np.linalg.inv(A)
AiMAP = np.linalg.inv(A.T.dot(gamma_inv).dot(A) + np.eye(n)).dot(A.T).dot(gamma_inv)


with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, shape=[None, n])
    x_expected = tf.placeholder(tf.float32, shape=[None, n])
    y = tf.placeholder(tf.float32, shape=[None, m])

with tf.name_scope('neural_net'):
    l1 = tf.contrib.layers.fully_connected(y, 4,
                                           weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                           biases_initializer=None,
                                           activation_fn=tf.nn.relu)
    result = tf.contrib.layers.fully_connected(l1, m,
                                               activation_fn=None,
                                               weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                               biases_initializer=None)

loss = tf.reduce_mean((result - x_true)**2)

error = tf.reduce_mean((x_expected - x_true)**2)

with tf.name_scope('optimizer'):
    # Learning rate
    global_step = tf.train.create_global_step()
    maximum_steps = 100000
    starter_learning_rate = 1e-3
    learning_rate = cosine_decay(starter_learning_rate,
                                 global_step,
                                 maximum_steps,
                                 name='learning_rate')

    opt_func = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = opt_func.minimize(loss, global_step=global_step)


# Initialize all TF variables
sess.run(tf.global_variables_initializer())

from sklearn.decomposition import PCA

n_data = 1000
x_validate_arr = np.random.randn(n_data, n)
y_validate_arr = A.dot(x_validate_arr.T).T + eta * np.random.randn(n_data, m)
x_validate_expected_arr = AiMAP.dot(y_validate_arr.T).T

pca = PCA(n_components=n, whiten=True).fit(y_validate_arr)

# Train the network
for i in range(maximum_steps):
    x_arr = np.random.randn(n_data, n)
    y_arr = A.dot(x_arr.T).T + eta * np.random.randn(n_data, m)
    x_expected_arr = AiMAP.dot(y_arr.T).T

    _ = sess.run([optimizer, result, loss, error],
                 feed_dict={x_true: x_arr,
                            x_expected: x_expected_arr,
                            y: pca.transform(y_arr) + pca.mean_})

    if i % 10 == 0:
        result_result, loss_result, error_result = sess.run([result, loss, error],
                     feed_dict={x_true: x_validate_arr,
                                x_expected: x_validate_expected_arr,
                                y: pca.transform(y_validate_arr) + pca.mean_})

        print('iterate={}, loss={}, optimal loss={}'.format(i, loss_result, error_result))
