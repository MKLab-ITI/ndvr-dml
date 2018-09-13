import os
import tensorflow as tf

class DNN(object):

    def __init__(self,
                 input_dimentions,
                 hidden_layer_sizes,
                 model_path,
                 load_model=False,
                 trainable=True,
                 learning_rate=1e-5,
                 weight_decay=5e-3,
                 gamma=0.1):
        self.input = tf.placeholder(tf.float32, shape=(None, input_dimentions), name='input')

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
        self.output = self.build(hidden_layer_sizes, trainable)
        self.path = os.path.join(model_path, 'model')
        self.saver = tf.train.Saver()
        self.counter = 1

        if trainable:
            with tf.name_scope('training'):
                anchor, positive, negative = \
                    tf.unstack(tf.reshape(self.output, [-1, 3, self.output.get_shape().as_list()[1]]), 3, 1)
                loss, error = self.triplet_loss(anchor, positive, negative, gamma)

                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
                with tf.name_scope('cost'):
                    cost = loss + reg_term
                    tf.summary.scalar('cost', cost)

                train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
                self.train_op = [train, loss, cost, error]

            summary = tf.summary.merge_all()
            self.test_op = [summary, loss, cost, error]

        if load_model:
            self.load_model()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        init = tf.global_variables_initializer()

        self.sess = tf.Session(config=config)
        self.sess.run(init)

        if trainable:
            self.summary_writer = tf.summary.FileWriter(model_path, self.sess.graph)

    def build(self, hidden_layer_sizes, trainable):
        net = self.input
        for M in hidden_layer_sizes:
            net = tf.contrib.layers.fully_connected(net, M,
                        activation_fn=tf.nn.tanh,
                        weights_regularizer=self.regularizer,
                        biases_regularizer=self.regularizer,
                        trainable=trainable)
        with tf.name_scope('embeddings'):
            net = tf.nn.l2_normalize(net, 1, 1e-15)
            tf.summary.histogram('embeddings', net)
        return net

    def load_model(self):
        print 'load model...'
        previous_variables = [var_name for var_name, _
                              in tf.contrib.framework.list_variables(self.path)]
        restore_map = {variable.op.name: variable for variable in tf.global_variables()
                       if variable.op.name in previous_variables}
        tf.contrib.framework.init_from_checkpoint(self.path, restore_map)

    def euclidean_distance(self, x, y):
        with tf.name_scope('euclidean_distance'):
            return tf.reduce_sum(tf.square(tf.subtract(x, y)), 1)

    def euclidean_loss(self, anchor, positive, negative, alpha):
        with tf.name_scope('euclidean_loss'):
            pos_dist = self.euclidean_distance(anchor, positive)
            neg_dist = self.euclidean_distance(anchor, negative)
            return tf.maximum(0., pos_dist - neg_dist + alpha)

    def triplet_loss(self, anchor, positive, negative, alpha=0.1):
        with tf.name_scope('triplet_loss'):
            basic_loss = self.euclidean_loss(anchor, positive, negative, alpha)
            loss = tf.reduce_mean(basic_loss)
        error = tf.count_nonzero(basic_loss, dtype=tf.float32) / \
                tf.cast(tf.shape(anchor)[0], tf.float32) * tf.constant(100.0)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('error', error)
        return loss, error

    def save(self):
        print 'save model...'
        return self.saver.save(self.sess, self.path)

    def train(self, X):
        return self.sess.run(self.train_op, feed_dict={self.input: X})

    def test(self, X):
        summary, loss, cost, error = self.sess.run(self.test_op, feed_dict={self.input: X})
        self.summary_writer.add_summary(summary, self.counter)
        self.counter += 1
        return loss, cost, error

    def embeddings(self, X):
        return self.sess.run(self.output, feed_dict={self.input: X})