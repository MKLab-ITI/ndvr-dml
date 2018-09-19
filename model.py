# Copyright 2018 Giorgos Kordopatis-Zilos. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Tensorflow implementation of the DNN network used for Deep Metric Learning.
"""

import os
import tensorflow as tf


class DNN(object):

    def __init__(self,
                 input_dimensions,
                 hidden_layer_sizes,
                 model_path,
                 load_model=False,
                 trainable=True,
                 learning_rate=1e-5,
                 weight_decay=5e-3,
                 gamma=0.1):
        """
          Class initializer.

          Args:
            input_dimensions: dimension of the input vectors
            hidden_layer_sizes: number of neurons of the DNN layers
            model_path: path to store the trained model
            load_model: load of the model weight from the model_path
            trainable: indicator of whether it is training or evaluation phase
            learning_rate: learning rate that weights are updated
            weight_decay: regularization parameter for weight decay
            gamma: margin parameter between positive-query and negative-query distance
        """
        self.trainable = trainable
        self.path = os.path.join(model_path, 'model')

        self.input = tf.placeholder(tf.float32, shape=(None, input_dimensions), name='input')
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay) if trainable else None
        if load_model:
            self.output = self.load_model()
        else:
            self.output = self.build(hidden_layer_sizes)

        self.saver = tf.train.Saver()
        if trainable:
            self.global_step = 1
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

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        init = tf.global_variables_initializer()
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        if trainable:
            self.summary_writer = tf.summary.FileWriter(model_path, self.sess.graph)

    def build(self, hidden_layer_sizes):
        """
          Function that builds the DNN model.

          Args:
            hidden_layer_sizes: number of neurons of the DNN layers
            trainable: indicator of whether it is training or evaluation phase

          Returns:
            net: output tensor of the constructed network
        """
        net = self.input
        for M in hidden_layer_sizes:
            net = tf.contrib.layers.fully_connected(net, M,
                        activation_fn=tf.nn.tanh,
                        weights_regularizer=self.regularizer,
                        biases_regularizer=self.regularizer,
                        trainable=self.trainable)
        with tf.name_scope('embeddings'):
            net = tf.nn.l2_normalize(net, 1, 1e-15)
            tf.summary.histogram('embeddings', net)
        return net

    def load_model(self):
        """
          Function that loads the weight of DNN layers from the saved model.
        """
        previous_sizes = [size[1] for _, size in
                          tf.contrib.framework.list_variables(self.path) if len(size) == 2]
        net = self.build(previous_sizes)

        previous_variables = [var_name for var_name, _
                              in tf.contrib.framework.list_variables(self.path)]
        restore_map = {variable.op.name: variable for variable in tf.global_variables()
                       if variable.op.name in previous_variables}
        tf.contrib.framework.init_from_checkpoint(self.path, restore_map)
        return net

    def euclidean_distance(self, x, y):
        """
          Euclidean distance calculation between each sample N of two matrices (NxM).

          Args:
            x: first feature matrix (NxM)
            y: second feature matrix (NxM)

          Returns:
            their euclidean distance in sample N dimension (axis 1)
        """
        with tf.name_scope('euclidean_distance'):
            return tf.reduce_sum(tf.square(tf.subtract(x, y)), 1)

    def triplet_loss(self, anchor, positive, negative, gamma=0.1):
        """
          Triplet loss calculation.

          Args:
            anchor: anchor feature matrix (NxM)
            positive: positive feature matrix (NxM)
            negative: negative feature matrix (NxM)
            gamma: margin parameter

          Returns:
            loss: total triplet loss
            error: number of triplets with positive loss
        """
        with tf.name_scope('triplet_loss'):
            pos_dist = self.euclidean_distance(anchor, positive)
            neg_dist = self.euclidean_distance(anchor, negative)
            loss = tf.maximum(0., pos_dist - neg_dist + gamma)
            error = tf.count_nonzero(loss, dtype=tf.float32) / \
                    tf.cast(tf.shape(anchor)[0], tf.float32) * tf.constant(100.0)
            loss = tf.reduce_mean(loss)
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('error', error)
            return loss, error

    def save(self):
        """
          Function that saves the DNN model in the provided directory.
        """
        print 'save model...'
        return self.saver.save(self.sess, self.path)

    def train(self, X):
        """
          Training of the network with the provided triplets.

          Args:
            X: input feature matrix (3*NxM)

          Returns:
            train: training argument
            loss: total triplet loss
            cost: total cost
            error: number of triplets with positive loss
        """
        return self.sess.run(self.train_op, feed_dict={self.input: X})

    def test(self, X):
        """
          Test of the network with the provided triplets.

          Args:
            X: input feature matrix (3*NxM)

          Returns:
            loss: total triplet loss
            cost: total cost
            error: number of triplets with positive loss
        """
        summary, loss, cost, error = self.sess.run(self.test_op, feed_dict={self.input: X})
        self.summary_writer.add_summary(summary, self.global_step)
        self.global_step += 1
        return loss, cost, error

    def embeddings(self, X):
        """
          Extraction of the feature embeddings of the input vectors.
          Args:
            X: input feature matrix (NxM)

          Returns:
            embeddings: embedding matrix (NxM)
        """
        return self.sess.run(self.output, feed_dict={self.input: X})