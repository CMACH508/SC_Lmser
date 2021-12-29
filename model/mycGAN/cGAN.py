from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model.mycGAN.net import Generator, Discriminator2 as Discriminator
from util.image_pool import ImagePool
import model.mycGAN.ops as ops


class Model(object):

    def __init__(self, hps):
        self.hps = hps
        self.config_model()
        self.build_model("fss")

    def config_model(self):
        self.input_photo = tf.placeholder(
            dtype=tf.float32,
            shape=[self.hps.batch_size, self.hps.out_size[0], self.hps.out_size[1], 3])
        self.input_sketch = tf.placeholder(
            dtype=tf.float32,
            shape=[self.hps.batch_size, self.hps.out_size[0], self.hps.out_size[1], 1])

        # Normalizing image
        # [N, H, W, C], [-1, 1]
        self.input_x = self.input_photo / 127.5 - 1
        self.input_y = self.input_sketch / 127.5 - 1

        self.G = Generator(out_channels=1, ch=64, use_bias=False, scope="G", training=self.hps.is_training)
        self.F = Generator(out_channels=3, ch=64, use_bias=False, scope="F", training=self.hps.is_training)
        if self.hps.is_training:
            self.start_decay_steps = self.hps.start_decay_epochs * self.hps.steps_per_epoch
            self.decay_steps = self.hps.decay_epochs * self.hps.steps_per_epoch
            self.Dx = Discriminator(ndf=64, n_layers=3, use_bias=False, training=True, scope="Dx")
            self.Dy = Discriminator(ndf=64, n_layers=3, use_bias=False, training=True, scope="Dy")
            self.Fake_X_Pool = ImagePool(self.hps.pool_size)
            self.Fake_Y_Pool = ImagePool(self.hps.pool_size)

    def build_model(self, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.build_model_basic()
            if self.hps.is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.build_model_extra()
                self.build_losses()
                self.optimize_model()

    def build_model_basic(self):
        self.fake_y, _ = self.G(self.input_x)

    def build_model_extra(self):
        self.fake_x, _ = self.F(self.input_y)
        self.cyclic_x, _ = self.F(self.fake_y)
        self.fake_y2, _ = self.G(self.cyclic_x)

        self.real_xy = tf.concat([self.input_x, self.input_y], 3)
        self.fake_xy1 = tf.concat([self.input_x, self.fake_y], 3)
        self.fake_xy2 = tf.concat([self.input_x, self.fake_y2], 3)

        self.pool_xy1 = self.Fake_Y_Pool.query(self.fake_xy1)
        self.pred_real_xy = self.Dy(self.real_xy)
        self.pred_pool_xy1 = self.Dy(self.pool_xy1)
        self.pred_fake_xy1 = self.Dy(self.fake_xy1)

        self.pool_xy2 = self.Fake_Y_Pool.query(self.fake_xy2)
        self.pred_pool_xy2 = self.Dy(self.pool_xy2)
        self.pred_fake_xy2 = self.Dy(self.fake_xy2)

        self.real_yx = tf.concat([self.input_y, self.input_x], 3)
        self.fake_yx = tf.concat([self.input_y, self.fake_x], 3)
        self.pool_yx = self.Fake_X_Pool.query(self.fake_yx)
        self.pred_real_yx = self.Dx(self.real_yx)
        self.pred_fake_yx = self.Dx(self.fake_yx)
        self.pred_pool_yx = self.Dx(self.pool_yx)

    def build_losses(self):
        self.loss_Dy1 = ops.discriminator_loss(self.hps.gan_type, self.pred_real_xy, self.pred_pool_xy1)
        self.loss_G1 = self.hps.l1_lambda * tf.reduce_mean(tf.abs(self.fake_y - self.input_y)) + \
                      ops.generator_loss(self.hps.gan_type, self.pred_fake_xy1)

        self.loss_Dy2 = ops.discriminator_loss(self.hps.gan_type, self.pred_real_xy, self.pred_pool_xy2)
        self.loss_G2 = self.hps.l1_lambda * tf.reduce_mean(tf.abs(self.fake_y2 - self.input_y)) + \
                      ops.generator_loss(self.hps.gan_type, self.pred_fake_xy2)

        self.loss_Dx = ops.discriminator_loss(self.hps.gan_type, self.pred_real_yx, self.pred_pool_yx)
        self.loss_F = self.hps.l1_lambda * tf.reduce_mean(tf.abs(self.fake_x - self.input_x)) + \
                       ops.generator_loss(self.hps.gan_type, self.pred_fake_yx)

    def optimize_model(self):
        def make_optimizer(loss, lr, variables, optimizer="Adam", name='Adam'):
            # tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)
            step = tf.Variable(0, name=name + "_step", trainable=False)
            if optimizer == "Adam":
                learning_step = (
                    tf.train.AdamOptimizer(lr, beta1=self.hps.beta1, name=name)
                        .minimize(loss, global_step=step, var_list=variables)
                )
            elif optimizer == "RMSProp":
                learning_step = (
                    tf.train.RMSPropOptimizer(lr, name=name)
                        .minimize(loss, global_step=step, var_list=variables)
                )
            elif optimizer == "GD" or optimizer == "SGD":
                learning_step = (
                    tf.train.GradientDescentOptimizer(lr, name=name)
                        .minimize(loss, global_step=step, var_list=variables)
                )
            else:
                raise Exception("Unexpected optimizer: {}".format(name))

            return learning_step

        # self.g_lr = (
        #     tf.where(
        #         tf.greater_equal(self.global_step, self.start_decay_steps),
        #         tf.train.polynomial_decay(self.hps.lr / 2, self.global_step - self.start_decay_steps,
        #                                   self.decay_steps, self.hps.min_learning_rate / 2,
        #                                   power=1.0),
        #         self.hps.lr / 2
        #     )
        # )
        # self.d_lr = (
        #     tf.where(
        #         tf.greater_equal(self.global_step, self.start_decay_steps),
        #         tf.train.polynomial_decay(self.hps.lr, self.global_step - self.start_decay_steps,
        #                                   self.decay_steps, self.hps.min_learning_rate,
        #                                   power=1.0),
        #         self.hps.lr
        #     )
        # )
        # self.g_lr = self.hps.g_lr
        # self.d_lr = self.hps.d_lr
        self.g_lr = 0.0001
        self.d_lr = 0.0002

        g_vars = self.G.get_variables()
        f_vars = self.F.get_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.Dy_optimizer1 = make_optimizer(self.loss_Dy1, self.d_lr,
                                               self.Dy.get_variables(),
                                               optimizer="Adam", name='Adam_Dy')
            self.Dy_optimizer2 = make_optimizer(self.loss_Dy2, self.d_lr,
                                                self.Dy.get_variables(),
                                                optimizer="Adam", name='Adam_Dy')
            self.G_optimizer1 = make_optimizer(self.loss_G1, self.g_lr,
                                              g_vars,
                                              optimizer="Adam", name='Adam_G')
            self.G_optimizer2 = make_optimizer(self.loss_G2, self.g_lr,
                                              g_vars,
                                              optimizer="Adam", name='Adam_G')
            self.Dx_optimizer = make_optimizer(self.loss_Dx, self.d_lr,
                                                self.Dx.get_variables(),
                                                optimizer="Adam", name='Adam_Dx')
            self.F_optimizer = make_optimizer(self.loss_F, self.g_lr,
                                               f_vars,
                                               optimizer="Adam", name='Adam_F')
            self.step_op = tf.assign(self.global_step, self.global_step + 1)
