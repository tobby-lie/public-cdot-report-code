import tensorflow as tf
from tensorflow.keras.optimizers import Adamax


class MultiLinearModel():

    def __init__(self, out_steps, num_features):
        self._out_steps = out_steps
        self._num_features = num_features

    def model(self, hp):
        model = tf.keras.Sequential([
            # Take the last time-step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(
                lambda x: x[:, :, :]),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                hp.Int("dense_1", min_value=64, max_value=512, step=64)),

            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(self._out_steps*self._num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([self._out_steps, self._num_features])
        ])

        lr = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3])
        opt = hp.Choice("optimizer", values=['adam', 'sgd'])

        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=opt,
                      metrics=[tf.metrics.MeanAbsoluteError()])

        return model


def build_model(hp):

    with open('api/Models/config_steps_features.txt', 'r') as file:
        lines = file.readlines()
        num_features = int(lines[0])
        OUT_STEPS = int(lines[1])
        CONV_WIDTH = int(lines[2])

    model = tf.keras.Sequential([
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        # tf.keras.layers.Lambda(
        #     lambda x: x[:, -hp.Int("conv_width", min_value=1, max_value=CONV_WIDTH, step=1):, :]),\
        # tf.keras.layers.Lambda(
        #     lambda x: x[:, :, :]),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            hp.Int("dense_1", min_value=16, max_value=128, step=16),
            kernel_regularizer=tf.keras.regularizers.L1(hp.Float("regularizer", min_value=0.0, max_value=0.8, step=0.2))),

        tf.keras.layers.Dropout(
            hp.Float("dropout", min_value=0.0, max_value=0.8, step=0.2)),

        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    opt = hp.Choice("optimizer", values=['adam', 'RMSProp'])

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=[tf.metrics.MeanAbsoluteError()])

    return model
