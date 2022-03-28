import tensorflow as tf


class BILSTMModel():

    def __init__(self, out_steps, num_features):
        self._out_steps = out_steps
        self._num_features = num_features

    def model(self):
        model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units]
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=False)),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(self._out_steps*self._num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([self._out_steps, self._num_features])
        ])
        return model


def build_model(hp):

    with open('Models/config_steps_features.txt', 'r') as file:
        lines = file.readlines()
        num_features = int(lines[0])
        OUT_STEPS = int(lines[1])
        CONV_WIDTH = int(lines[2])

    model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            hp.Int("lstm_1", min_value=16, max_value=128, step=16), return_sequences=False)),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    opt = hp.Choice("optimizer", values=['adam'])

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=[tf.metrics.MeanAbsoluteError()])

    return model
