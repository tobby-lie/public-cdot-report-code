import tensorflow as tf


class CNNModel():

    # def __init__(self, out_steps, num_features):
    #     self._out_steps = out_steps
    #     self._num_features = num_features

    def model(self):

        with open('api/Models/config_steps_features.txt', 'r') as file:
            lines = file.readlines()
            num_features = int(lines[0])
            OUT_STEPS = int(lines[1])
            CONV_WIDTH = int(lines[2])

        model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            # tf.keras.layers.Lambda(lambda x: x[:, :, :]),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(
                64, activation='relu', kernel_size=(CONV_WIDTH)),
            
            tf.keras.layers.Dropout(0.2),
            # Shape => [batch, 1,  out_steps*features]

            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])
        return model


def build_model(hp):

    with open('api/Models/config_steps_features.txt', 'r') as file:
        lines = file.readlines()
        num_features = int(lines[0])
        OUT_STEPS = int(lines[1])
        CONV_WIDTH = int(lines[2])

    C_WIDTH = CONV_WIDTH
    model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]

        # tf.keras.layers.Lambda(
        #     lambda x: x[:, :, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(
            hp.Int("conv_1", min_value=16, max_value=128, step=16),
            kernel_regularizer=tf.keras.regularizers.L1(
                hp.Float("regularizer", min_value=0.0, max_value=0.8, step=0.2)),
            activation='relu', kernel_size=(C_WIDTH)),

        tf.keras.layers.Dropout(
            hp.Float("dropout", min_value=0.0, max_value=0.8, step=0.2)),
        # Shape => [batch, 1,  out_steps*features]
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
