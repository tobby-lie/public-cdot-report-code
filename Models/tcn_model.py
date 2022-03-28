import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tcn import TCN


class TCNModel():

    def __init__(self, data, inputs_size, out_steps, num_features):
        self._data = data
        self._inputs_size = inputs_size
        self._out_steps = out_steps
        self._num_features = num_features

    def model(self):
        batch_size, timesteps, input_dim = None, int(
            (self._data.train.shape[1])*self._inputs_size), self._num_features

        i = Input(batch_shape=(batch_size, timesteps, input_dim))

        o = TCN(return_sequences=False)(i)
        o = Dense(self._out_steps*self._num_features,
                  kernel_initializer=tf.initializers.zeros())(o)
        o = tf.keras.layers.Reshape([self._out_steps, self._num_features])(o)

        model = Model(inputs=[i], outputs=[o])
        return model

def build_model(hp):

    with open('api/Models/config_steps_features.txt', 'r') as file:
        lines = file.readlines()
        num_features = int(lines[0])
        OUT_STEPS = int(lines[1])
        CONV_WIDTH = int(lines[2])

    batch_size, timesteps, input_dim = None, CONV_WIDTH, num_features

    i = Input(batch_shape=(batch_size, timesteps, input_dim))

    o = TCN(nb_filters=hp.Int("nb_filters", min_value=32, max_value=64, step=32),
            nb_stacks=hp.Int("nb_stacks", min_value=3, max_value=6, step=3),
            dropout_rate=hp.Float("dropout_rate", min_value=0.0,
                                  max_value=0.8, step=0.2),
            use_batch_norm=hp.Choice("use_batch_norm", values=[True, False]),
            # use_layer_norm=hp.Choice("use_layer_norm", values=[True, False]),
            # use_weight_norm=hp.Choice("use_weight_norm", values=[True, False]),
            return_sequences=False)(i)
    o = Dense(OUT_STEPS*num_features,
              kernel_initializer=tf.initializers.zeros())(o)
    o = tf.keras.layers.Reshape([OUT_STEPS, num_features])(o)

    model = Model(inputs=[i], outputs=[o])
    opt = hp.Choice("optimizer", values=['adam'])

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=[tf.metrics.MeanAbsoluteError()])

    return model
