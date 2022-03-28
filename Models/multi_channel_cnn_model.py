import tensorflow as tf


class MultiChannelCNNModel():

    def __init__(self, data, out_steps, num_features):
        self._out_steps = out_steps
        self._num_features = num_features
        self._data = data

    def model(self):
        inputs_list = []
        flats_list = []

        # channel 1
        inputs1 = tf.keras.Input(
            shape=(self._data.train.shape[1] - self._out_steps, 1))
        conv1 = tf.keras.layers.Conv1D(
            filters=8, kernel_size=5, activation='relu')(inputs1)
        drop1 = tf.keras.layers.Dropout(0.5)(conv1)
        pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop1)
        conv1_2 = tf.keras.layers.Conv1D(
            filters=4, kernel_size=5, activation='relu')(pool1)
        drop1_2 = tf.keras.layers.Dropout(0.5)(conv1_2)
        pool1_2 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop1_2)
        flat1 = tf.keras.layers.Flatten()(pool1_2)

        inputs_list.append(inputs1)
        flats_list.append(flat1)

        for _ in range(1, self._num_features):
            # channel
            inputs1 = tf.keras.Input(
                shape=(self._data.train.shape[1] - self._out_steps, 1))
            conv1 = tf.keras.layers.Conv1D(
                filters=64, kernel_size=3, activation='relu')(inputs1)
            drop1 = tf.keras.layers.Dropout(0.5)(conv1)
            pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop1)
            conv1_2 = tf.keras.layers.Conv1D(
                filters=4, kernel_size=5, activation='relu')(pool1)
            drop1_2 = tf.keras.layers.Dropout(0.5)(conv1_2)
            pool1_2 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop1_2)
            flat1 = tf.keras.layers.Flatten()(pool1_2)

            inputs_list.append(inputs1)
            flats_list.append(flat1)

        # merged
        merged = tf.keras.layers.concatenate(flats_list)

        # dense output layers
        dense1 = tf.keras.layers.Dense(units=732, activation='relu')(merged)

        dense2 = tf.keras.layers.Dense(self._out_steps*self._num_features,
                                       kernel_initializer=tf.initializers.zeros())(dense1)

        # Shape => [batch, out_steps, features]
        output = tf.keras.layers.Reshape(
            [self._out_steps, self._num_features])(dense2)

        model = tf.keras.Model(inputs=inputs_list, outputs=output)

        return model


def build_model(hp):

    with open('Models/config_steps_features.txt', 'r') as file:
        lines = file.readlines()
        num_features = int(lines[0])
        OUT_STEPS = int(lines[1])
        CONV_WIDTH = int(lines[2])

    inputs_list = []
    flats_list = []

    nb_filters = hp.Int("nb_filters", min_value=32, max_value=64, step=32)
    dropout_rate = hp.Float("dropout_rate", min_value=0.0,
                            max_value=0.8, step=0.2)
    dense_1 = hp.Int("dense", min_value=32, max_value=64, step=32)

    # channel 1
    inputs1 = tf.keras.Input(
        shape=(CONV_WIDTH, 1))
    conv1 = tf.keras.layers.Conv1D(
        filters=nb_filters, kernel_size=3, activation='relu')(inputs1)
    drop1 = tf.keras.layers.Dropout(dropout_rate)(conv1)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop1)
    conv1_2 = tf.keras.layers.Conv1D(
        filters=nb_filters, kernel_size=3, activation='relu')(pool1)
    drop1_2 = tf.keras.layers.Dropout(dropout_rate)(conv1_2)
    pool1_2 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop1_2)
    flat1 = tf.keras.layers.Flatten()(pool1_2)

    inputs_list.append(inputs1)
    flats_list.append(flat1)

    for _ in range(1, num_features):
        # channel
        inputs1 = tf.keras.Input(
            shape=(CONV_WIDTH, 1))
        conv1 = tf.keras.layers.Conv1D(
            filters=nb_filters, kernel_size=3, activation='relu')(inputs1)
        drop1 = tf.keras.layers.Dropout(dropout_rate)(conv1)
        pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop1)
        conv1_2 = tf.keras.layers.Conv1D(
            filters=nb_filters, kernel_size=3, activation='relu')(pool1)
        drop1_2 = tf.keras.layers.Dropout(dropout_rate)(conv1_2)
        pool1_2 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop1_2)
        flat1 = tf.keras.layers.Flatten()(pool1_2)

        inputs_list.append(inputs1)
        flats_list.append(flat1)

    # merged
    merged = tf.keras.layers.concatenate(flats_list)

    # dense output layers
    dense1 = tf.keras.layers.Dense(units=dense_1, activation='relu')(merged)

    dense2 = tf.keras.layers.Dense(OUT_STEPS*num_features,
                                   kernel_initializer=tf.initializers.zeros())(dense1)

    # Shape => [batch, out_steps, features]
    output = tf.keras.layers.Reshape(
        [OUT_STEPS, num_features])(dense2)

    model = tf.keras.Model(inputs=inputs_list, outputs=output)

    opt = hp.Choice("optimizer", values=['adam'])

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=opt,
                  metrics=[tf.metrics.MeanAbsoluteError()])

    return model
