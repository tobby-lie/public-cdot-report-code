# python dependencies
import tensorflow as tf
from math import ceil
import tqdm
import keras_tuner as kt

# config
from . import config
from api import config as ac

# api utils
from api.utils.window_generator import WindowGenerator
from api.utils.rmse_generator import RMSEGenerator


def get_model_config_from_file():
    """
    Get config variables for model parameters
    """
    with open('api/Models/config_steps_features.txt', 'r') as file:
        lines = file.readlines()
        num_features = int(lines[0])
        OUT_STEPS = int(lines[1])
        CONV_WIDTH = int(lines[2])

    return num_features, OUT_STEPS, CONV_WIDTH


def compile_and_fit_linear(model, window, tuner_outfile, normalized=False, d_mean=None, d_std=None, max_epochs=100, patience=50):
    """
    Compiles and fits a model passed in using window passed in
    Parameters
    ----------
    model: tensorflow model
        tensorflow model to train on time series data
    window: WindowGenerator object
        train, valid, and test window for training
    patience: int
        optinoal patience for callback into model

    Raises
    ------
    ValueError:
        if normalized is set to True and no d_mean and d_std are provided
    """

    num_features, OUT_STEPS, CONV_WIDTH = get_model_config_from_file()

    if normalized == True:
        if d_mean == None or d_std == None:
            raise ValueError(
                "normalized=True but d_mean=None or d_std=None")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    if normalized == True:

        print("* instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=tuner_outfile,
            overwrite=True,
            project_name="hyperband")

        trainX = (window.train[0] - d_mean) / d_std
        validX = (window.val[0] - d_mean) / d_std

        tuner.search(
            x=trainX, y=window.train[1],
            validation_data=(validX, window.val[1]),
            batch_size=config.BS,
            callbacks=[early_stopping],
            epochs=max_epochs
        )

        # grab the best hyperparameters
        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of units in dense_1 layer: {}".format(
            bestHP.get("dense_1")))
        print("* optimal regularizer: {}".format(
            bestHP.get("regularizer")))
        print("* optimal dropout: {}".format(
            bestHP.get("dropout")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)

        history = model.fit(x=trainX, y=window.train[1],
                            validation_data=(
            validX, window.val[1]), batch_size=config.BS,
            epochs=max_epochs, callbacks=[early_stopping], verbose=1)

    else:

        print("* instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=tuner_outfile,
            overwrite=True,
            project_name="hyperband")

        tuner.search(
            x=window.train[0], y=window.train[1],
            validation_data=(window.val[0], window.val[1]),
            batch_size=config.BS,
            callbacks=[early_stopping],
            epochs=max_epochs
        )

        # grab the best hyperparameters
        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of units in dense_1 layer: {}".format(
            bestHP.get("dense_1")))
        print("* optimal regularizer: {}".format(
            bestHP.get("regularizer")))
        print("* optimal dropout: {}".format(
            bestHP.get("dropout")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)
        history = model.fit(x=window.train[0], y=window.train[1],
                            validation_data=(
            window.val[0], window.val[1]), batch_size=config.BS,
            epochs=max_epochs, callbacks=[early_stopping], verbose=1)

        rmse_generator = RMSEGenerator(
            model=model, window=window, num_steps=OUT_STEPS)

        params = {
            "linear_raw_dense_1": bestHP.get("dense_1"),
            "linear_raw_regularizer": bestHP.get("regularizer"),
            "linear_raw_dropout": bestHP.get("dropout"),
            "linear_raw_Optimizer": bestHP.get("optimizer")
        }

        metrics = {"linear_raw_deck_condition_rmse": rmse_generator.average_rmse_across_timesteps(1),
                   "linear_raw_superstructure_condition_rmse": rmse_generator.average_rmse_across_timesteps(2),
                   "linear_raw_substructure_condition_rmse": rmse_generator.average_rmse_across_timesteps(3)}

        model_name = "linear_raw"
        artifact_path = "model"

    return history, model, params, metrics, model_name, artifact_path


def compile_and_fit_tcn(model, window, tuner_outfile, normalized=False, d_mean=None, d_std=None, max_epochs=100, patience=50):
    """
    Compiles and fits a model passed in using window passed in
    Parameters
    ----------
    model: tensorflow model
        tensorflow model to train on time series data
    window: WindowGenerator object
        train, valid, and test window for training
    patience: int
        optinoal patience for callback into model

    Raises
    ------
    ValueError:
        if normalized is set to True and no d_mean and d_std are provided
    """

    num_features, OUT_STEPS, CONV_WIDTH = get_model_config_from_file()

    if normalized == True:
        if d_mean == None or d_std == None:
            raise ValueError(
                "normalized=True but d_mean=None or d_std=None")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    if normalized == True:

        print("* instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            model,
            objective="val_loss",
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=tuner_outfile,
            overwrite=True,
            project_name="hyperband")

        trainX = (window.train[0] - d_mean) / d_std
        validX = (window.val[0] - d_mean) / d_std

        tuner.search(
            x=trainX, y=window.train[1],
            validation_data=(validX, window.val[1]),
            batch_size=config.BS,
            callbacks=[early_stopping],
            epochs=max_epochs
        )

        # grab the best hyperparameters
        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of filters: {}".format(
            bestHP.get("nb_filters")))
        print("* optimal number of stacks: {}".format(
            bestHP.get("nb_stacks")))
        print("* optimal dropout rate: {}".format(
            bestHP.get("dropout_rate")))
        print("* optimal batch_norm utilization: {}".format(
            bestHP.get("use_batch_norm")))
        print("* optimal layer_norm utilization: {}".format(
            bestHP.get("use_layer_norm")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)

        history = model.fit(x=trainX, y=window.train[1],
                            validation_data=(
            validX, window.val[1]), batch_size=config.BS,
            epochs=max_epochs, callbacks=[early_stopping], verbose=1)

    else:

        # print("* instantiating a hyperband tuner object...")
        # tuner = kt.Hyperband(
        #     model,
        #     objective="val_loss",
        #     max_epochs=max_epochs,
        #     factor=3,
        #     seed=42,
        #     directory=tuner_outfile,
        #     overwrite=True,
        #     project_name="hyperband")

        # tuner.search(
        #     x=window.train[0], y=window.train[1],
        #     validation_data=(window.val[0], window.val[1]),
        #     batch_size=config.BS,
        #     callbacks=[early_stopping],
        #     epochs=max_epochs
        # )

        print("* instantiating a hyperband tuner object...")
        tuner = kt.BayesianOptimization(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            # max_epochs=ac.MAX_EPOCHS_TUNER,
            # factor=3,
            seed=42,
            directory=tuner_outfile,
            overwrite=True,
            project_name="hyperband",
            max_trials=4)

        tuner.search(
            x=window.train[0], y=window.train[1],
            validation_data=(window.val[0], window.val[1]),
            batch_size=config.BS,
            # callbacks=[early_stopping],
            epochs=ac.MAX_EPOCHS_TUNER
        )

        # grab the best hyperparameters
        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of filters: {}".format(
            bestHP.get("nb_filters")))
        print("* optimal number of stacks: {}".format(
            bestHP.get("nb_stacks")))
        print("* optimal dropout rate: {}".format(
            bestHP.get("dropout_rate")))
        print("* optimal batch_norm utilization: {}".format(
            bestHP.get("use_batch_norm")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        tf.keras.backend.clear_session()

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)
        history = model.fit(x=window.train[0], y=window.train[1],
                            validation_data=(
            window.val[0], window.val[1]), batch_size=config.BS,
            epochs=max_epochs, callbacks=[early_stopping], verbose=1)

        tf.keras.backend.clear_session()

        rmse_generator = RMSEGenerator(
            model=model, window=window, num_steps=OUT_STEPS)

        # params = {
        #     "tcn_raw_nb_filters": bestHP.get("nb_filters"),
        #     "tcn_raw_nb_stacks": bestHP.get("nb_stacks"),
        #     "tcn_raw_dropout_rate": bestHP.get("dropout_rate"),
        #     "tcn_raw_use_batch_norm": bestHP.get("use_batch_norm"),\
        #     "tcn_raw_optimizer": bestHP.get("optimizer")
        # }

        params = {
            "tcn_raw_nb_filters": None,
            "tcn_raw_nb_stacks": None,
            "tcn_raw_dropout_rate": None,
            "tcn_raw_use_batch_norm": None,
            "tcn_raw_optimizer": None
        }

        metrics = {"tcn_raw_deck_condition_rmse": rmse_generator.average_rmse_across_timesteps(0),
                   "tcn_raw_superstructure_condition_rmse": rmse_generator.average_rmse_across_timesteps(1),
                   "tcn_raw_substructure_condition_rmse": rmse_generator.average_rmse_across_timesteps(2)}

        model_name = "tcn_raw"
        artifact_path = "model"

    return history, model, params, metrics, model_name, artifact_path


def compile_and_fit_cnn(model, window, tuner_outfile, normalized=False, d_mean=None, d_std=None, max_epochs=100, patience=50):
    """
    Compiles and fits a model passed in using window passed in
    Parameters
    ----------
    model: tensorflow model
        tensorflow model to train on time series data
    window: WindowGenerator object
        train, valid, and test window for training
    patience: int
        optinoal patience for callback into model

    Raises
    ------
    ValueError:
        if normalized is set to True and no d_mean and d_std are provided
    """

    num_features, OUT_STEPS, CONV_WIDTH = get_model_config_from_file()

    if normalized == True:
        if d_mean == None or d_std == None:
            raise ValueError(
                "normalized=True but d_mean=None or d_std=None")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    if normalized == True:

        print("* instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=tuner_outfile,
            overwrite=True,
            project_name="hyperband")

        trainX = (window.train[0] - d_mean) / d_std
        validX = (window.val[0] - d_mean) / d_std

        tuner.search(
            x=trainX, y=window.train[1],
            validation_data=(validX, window.val[1]),
            batch_size=config.BS,
            callbacks=[early_stopping],
            epochs=max_epochs
        )

        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of units in conv_1 layer: {}".format(
            bestHP.get("conv_1")))
        print("* optimal L1 layer: {}".format(
            bestHP.get("regularizer")))
        print("* optimal dropout: {}".format(
            bestHP.get("dropout")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)

        history = model.fit(x=trainX, y=window.train[1],
                            validation_data=(
            validX, window.val[1]), batch_size=config.BS,
            epochs=max_epochs, callbacks=[early_stopping], verbose=1)

    else:

        # print("* instantiating a hyperband tuner object...")
        # tuner = kt.Hyperband(
        #     model,
        #     objective="val_loss",
        #     max_epochs=max_epochs,
        #     factor=3,
        #     seed=42,
        #     directory=tuner_outfile,
        #     overwrite=True,
        #     project_name="hyperband")

        # tuner.search(
        #     x=window.train[0], y=window.train[1],
        #     validation_data=(window.val[0], window.val[1]),
        #     batch_size=config.BS,
        #     callbacks=[early_stopping],
        #     epochs=max_epochs
        # )

        print("* instantiating a hyperband tuner object...")
        tuner = kt.BayesianOptimization(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            # max_epochs=ac.MAX_EPOCHS_TUNER,
            # factor=3,
            seed=42,
            directory=tuner_outfile,
            overwrite=True,
            project_name="hyperband",
            max_trials=10)

        tuner.search(
            x=window.train[0], y=window.train[1],
            validation_data=(window.val[0], window.val[1]),
            batch_size=config.BS,
            # callbacks=[early_stopping],
            epochs=ac.MAX_EPOCHS_TUNER
        )

        tf.keras.backend.clear_session()

        # grab the best hyperparameters
        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of units in conv_1 layer: {}".format(
            bestHP.get("conv_1")))
        print("* optimal L1 layer: {}".format(
            bestHP.get("regularizer")))
        print("* optimal dropout: {}".format(
            bestHP.get("dropout")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)

        # model.compile(loss=tf.losses.MeanSquaredError(),
        #             optimizer='adam',
        #             metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(x=window.train[0], y=window.train[1],
                            validation_data=(
            window.val[0], window.val[1]), batch_size=config.BS,
            epochs=max_epochs, callbacks=[early_stopping], verbose=1)
        
        tf.keras.backend.clear_session()

        rmse_generator = RMSEGenerator(
            model=model, window=window, num_steps=OUT_STEPS)

        # params = {
        #     "cnn_raw_conv_1": bestHP.get("conv_1"),
        #     "cnn_raw_regularizer": bestHP.get("regularizer"),
        #     "cnn_raw_dropout": bestHP.get("dropout"),
        #     "cnn_raw_Optimizer": bestHP.get("optimizer")
        # }

        params = {
            "cnn_raw_conv_1": None,
            "cnn_raw_regularizer": None,
            "cnn_raw_dropout": None,
            "cnn_raw_Optimizer": None
        }

        metrics = {"cnn_raw_deck_condition_rmse": rmse_generator.average_rmse_across_timesteps(0),
                   "cnn_raw_superstructure_condition_rmse": rmse_generator.average_rmse_across_timesteps(1),
                   "cnn_raw_substructure_condition_rmse": rmse_generator.average_rmse_across_timesteps(2)}

        model_name = "cnn_raw"
        artifact_path = "model"

    return history, model, params, metrics, model_name, artifact_path


def compile_and_fit_lstm(model, window, tuner_outfile, normalized=False, d_mean=None, d_std=None, max_epochs=100, patience=50):
    """
    Compiles and fits a model passed in using window passed in
    Parameters
    ----------
    model: tensorflow model
        tensorflow model to train on time series data
    window: WindowGenerator object
        train, valid, and test window for training
    patience: int
        optinoal patience for callback into model

    Raises
    ------
    ValueError:
        if normalized is set to True and no d_mean and d_std are provided
    """

    if normalized == True:
        if d_mean == None or d_std == None:
            raise ValueError(
                "normalized=True but d_mean=None or d_std=None")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    if normalized == True:

        print("* instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=tuner_outfile,
            project_name="hyperband")

        trainX = (window.train[0] - d_mean) / d_std
        validX = (window.val[0] - d_mean) / d_std

        tuner.search(
            x=trainX, y=window.train[1],
            validation_data=(validX, window.val[1]),
            batch_size=config.BS,
            callbacks=[early_stopping],
            epochs=max_epochs
        )

        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of units in lstm_1 layer: {}".format(
            bestHP.get("lstm_1")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)

        history = model.fit(x=trainX, y=window.train[1],
                            validation_data=(
            validX, window.val[1]), batch_size=config.BS,
            epochs=max_epochs, callbacks=[early_stopping], verbose=1)

    else:

        print("* instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=tuner_outfile,
            project_name="hyperband")

        tuner.search(
            x=window.train[0], y=window.train[1],
            validation_data=(window.val[0], window.val[1]),
            batch_size=config.BS,
            callbacks=[early_stopping],
            epochs=max_epochs
        )

        # grab the best hyperparameters
        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of units in lstm_1 layer: {}".format(
            bestHP.get("lstm_1")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)
        history = model.fit(x=window.train[0], y=window.train[1],
                            validation_data=(
            window.val[0], window.val[1]), batch_size=config.BS,
            epochs=max_epochs, callbacks=[early_stopping], verbose=1)

    return history, model


def compile_and_fit_bilstm(model, window, tuner_outfile, normalized=False, d_mean=None, d_std=None, max_epochs=100, patience=50):
    """
    Compiles and fits a model passed in using window passed in
    Parameters
    ----------
    model: tensorflow model
        tensorflow model to train on time series data
    window: WindowGenerator object
        train, valid, and test window for training
    patience: int
        optinoal patience for callback into model

    Raises
    ------
    ValueError:
        if normalized is set to True and no d_mean and d_std are provided
    """

    if normalized == True:
        if d_mean == None or d_std == None:
            raise ValueError(
                "normalized=True but d_mean=None or d_std=None")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    if normalized == True:

        print("* instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=tuner_outfile,
            project_name="hyperband")

        trainX = (window.train[0] - d_mean) / d_std
        validX = (window.val[0] - d_mean) / d_std

        tuner.search(
            x=trainX, y=window.train[1],
            validation_data=(validX, window.val[1]),
            batch_size=config.BS,
            callbacks=[early_stopping],
            epochs=max_epochs
        )

        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of units in lstm_1 layer: {}".format(
            bestHP.get("lstm_1")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)

        history = model.fit(x=trainX, y=window.train[1],
                            validation_data=(
            validX, window.val[1]), batch_size=config.BS,
            epochs=max_epochs, callbacks=[early_stopping], verbose=1)

    else:

        print("* instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=tuner_outfile,
            project_name="hyperband")

        tuner.search(
            x=window.train[0], y=window.train[1],
            validation_data=(window.val[0], window.val[1]),
            batch_size=config.BS,
            callbacks=[early_stopping],
            epochs=max_epochs
        )

        # grab the best hyperparameters
        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of units in lstm_1 layer: {}".format(
            bestHP.get("lstm_1")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)
        history = model.fit(x=window.train[0], y=window.train[1],
                            validation_data=(
            window.val[0], window.val[1]), batch_size=config.BS,
            epochs=max_epochs, callbacks=[early_stopping], verbose=1)

    return history, model


def compile_and_fit_gru(model, window, tuner_outfile, normalized=False, d_mean=None, d_std=None, max_epochs=100, patience=50):
    """
    Compiles and fits a model passed in using window passed in
    Parameters
    ----------
    model: tensorflow model
        tensorflow model to train on time series data
    window: WindowGenerator object
        train, valid, and test window for training
    patience: int
        optinoal patience for callback into model

    Raises
    ------
    ValueError:
        if normalized is set to True and no d_mean and d_std are provided
    """

    if normalized == True:
        if d_mean == None or d_std == None:
            raise ValueError(
                "normalized=True but d_mean=None or d_std=None")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    if normalized == True:

        print("* instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=tuner_outfile,
            project_name="hyperband")

        trainX = (window.train[0] - d_mean) / d_std
        validX = (window.val[0] - d_mean) / d_std

        tuner.search(
            x=trainX, y=window.train[1],
            validation_data=(validX, window.val[1]),
            batch_size=config.BS,
            callbacks=[early_stopping],
            epochs=max_epochs
        )

        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of units in gru_1 layer: {}".format(
            bestHP.get("gru_1")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)

        history = model.fit(x=trainX, y=window.train[1],
                            validation_data=(
            validX, window.val[1]), batch_size=config.BS,
            epochs=max_epochs, callbacks=[early_stopping], verbose=1)

    else:

        print("* instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=tuner_outfile,
            project_name="hyperband")

        tuner.search(
            x=window.train[0], y=window.train[1],
            validation_data=(window.val[0], window.val[1]),
            batch_size=config.BS,
            callbacks=[early_stopping],
            epochs=max_epochs
        )

        # grab the best hyperparameters
        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of units in gru_1 layer: {}".format(
            bestHP.get("gru_1")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)
        history = model.fit(x=window.train[0], y=window.train[1],
                            validation_data=(
            window.val[0], window.val[1]), batch_size=config.BS,
            epochs=max_epochs, callbacks=[early_stopping], verbose=1)

    return history, model


def compile_and_fit_cnn_bilstm(model, window, tuner_outfile, normalized=False, d_mean=None, d_std=None, max_epochs=100, patience=50):
    """
    Compiles and fits a model passed in using window passed in
    Parameters
    ----------
    model: tensorflow model
        tensorflow model to train on time series data
    window: WindowGenerator object
        train, valid, and test window for training
    patience: int
        optinoal patience for callback into model

    Raises
    ------
    ValueError:
        if normalized is set to True and no d_mean and d_std are provided
    """

    if normalized == True:
        if d_mean == None or d_std == None:
            raise ValueError(
                "normalized=True but d_mean=None or d_std=None")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    if normalized == True:

        print("* instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=tuner_outfile,
            project_name="hyperband")

        trainX = (window.train[0] - d_mean) / d_std
        validX = (window.val[0] - d_mean) / d_std

        tuner.search(
            x=trainX, y=window.train[1],
            validation_data=(validX, window.val[1]),
            batch_size=config.BS,
            callbacks=[early_stopping],
            epochs=max_epochs
        )

        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of units in conv_1 layer: {}".format(
            bestHP.get("conv_1")))
        print("* optimal number of units in lstm_1 layer: {}".format(
            bestHP.get("lstm_1")))
        print("* optimal dropout_2: {}".format(
            bestHP.get("dropout_2")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)

        history = model.fit(x=trainX, y=window.train[1],
                            validation_data=(
            validX, window.val[1]), batch_size=config.BS,
            epochs=max_epochs, callbacks=[early_stopping], verbose=1)

    else:

        print("* instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=tuner_outfile,
            project_name="hyperband")

        tuner.search(
            x=window.train[0], y=window.train[1],
            validation_data=(window.val[0], window.val[1]),
            batch_size=config.BS,
            callbacks=[early_stopping],
            epochs=max_epochs
        )

        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of units in conv_1 layer: {}".format(
            bestHP.get("conv_1")))
        print("* optimal number of units in lstm_1 layer: {}".format(
            bestHP.get("lstm_1")))
        print("* optimal dropout_2: {}".format(
            bestHP.get("dropout_2")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)
        history = model.fit(x=window.train[0], y=window.train[1],
                            validation_data=(
            window.val[0], window.val[1]), batch_size=config.BS,
            epochs=max_epochs, callbacks=[early_stopping], verbose=1)

    return history, model


def compile_and_fit_multi_channel_cnn(model, window, tuner_outfile, normalized=False, d_mean=None, d_std=None, max_epochs=100, patience=50):
    """
    Compiles and fits a model passed in using window passed in
    Parameters
    ----------
    model: tensorflow model
        tensorflow model to train on time series data
    window: WindowGenerator object
        train, valid, and test window for training
    patience: int
        optinoal patience for callback into model

    Raises
    ------
    ValueError:
        if normalized is set to True and no d_mean and d_std are provided
    """

    if normalized == True:
        if d_mean == None or d_std == None:
            raise ValueError(
                "normalized=True but d_mean=None or d_std=None")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    if normalized == True:

        print("* instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=tuner_outfile,
            project_name="hyperband")

        trainX = (window.train[0] - d_mean) / d_std
        validX = (window.val[0] - d_mean) / d_std

        trainX = [trainX[:, :, idx] for idx in range(trainX.shape[2])]
        validX = [validX[:, :, idx] for idx in range(validX.shape[2])]

        tuner.search(
            x=trainX, y=window.train[1],
            validation_data=(validX, window.val[1]),
            batch_size=config.BS,
            callbacks=[early_stopping],
            epochs=max_epochs
        )

        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of units in nb_filters layer: {}".format(
            bestHP.get("nb_filters")))
        print("* optimal number of units in dense layer: {}".format(
            bestHP.get("dense")))
        print("* optimal dropout_rate: {}".format(
            bestHP.get("dropout_rate")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)

        history = model.fit(x=trainX, y=window.train[1], epochs=max_epochs,
                            validation_data=(validX, window.val[1]), batch_size=config.BS, callbacks=[early_stopping], verbose=1)

    else:
        trainX = [window.train[0][:, :, idx]
                  for idx in range(window.train[0].shape[2])]
        validX = [window.val[0][:, :, idx]
                  for idx in range(window.val[0].shape[2])]

        print("* instantiating a hyperband tuner object...")
        tuner = kt.Hyperband(
            model,
            objective=kt.Objective("val_loss", direction='min'),
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=tuner_outfile,
            project_name="hyperband")

        tuner.search(
            x=trainX, y=window.train[1],
            validation_data=(validX, window.val[1]),
            batch_size=config.BS,
            callbacks=[early_stopping],
            epochs=max_epochs
        )

        bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("* optimal number of units in nb_filters layer: {}".format(
            bestHP.get("nb_filters")))
        print("* optimal number of units in dense layer: {}".format(
            bestHP.get("dense")))
        print("* optimal dropout_rate: {}".format(
            bestHP.get("dropout_rate")))
        print("* optimal optimizer: {}".format(
            bestHP.get("optimizer")))

        print("* training the best model...")
        model = tuner.hypermodel.build(bestHP)

        history = model.fit(x=trainX, y=window.train[1], epochs=max_epochs,
                            validation_data=(validX, window.val[1]), batch_size=config.BS, callbacks=[early_stopping], verbose=1)

    return history, model


def discretize_timesteps(prediction_container):
    """
    Discretizes timestep predictions

    Parameters
    ----------
    prediction_container: PredictionContainer object
        prediction container encapsulating inputs and predictions
    """

    print("* Truncating out of bounds results . . .")
    for i in tqdm.tqdm(range(prediction_container.results_array.shape[0])):
        for j in range(prediction_container.results_array.shape[1]):
            if prediction_container.results_array[i][j][0] > 9:
                prediction_container.results_array[i][j][0] = 9
            if prediction_container.results_array[i][j][0] < 0:
                prediction_container.results_array[i][j][0] = 0

            if prediction_container.results_array[i][j][1] > 9:
                prediction_container.results_array[i][j][1] = 9
            if prediction_container.results_array[i][j][1] < 0:
                prediction_container.results_array[i][j][1] = 0

            if prediction_container.results_array[i][j][2] > 9:
                prediction_container.results_array[i][j][2] = 9
            if prediction_container.results_array[i][j][2] < 0:
                prediction_container.results_array[i][j][2] = 0

    print("* Applying ceil function to results . . .")
    for i in tqdm.tqdm(range(prediction_container.results_array.shape[0])):
        for j in range(prediction_container.results_array.shape[1]):
            prediction_container.results_array[i][j][0] = ceil(
                prediction_container.results_array[i][j][0])

            prediction_container.results_array[i][j][1] = ceil(
                prediction_container.results_array[i][j][1])

            prediction_container.results_array[i][j][2] = ceil(
                prediction_container.results_array[i][j][2])

    return prediction_container


def window_from_prediction_container(data, prediction_container, inputs_size, labels_size):

    # Data window constructed based on input size an labels size
    window = WindowGenerator(
        int((prediction_container.results_array.shape[1])*inputs_size), int(
            (prediction_container.results_array.shape[1])*labels_size),
        prediction_container.results_array[:int(prediction_container.results_array.shape[0]*0.7), :,
                                           :], prediction_container.results_array[:int(prediction_container.results_array.shape[0]*0.2), :, :],
        prediction_container.results_array[:int(prediction_container.results_array.shape[0]*0.1), :, :], data.train_bridge_ids, data.valid_bridge_ids, data.test_bridge_ids, prediction_container.results_array.shape[1])

    return window
