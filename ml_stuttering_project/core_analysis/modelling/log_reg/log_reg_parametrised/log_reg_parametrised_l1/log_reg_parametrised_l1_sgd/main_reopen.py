#!/usr/bin/env python
# coding: utf-8
from tensorflow.keras import callbacks
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import warnings
from instantiate_data import *
from initialise_settings_add_dim_x import *
from standard_scaler import *
from reformat_data import *
from get_summary_stats import *
from create_output_dir import *
from l1_sgd_get_best_hyperparameters import *
from log_reg_layer import *

def log_reg_loss(labels, y_hat):
    epsilon = tf.keras.backend.epsilon()
    y_hat_clipped = tf.clip_by_value(y_hat, epsilon, 1 - epsilon)
    y_hat_log = tf.math.log(y_hat_clipped)
    cross_entropy = -tf.reduce_sum(labels * y_hat_log, axis=1)
    loss_f = tf.reduce_mean(cross_entropy)
    return loss_f

def get_l1_sgd_best_params(l1_sgd_best_hps):
    sgd_learning_rate = l1_sgd_best_hps.get('sgd_learning_rate')
    sgd_momentum = l1_sgd_best_hps.get('sgd_momentum')
    sgd_nesterov = l1_sgd_best_hps.get('sgd_nesterov')
    kernel_regularizer_l1 = l1_sgd_best_hps.get('kernel_regularizer_l1')
    kernel_regularizer_l2 = l1_sgd_best_hps.get('kernel_regularizer_l2')
    bias_regularizer_value = l1_sgd_best_hps.get('bias_regularizer_value')
    return [sgd_learning_rate, sgd_momentum, sgd_nesterov, kernel_regularizer_l1, kernel_regularizer_l2, bias_regularizer_value]

def create_l1_sgd_model_from_best_hps(sgd_learning_rate, sgd_momentum, sgd_nesterov,
                                       kernel_regularizer_l1, kernel_regularizer_l2, bias_regularizer_value):
    data = InstantiateData(data_dir='/home/debian/DATA')
    kernel_regularizer = regularizers.l1_l2(l1=kernel_regularizer_l1, l2=kernel_regularizer_l2)
    bias_regularizer = regularizers.l1(bias_regularizer_value)
    num_features = data.xtrain.shape[1]
    num_classes = len(np.unique(data.ytrain))
    images = tf.keras.Input(shape=(num_features,), name="images", dtype=tf.dtypes.int32)
    labels = tf.keras.Input(shape=(1,), name="labels", dtype=tf.dtypes.int32)
    y_hat = LogRegLayer(num_classes, num_features, kernel_regularizer, bias_regularizer)(images, labels)
    log_reg_model = tf.keras.Model(inputs=[images, labels], outputs=y_hat, name="log_reg_model")
    log_reg_model.compile(optimizer=tf.keras.optimizers.SGD(
            learning_rate=sgd_learning_rate,
            momentum=sgd_momentum,
            nesterov=sgd_nesterov,
            name='SGD'), loss=[log_reg_loss])
    return log_reg_model



def fit_model(model, data, batch_size, epochs, log_reg_l1_sgd_dir):
    my_callbacks = [callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00001, patience=4),
                    callbacks.ModelCheckpoint(
                        filepath=str(log_reg_l1_sgd_dir.log_reg_l1_sgd_param_partial_models) + '/model.{epoch:02d}.h5'),
                   ]
    model.fit(data.training_data, validation_data=(data.validation_data), shuffle=False, epochs=epochs,
              batch_size=batch_size, callbacks=my_callbacks)
    return model

class OutputModel:
    def __init__(self, lr_output, model, name, *args, **kwargs):
        super(OutputModel, self).__init__(*args, **kwargs)
        self.model_dir = lr_output.log_reg_l1_sgd_param_final_model
        self.model = model
        self.name = name
        self.save_model()

    def save_model(self):
        os.chdir(self.model_dir)
        self.model.save(str(str(self.name) + '.model'))
        return self

def run_parameterised_l1_sgd_log_reg():
    start_time = datetime.now()
    batch_size = 14000
    #epochs = 100000
    epochs = 100

    data_dir = '/home/debian/DATA/'
    results_dir = '/home/debian/RESULTS/'
    project_name = 'log_reg_tf_l1_sgd_hyperbanding_tensorboard'

    log_reg_l1_sgd_dir = CreateOutputDirectory(results_dir)

    data = InstantiateData(data_dir)
    data = ConductSklearnStandardScaling(data)
    init_sets = InitialiseSettings(seed_value=1234)
    data = DimXNumCats(data, init_sets)
    data = ReformatData(data)

    #GET BEST PARAMETERS
    l1_sgd_best_hps = run_tuner_get_best_hyperparameters_l1_sgd(log_reg_l1_sgd_dir, project_name, epochs)

    [sgd_learning_rate, sgd_momentum, sgd_nesterov, kernel_regularizer_l1,
     kernel_regularizer_l2, bias_regularizer_value] = get_l1_sgd_best_params(l1_sgd_best_hps)

    #COMPILE MODEL BASED ON BEST PARAMETERS
    log_reg_l1_sgd_model = create_l1_sgd_model_from_best_hps(sgd_learning_rate, sgd_momentum, sgd_nesterov,
                                       kernel_regularizer_l1, kernel_regularizer_l2, bias_regularizer_value)

    #FIT MODEL
    log_reg_l1_sgd_model = fit_model(log_reg_l1_sgd_model, data, batch_size, epochs, log_reg_l1_sgd_dir)

    #GET STATISTICAL OUTPUT
    test_results = log_reg_l1_sgd_model.evaluate(data.validation_data, verbose=1)
    stutter_preds = log_reg_l1_sgd_model.predict(data.validation_data)

    num_classes = len(np.unique(np.argmax(data.ytest, axis=1)))

    class_report = classification_report(np.argmax(data.ytest, axis=1),np.argmax(stutter_preds, axis=1))
    class_report = pd.DataFrame([class_report]).transpose()
    conf_mat = confusion_matrix(np.argmax(data.ytest, axis=1), np.argmax(stutter_preds, axis=1))
    conf_mat = pd.DataFrame(conf_mat)
    test_results = pd.DataFrame([test_results])
    test_results.columns = ['Loss', 'Accuracy']

    roc_data_ytest=np.reshape(np.argmax(to_categorical(np.array(data.ytest), num_classes), axis=1).astype(int).ravel(),[data.ytest.shape[0], num_classes])
    roc_data_ytest[roc_data_ytest > 0] = 1

    roc_stutter_preds=np.reshape(np.argmax(to_categorical(np.array(stutter_preds), num_classes), axis=1).astype(int).ravel(),[data.ytest.shape[0], num_classes])
    roc_stutter_preds[roc_stutter_preds > 0] = 1

    [mse, bic, aic] = get_mse_bic_aic(np.argmax(data.ytest, axis=1), np.argmax(stutter_preds, axis=1), log_reg_l1_sgd_model)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        roc_auc = compute_roc_and_auc(np.argmax(data.ytest, axis=1), np.argmax(stutter_preds, axis=1), num_classes)

    summary_stats = pd.concat([roc_auc, mse, bic, aic], axis=1).T.reset_index()

    best_hps_vals_df = pd.DataFrame(l1_sgd_best_hps.values, index=[0]).T
    os.chdir(log_reg_l1_sgd_dir.log_reg_l1_sgd_param_results)
    test_results.to_csv('test_results.csv', index=False)
    summary_stats.to_csv('summary_stats.csv', index=False)
    pd.DataFrame(conf_mat).to_csv('conf_mat.csv', index=False)
    pd.DataFrame(class_report).to_csv('class_report.csv', index=False)
    best_hps_vals_df.to_csv('best_hps_vals.csv', index=False)

    OutputModel(log_reg_l1_sgd_dir, log_reg_l1_sgd_model, name='log_reg_l1_sgd_model')

    time_delta = datetime.now() - start_time

    exit_message = 'LR NN L1 sgd PARAMETERISED RAN SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN ALL
run_parameterised_l1_sgd_log_reg()
