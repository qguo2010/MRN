import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from models.models import *
from configs.config import *
from preprocess.load_data import *

import sklearn.metrics as metrics
from functools import reduce
from operator import mul
from matplotlib import pyplot as plt

CONFIG = NasdaqConfig
DATASET = NasdaqDataset

MODEL_DIR = os.path.join('logs', 'checkpoints')
LOG_DIR = os.path.join('logs', 'graphs')


def get_num_params():
    num_params = 0
    for v in tf.trainable_variables():
        shape = v.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


def make_config_string(config):
    return "Nsteps%s_Nfeatures%s_Nhiddens%s_horizons%s_bs%s_w%s" % \
           (config.IAE_steps, config.IAE_inputnum, config.IAE_hidden, \
            config.horizon, config.batch_size, config.W)


def make_log_dir(config, dataset):
    return os.path.join(LOG_DIR, dataset.name, 'horizon' + str(config.horizon), make_config_string(config))


def make_model_path(config, dataset):
    dir = os.path.join(MODEL_DIR, dataset.name, 'horizon' + str(config.horizon))
    if not os.path.exists(dir):
        os.makedirs(dir)
    return os.path.join(dir, make_config_string(config))


def calc_rmse(y_real_list, y_pred_list):
    rmse = np.sqrt(metrics.mean_squared_error(y_real_list, y_pred_list))
    return rmse


def calc_rrse(y_real_list, y_pred_list):
    sub_y_true = y_real_list - np.mean(y_real_list)
    rrse = np.sqrt(np.sum((y_real_list - y_pred_list) * (y_real_list - y_pred_list))) / np.sqrt(
        np.sum(sub_y_true * sub_y_true))
    return rrse


def calc_smape(y_real_list, y_pred_list):
    eps = 0.0001
    smape = np.mean(np.abs(y_real_list - y_pred_list) / (np.abs(y_real_list) + np.abs(y_pred_list) + eps)) * 100
    return smape


def calc_mae(y_real_list, y_pred_list):
    mae = metrics.mean_absolute_error(y_real_list, y_pred_list)
    return mae


def calc_corr(y_real_list, y_pred_list):
    sigma_p = (y_pred_list).std(axis=0)
    sigma_g = (y_real_list).std(axis=0)
    mean_p = y_pred_list.mean(axis=0)
    mean_g = y_real_list.mean(axis=0)
    corr = ((y_pred_list - mean_p) * (y_real_list - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    corr = (corr).mean()
    return corr


def run_train_config(config):
    iters = 50001
    display_step = 2000
    count = 1
    # load model
    model = MRN_CSG(config)
    saver = tf.train.Saver()
    # data process
    ds_handler = DATASET(config)
    # generate log and model stored paths
    log_path = make_log_dir(config, ds_handler)
    model_path = make_model_path(config, ds_handler)
    print('------------------------------------------------------------')
    print('Train Config:', make_config_string(config), '. Total training iters:', iters)
    print('Trainable parameter count:', get_num_params())
    print('------------------------------------------------------------')

    with tf.Session() as sess:
        # tensor board
        train_writer = tf.summary.FileWriter(os.path.join(log_path, 'train'), sess.graph)

        # Actually run
        sess.run(tf.global_variables_initializer())
        min_valloss = np.inf
        # training
        for step in range(iters):
            # adjust learning rate
            count += 1
            if count >= 10000:
                config.lr *= 0.1
                count = 0

            batch_x, batch_y, prev_y, IAE_states = ds_handler.next_batch()
            train_batch_data = [batch_x, batch_y, prev_y, IAE_states]
            # train
            loss_per,sss = model.train(train_batch_data, sess)
            train_writer.add_summary(sss, step)

            # evaluate
            if step % display_step == 0:
                val_x, val_y, val_prev_y, IAE_states_val = ds_handler.validation()
                valid_batch_data = [val_x, val_y, val_prev_y, IAE_states_val]
                loss_val = model.validation(valid_batch_data, sess)

                print("Iter {:-5d}".format(step) + ", Training Loss= {:.8f}".format(
                    loss_per / config.batch_size) + ", Validation loss = {:.8f}".format(loss_val))

                if loss_val <= min_valloss:  # save model
                    saver.save(sess, model_path + '\MRN_CSG.ckpt')
                    min_valloss = loss_val

        train_writer.close()
    # free default graph
    tf.reset_default_graph()


def run_test_config(config):
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        model = MRN_CSG(config)
        saver = tf.train.Saver()
        # data process
        ds_handler = DATASET(config)
        # generate log and model stored paths
        model_path = make_model_path(config, ds_handler)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))  # 加载变量值

        test_x, test_y, test_prev_y, IAE_states_test = ds_handler.testing()
        test_batch_data = [test_x, test_y, test_prev_y, IAE_states_test]
        loss_test, pred_y = model.predict(test_batch_data, sess)

        y_true = np.array(ds_handler.mean)[-1] + np.array(ds_handler.stdev)[-1] * test_y
        y_pred = np.array(ds_handler.mean)[-1] + np.array(ds_handler.stdev)[-1] * pred_y

        plt.figure()
        plt.plot(y_pred, label='Predicted')
        plt.plot(y_true, label="True")
        plt.legend(loc='upper left')
        plt.show()
        print("Testing loss:", loss_test)

        MAE = calc_mae(y_true, y_pred)
        sMAPE = calc_smape(y_true, y_pred)
        RMSE = calc_rmse(y_true, y_pred)
        RRSE = calc_rrse(y_true, y_pred)
        CORR = calc_corr(y_true, y_pred)

        print("MAE: {:.8f}".format(MAE), "      sMAPE: {:.8f}".format(sMAPE), "       RMSE: {:.8f}".format(RMSE),
              "       RRSE: {:.8f}".format(RRSE), "      CORR: {:.8f}".format(CORR))


if __name__ == '__main__':
    config = CONFIG()

    # training
    run_train_config(config)
    # testing
    run_test_config(config)
