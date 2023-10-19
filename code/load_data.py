import numpy as np
import random
import pandas as pd

class Dataset(object):
    '''
       Subclass must implements below functions:
           get_dataset()
           next_batch()
           validation()
           testing()
       '''

    def __init__(self, config):
        self.train_split = config.train_split
        self.val_split = config.val_split
        self.test_split = config.test_split
        self.m = config.IAE_inputnum

        self.pre_horizon = config.horizon-1

        # get dataset
        self.raw_dataset = self.get_dataset()
        # get scaled dataset and sacler
        self.train,self.test,self.val, self.mean, self.stdev = self.split_scaling(self.raw_dataset)

    def split_scaling(self,data):

        self.train_len = int(len(data) * self.train_split)
        self.val_len = int(len(data) * self.val_split)
        self.test_len = int(len(data) * self.test_split)

        self.train = data[:self.train_len, :]
        self.val = data[self.train_len:(self.train_len + self.val_len), :]
        self.test = data[(self.train_len + self.val_len):(self.train_len + self.val_len + self.test_len), :]

        self.mean = np.mean(self.train, axis=0)
        self.stdev = np.std(self.train, axis=0)
        # in case the stdev=0,then we will get nan
        for i in range(len(self.stdev)):
            if self.stdev[i] < 0.00000001:
                self.stdev[i] = 1
        self.train = (self.train - self.mean) / self.stdev
        self.test = (self.test - self.mean) / self.stdev
        self.val = (self.val - self.mean) / self.stdev
        return self.train,self.test,self.val, self.mean, self.stdev



class NasdaqDataset(Dataset):
    name = 'Nasdaq100'
    data_filename = './datasets/nasdaq100_padding.csv'
    def __init__(self, config):
        Dataset.__init__(self,config)
        # parameters for the network
        self.batch_size = config.batch_size
        self.IAE_steps = config.IAE_steps
        self.GTAD_steps = config.GTAD_steps

        self.n_train = len(self.train)
        self.n_val = len(self.val)
        self.n_test = len(self.test)
        self.n_feature = config.IAE_inputnum
        self.n_label = 1


    def get_dataset(self):
        data = pd.read_csv(self.data_filename)
        data = np.array(data)
        return data

    def next_batch(self):
        # generate of a random index from the range [0, self.n_train -self.GTAD_steps +1]
        index = random.sample(np.arange(0, self.n_train - self.GTAD_steps - self.pre_horizon - 1).tolist(),
                              self.batch_size)
        index = np.array(index)
        # the shape of batch_x, label, previous_y
        batch_x = np.zeros([self.batch_size, self.IAE_steps, self.n_feature])
        label = np.zeros([self.batch_size, self.n_label])
        previous_y = np.zeros([self.batch_size, self.GTAD_steps, self.n_label])
        temp = 0
        for item in index:
            A = self.train[item:item + self.IAE_steps - 1, :self.n_feature]  # flx
            B = self.train[item + self.IAE_steps + self.pre_horizon - 1, :self.n_feature].reshape(1, self.m)  # flx
            batch_x[temp, :, :] = np.append(A, B, axis=0)  # flx
            previous_y[temp, :, 0] = self.train[item:item + self.GTAD_steps, -1]
            temp += 1
        label[:, 0] = np.array(self.train[index + self.GTAD_steps + self.pre_horizon, -1])
        IAE_states = np.swapaxes(batch_x, 1, 2)
        return batch_x, label, previous_y, IAE_states

    def validation(self):
        index = np.arange(0, self.n_val - self.GTAD_steps - self.pre_horizon - 1)
        index_size = len(index)
        val_x = np.zeros([index_size, self.IAE_steps, self.n_feature])
        val_label = np.zeros([index_size, self.n_label])
        val_prev_y = np.zeros([index_size, self.GTAD_steps, self.n_label])
        temp = 0
        for item in index:
            A = self.val[item:item + self.IAE_steps - 1, :self.n_feature]
            B = self.val[item + self.IAE_steps + self.pre_horizon - 1, :self.n_feature].reshape(1, self.m)  # flx
            val_x[temp, :, :] = np.append(A, B, axis=0)
            val_prev_y[temp, :, 0] = self.val[item:item + self.GTAD_steps, -1]
            temp += 1

        val_label[:, 0] = np.array(self.val[index + self.GTAD_steps + self.pre_horizon, -1])

        IAE_states_val = np.swapaxes(val_x, 1, 2)
        return val_x, val_label, val_prev_y, IAE_states_val

    def testing(self):
        index = np.arange(0, self.n_test - self.GTAD_steps - self.pre_horizon - 1)
        index_size = len(index)
        test_x = np.zeros([index_size, self.IAE_steps, self.n_feature])
        test_label = np.zeros([index_size, self.n_label])
        test_prev_y = np.zeros([index_size, self.GTAD_steps, self.n_label])
        temp = 0
        for item in index:
            A = self.test[item:item + self.IAE_steps - 1, :self.n_feature]  # flx
            B = self.test[item + self.IAE_steps + self.pre_horizon - 1, :self.n_feature].reshape(1, self.m)  # flx
            test_x[temp, :, :] = np.append(A, B, axis=0)  # flx
            # test_x[temp,:,:] = self.test[item:item + self.IAE_steps, :self.n_feature]
            test_prev_y[temp, :, 0] = self.test[item:item + self.GTAD_steps, -1]
            temp += 1

        test_label[:, 0] = np.array(self.test[index + self.GTAD_steps + self.pre_horizon, -1])
        # test_x = self.get_wavelet(test_x)                 #wr
        IAE_states_test = np.swapaxes(test_x, 1, 2)
        return test_x, test_label, test_prev_y, IAE_states_test





