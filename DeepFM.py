'''
Tensorflow implementation of Neural Factorization Machines as described in:
Xiangnan He, Tat-Seng Chua. Neural Factorization Machines for Sparse Predictive Analytics. In Proc. of SIGIR 2017.

This is a deep version of factorization machine and is more expressive than FM.

@author: 
Xiangnan He (xiangnanhe@gmail.com)
Lizi Liao (liaolizi.llz@gmail.com)

@references:
'''
import os
import sys
import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from time import time
import argparse
import LoadData_v2 as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural FM.")
    parser.add_argument('--path', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='company',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='Pre-train flag. 0: train from scratch; 1: load from pretrain file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=8,
                        help='Number of hidden factors.')
    parser.add_argument('--layers', nargs='?', default='[64,64]',
                        help="Size of each layer.")
    parser.add_argument('--keep_prob', nargs='?', default='[0.8,0.8,0.5]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--lamda', type=float, default=0.0001,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='square_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=1,
                    help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu',
                    help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--early_stop', type=int, default=1,
                    help='Whether to perform early stop (0 or 1)')
    # #valid_dimen是lib文件中字段的个数
    parser.add_argument('--valid_dimension', type=int, default=36,
                         help='Valid dimension of the dataset. (e.g. frappe=10, ml-tag=3)')
    # #define attention
    parser.add_argument('--attention', type=int, default=1,
                         help='flag for attention. 1: use attention; 0: no attention')
    return parser.parse_args()

class NeuralFM(BaseEstimator, TransformerMixin):
    def __init__(self, features_M, attention, valid_dimension, hidden_factor, layers, loss_type, pretrain_flag, epoch, batch_size, learning_rate, lamda_bilinear,
                 keep_prob, optimizer_type, batch_norm, activation_function, verbose, early_stop, random_seed=2016):
        # bind params to class

        self.batch_size = batch_size
        self.hidden_factor = hidden_factor
        self.layers = layers
        self.loss_type = loss_type
        self.pretrain_flag = pretrain_flag
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.epoch = epoch
        self.random_seed = random_seed
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for i in range(len(keep_prob))])
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.activation_function = activation_function
        self.early_stop = early_stop
        #
        self.valid_dimension = valid_dimension
        self.attention = attention
        # performance of each epoch
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_features = tf.placeholder(tf.int64, shape=[None, None])  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # ---------- first order term ----------
            self.y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], self.train_features)
            self.y_first_order = tf.reduce_sum(self.y_first_order, 1)
            # self.y_first_order = tf.reshape(self.y_first_order, shape=[-1, self.valid_dimension])
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep[0])

            # Model.
            # _________ sum_square part _____________
            # get the summed up embeddings of features.
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)
            self.summed_features_emb = tf.reduce_sum(nonzero_embeddings, 1) # None * K
            # get the element-multiplication
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # _________ square_sum part _____________
            self.squared_features_emb = tf.square(nonzero_embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            if self.attention:
                element_wise_product_list = []
                for i in range(self.valid_dimension):
                    for j in range(i + 1, self.valid_dimension):
                        element_wise_product_list.append(
                            tf.multiply(nonzero_embeddings[:, i, :], nonzero_embeddings[:, j, :]))
                self.element_wise_product = tf.stack(element_wise_product_list)
                self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1, 0, 2],
                                                         name='element_wise_product')

            # _________ attention part _____________
            num_interactions = self.valid_dimension * (self.valid_dimension - 1) / 2
            if self.attention:
                self.attention_mul = tf.reshape(tf.matmul(tf.reshape(self.element_wise_product, shape=[-1, self.hidden_factor]), \
                    self.weights['attention_W']), shape=[-1, int(num_interactions), 10])
                self.attention_relu = tf.reduce_sum(tf.multiply(tf.nn.relu(self.attention_mul + \
                    self.weights['attention_b']),self.weights['attention_h']), 2, keep_dims=True)
                self.attention_out = tf.nn.softmax(self.attention_relu)
                self.attention_out = tf.multiply(tf.transpose(self.weights['attention_p']), self.attention_out)
                self.attention_out = tf.nn.dropout(self.attention_out, self.dropout_keep[-1])

            # _________ Attention-aware Pairwise Interaction Layer _____________
            if self.attention:
                self.AFM = tf.reduce_sum(tf.multiply(self.attention_out,self.element_wise_product), 1, name="afm")
                self.AFM = tf.nn.dropout(self.AFM, self.dropout_keep[-1])

            # ________ FM __________
            self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep[-1]) # dropout at the bilinear interactin layer

            # ________ Deep Layers __________
            # self.deep = tf.reduce_sum(nonzero_embeddings, 1)
            self.deep = tf.reshape(nonzero_embeddings,shape=[-1,self.valid_dimension*self.hidden_factor])
            self.deep = tf.nn.dropout(self.deep, self.dropout_keep[0])
            for i in range(len(self.layers)):
                self.deep = tf.add(tf.matmul(self.deep, self.weights['layer_%d' %i]), self.weights['bias_%d'%i]) # None * layer[i] * 1
                if self.batch_norm:
                    self.deep = self.batch_norm_layer(self.deep, train_phase=self.train_phase, scope_bn='bn_%d' %i) # None * layer[i] * 1
                self.deep = self.activation_function(self.deep)
                self.deep = tf.nn.dropout(self.deep, self.dropout_keep[i]) # dropout at each Deep layer

            # _________DeepFM_________
            if self.attention:
                self.concat = tf.concat([self.y_first_order, self.AFM, self.deep], axis=1)
                self.out = tf.add(tf.matmul(self.concat, self.weights["prediction"]), self.weights["bias"])
            else:
                self.concat = tf.concat([self.y_first_order, self.FM, self.deep], axis=1)
                self.out = tf.add(tf.matmul(self.concat, self.weights["prediction"]), self.weights["bias"])

            # Compute the loss.
            if self.loss_type == 'square_loss':
                self.out = tf.sigmoid(self.out)
                if self.lamda_bilinear > 0:
                    self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                else:
                    self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
            elif self.loss_type == 'log_loss':
                self.out = tf.sigmoid(self.out)
                if self.lambda_bilinear > 0:
                    self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weight=1.0, epsilon=1e-07, scope=None) + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                else:
                    self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weight=1.0, epsilon=1e-07, scope=None)

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape() # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" %total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0: # with pretrain
            pretrain_file = '../pretrain/%s_%d/%s_%d' %(args.dataset, args.hidden_factor, args.dataset, args.hidden_factor)
            weight_saver = tf.train.import_meta_graph(pretrain_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, pretrain_file)
                fe, fb, b = sess.run([feature_embeddings, feature_bias, bias])
            all_weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32)
            all_weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32)
            all_weights['bias'] = tf.Variable(b, dtype=tf.float32)
        else: # without pretrain
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.01), name='feature_embeddings')  # features_M * K
            all_weights['feature_bias'] = tf.Variable(tf.random_uniform([self.features_M, 1], 0.0, 1.0), name='feature_bias')  # features_M * 1
            all_weights['bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32,name='bias')  # 1 * 1

        # attention
        if self.attention:
            glorot = np.sqrt(2.0 / (10 + self.hidden_factor))
            all_weights['attention_W'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor,10)), dtype=np.float32, name="attention_W")
            all_weights['attention_b'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, 10)), dtype=np.float32, name="attention_b")
            all_weights['attention_p'] = tf.Variable(
                np.random.normal(loc=0, scale=1, size=(self.hidden_factor)), dtype=np.float32, name="attention_p")
            all_weights['attention_h'] = tf.Variable(
                np.random.normal(loc=0, scale=1, size=(10,)), dtype=tf.float32, name='attention_h')

        # deep layers
        num_layer = len(self.layers)
        input_size = self.hidden_factor * self.valid_dimension
        if num_layer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.layers[0]))
            all_weights['layer_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, self.layers[0])), dtype=np.float32)
            all_weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[0])), dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.layers[i-1] + self.layers[i]))
                all_weights['layer_%d' %i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i-1], self.layers[i])), dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['bias_%d' %i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)  # 1 * layer[i]
	        # prediction layer
            in_size =  self.hidden_factor + self.layers[0]
            glorot = np.sqrt(2.0 / (in_size + 1))
            all_weights['prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(in_size+1, 1)), dtype=np.float32)  # layers[-1] * 1
        else:
            all_weights['prediction'] = tf.Variable(np.ones((self.hidden_factor*2+1, 1), dtype=np.float32))  # hidden_factor * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=None, trainable=True, scope=scope_bn, fused = False)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn, fused = False)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'], self.dropout_keep: self.keep_prob, self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X , Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}

    def shuffle_in_unison_scary(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def train(self, Train_data, Validation_data, Test_data):  # fit a dataset
        # Check Init performance
        a,b,c =[],[],[]
        if self.verbose > 0:
            t2 = time()
            init_train = self.evaluate(Train_data)
            init_valid = self.evaluate(Validation_data)
            init_test = self.evaluate(Test_data)
            print('init,train_loss:',init_train, 'validation_loss:',init_valid, 'test_loss:',init_test)
        #
        # os.remove('./results'+str(self.attention)+'.txt')
        # fi = open('./results'+str(self.attention)+'.txt', 'a')
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X'], Train_data['Y'])
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            c.append(epoch)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()

            # output validation
            train_result = self.evaluate(Train_data)
            valid_result = self.evaluate(Validation_data)
            test_result = self.evaluate(Test_data)
            a.append(valid_result[0])
            b.append(valid_result[1])
            self.train_rmse.append(train_result)
            self.valid_rmse.append(valid_result)
            self.test_rmse.append(test_result)
            if self.verbose > 0 and epoch%self.verbose == 0:
                # print('epoch:',epoch+1, 'train_loss:', train_result, 'validation_loss:',valid_result, 'test_loss:',test_result,file=fi)
                print('epoch:', epoch + 1, 'train_loss:', train_result, 'validation_loss:', valid_result, 'test_loss:',
                      test_result, 'time', time()-t1)

            if self.early_stop > 0 and self.eva_termination(self.valid_rmse):
                #print "Early stop at %d based on validation result." %(epoch+1)
                break
        # fi.close()
        plt.title("RMSE")
        plt.plot(c, a)
        plt.savefig('RMSE_'+str(self.attention)+'_'+str(self.epoch)+'.png')
        plt.show()

        plt.title("AUC")
        plt.plot(c, b)
        plt.savefig('AUC_'+str(self.attention)+'_'+str(self.epoch)+'.png')
        plt.show()


    def eva_termination(self, valid):
        if self.loss_type == 'square_loss':
            if len(valid) > 5:
                if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                    return True
        else:
            if len(valid) > 5:
                if valid[-1] < valid[-2] and valid[-2] < valid[-3] and valid[-3] < valid[-4] and valid[-4] < valid[-5]:
                    return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        feed_dict = {self.train_features: data['X'], self.train_labels: [[y] for y in data['Y']], self.dropout_keep: self.no_dropout, self.train_phase: False}
        predictions = self.sess.run(self.out, feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(data['Y'], (num_example,))

        if self.loss_type == 'square_loss':
            predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
            predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
            RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
            auc = roc_auc_score(y_true, y_pred)
            return [RMSE, auc]
        elif self.loss_type == 'log_loss':
            logloss = log_loss(y_true, y_pred) # I haven't checked the log_loss
            auc = roc_auc_score(y_true, y_pred)
            return [logloss, auc]

if __name__ == '__main__':
    # Data loading
    args = parse_args()
    data = DATA.LoadData(args.path, args.dataset, args.loss_type)

    if args.verbose > 0:
        print("Neural FM: dataset=%s, attention=%s, valid_dimension=%s, hidden_factor=%d, dropout_keep=%s, layers=%s, loss_type=%s, pretrain=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.4f, optimizer=%s, batch_norm=%d, activation=%s, early_stop=%d"
              %(args.dataset, args.attention, args.valid_dimension, args.hidden_factor, args.keep_prob, args.layers, args.loss_type, args.pretrain, args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer, args.batch_norm, args.activation, args.early_stop))
    activation_function = tf.nn.relu
    if args.activation == 'sigmoid':
        activation_function = tf.sigmoid
    elif args.activation == 'tanh':
        activation_function == tf.tanh
    elif args.activation == 'identity':
        activation_function = tf.identity

    # Training
    t1 = time()
    model = NeuralFM(data.features_M, args.attention, args.valid_dimension, args.hidden_factor, eval(args.layers), args.loss_type, args.pretrain, args.epoch, args.batch_size, args.lr, args.lamda, eval(args.keep_prob), args.optimizer, args.batch_norm, activation_function, args.verbose, args.early_stop)
    if args.dataset == 'ml-tag':
        model.train(data.Train_data, data.Validation_data, data.Test_data)
    elif args.dataset == 'company':
        Train, Valid = {}, {}
        folds = list(StratifiedKFold(n_splits=2, shuffle=True,
                                     random_state=2021).split(data.Train_data['X'], data.Train_data['Y']))
        _get = lambda x, l: [x[i] for i in l]
        for i, (train_idx, valid_idx) in enumerate(folds):
            Train['Y'], Train['X'] =  _get(data.Train_data['Y'], train_idx),_get(data.Train_data['X'], train_idx)
            Valid['Y'], Valid['X'] =  _get(data.Train_data['Y'], valid_idx),_get(data.Train_data['X'], valid_idx)
            model.train(Train, Valid, data.Test_data)


    # Find the best validation result across iterations
    best_valid_score = 0
    if args.loss_type == 'square_loss':
        best_valid_score = min(model.valid_rmse)
    elif args.loss_type == 'log_loss':
        best_valid_score = max(model.valid_rmse)
    best_epoch = model.valid_rmse.index(best_valid_score)
    # with open('./data/results_'+str(args.attention)+'.txt', 'a'):
    print('best_epoch:',best_epoch+1, 'train_loss:',model.train_rmse[best_epoch], 'validation_loss:',model.valid_rmse[best_epoch], 'test_loss:',model.test_rmse[best_epoch])
    print('Model training is over!!!!')