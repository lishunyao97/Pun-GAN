# -*- coding= utf-8 -*-
"""
 @version= python2.7
 @author= luofuli
 @time= 2018/3/30 11=03
"""
import sys
import pickle
import random


class BiLSTM_Config(object):
    """配置参数"""

    def __init__(self):
        self.name = 'CAN'

        # config for data
        self.g_len = 40  # max gloss words
        self.c_len_f = 10  # forward context words
        self.c_len_b = 10  # backward context words
        self.expand_type = 0  # 0=无 1=上位hyper，2=下位hypo 3=上+下 4：层次
        self.gloss_is_empty = True  # ** For comparision(GAS_baseline)
        self.back_off_type = 'MFS'  # MFS

        # config for word embeddings
        self.vocab_name = '6B'  # 6B(100200300) 42B(300) 840B(300)
        self.vocab_size = 100000  # 后面会更具具体情况修改，方便索引的
        self.embedding_size = 300
        self.use_pre_trained_emb = False  # ** Debug model
        self.train_word_emb = False
        self.has_pos = False
        self.pos_dim = 50

        # config for model
        self.batch_size = 32  # 32 is too big for all_words
        self.hidden_size = 128
        self.forget_bias = 0.0
        self.keep_prob = 0.5

        # config for lstm cell
        self.dropout = True  # 总开关
        self.state_dropout = True
        self.rnn_type = 'LSTM'  # GRU

        # config for train
        self.n_epochs = 30
        self.lr_start = 0.001 # original lr = 0.001
        self.decay_lr = False
        self.optimizer_type = 'Adam'  # Adagrad
        self.momentum = 0.1
        self.clip_gradients = True  # 是否裁剪梯度 !! 比较影响时间
        self.max_grad_norm = 10  # 裁剪梯度的最大梯度
        self.warm_start = True

        # config for print logs
        self.print_batch = True
        self.evaluate_gap = 100  # ** 每多少轮验证一次val的结果是否有提升,并打印输出
        self.store_log_gap = 10  # **
        self.save_best_model = True
        self.store_run_time = False  # ** 记录每个节点的运行时间（找出最慢的节点）
        self.run_time_epoch = 1   # ** 每隔多少epoch记录一次时间
        self.show_true_result = True  # 是否展示真正的结果（两种计算方法，1.打分函数 2.模型正确率+back_off_result正确率

        # Validation info
        self.sota_score = 0.706
        self.validate = True
        self.min_no_improvement = 5000  # 连续多少个epoch没提高就停止，而不是多少步step

#        with open('../tmp/pos_dic.pkl', 'rb') as f:
#            self.pos_to_id = pickle.load(f)
#            self.pos = self.pos_to_id.keys()

        self.changed_config = {}

    def store_change(self, param, name):
        self.changed_config[name] = param
        return param

    def random_config(self):

        self.hidden_size = self.store_change(random.choice([128, 256, 512]), 'hidden_size')
        self.lr_start = self.store_change(random.choice([0.01, 0.05, 0.001, 0.0001]), 'lr_start')
        self.rnn_type = self.store_change(random.choice(['LSTM', 'GRU']), 'rnn_type')
        self.optimizer_type = self.store_change(random.choice(['Adadelta', 'Adagrad', 'Adam']), 'optimizer_type')
        # self.use_pre_trained_emb = self.store_change(random.choice([False, True]), 'use_pre_trained_emb')

    def grid_search(self):
        for self.hidden_size in [256, 128, 512]:
            self.store_change(self.hidden_size, 'hidden_size')
            for self.lr_start in [0.001, 0.05]:   # 1.0差得惨不忍睹
                self.store_change(self.lr_start, 'lr_start')
                for self.affinity_method in ['general', 'dot_sum']:  # 目前看来两者差不多+1
                    self.store_change(self.affinity_method, 'affinity_method')
                    for self.update_context_encoding in [True, False]:   # 目前来看True更有用,但是5.2发现false更好
                        self.store_change(self.update_context_encoding, 'update_context_encoding')
                        yield self

    def get_grid_search_i(self, run_i):
        total_i = 0
        for conf in self.grid_search():
            total_i += 1

        for i, conf_ in enumerate(self.grid_search()):
            if i == (run_i % total_i):
                return conf_

    def degug_model(self):   # Debug模型的参数，跑通一轮就结束
        self.name = 'CAN_debug'
        self.use_pre_trained_emb = False
        self.evaluate_gap = 1  # 每多少轮验证一次val的结果是否有提升,并打印输出
        self.store_log_gap = 1  #
        self.store_run_time = False  # 记录每个节点的运行时间（找出最慢的节点）
        self.run_time_epoch = 1  # 每隔多少epoch记录一次时间
        self.min_no_improvement = 1  # 连续多少个epoch没提高就停止，而不是多少步step

    def detailed_show(self):
        self.name = 'CAN_detail'
        self.print_batch = True
        self.evaluate_gap = 1  # ** 每多少轮验证一次val的结果是否有提升,并打印输出
        self.store_log_gap = 1  # **
        self.save_best_model = True
        self.store_run_time = True  # ** 记录每个节点的运行时间（找出最慢的节点）
        self.run_time_epoch = 1  # ** 每隔多少epoch记录一次时间
        self.show_true_result = True  # 是否展示真正的结果（两种计算方法，1.打分函数 2.模型正确率+back_off_result正确率


if __name__ == "__main__":
    config = BiLSTM_Config()
    config.random_config()
    print config.hidden_size
    print config.changed_config

    print vars(config)

    i = 0
    for conf in config.grid_search():
        i += 1
        print i
