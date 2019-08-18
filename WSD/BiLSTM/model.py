# -*- coding: utf-8 -*-

# from data_on import *
import tensorflow as tf
import numpy as np
rnn = tf.contrib.rnn
# global_initializer = tf.contrib.layers.xavier_initializer()
global_initializer = tf.random_uniform_initializer(-0.1, 0.1)  # 随机均匀分布
lstm_initializer = tf.orthogonal_initializer()


class BiLSTM(object):
    def __init__(self, config, n_senses_from_target_id, init_word_vecs=None):
        self.name = 'BiLSTM'

        # config for data
        self.batch_size = batch_size = config.batch_size
        self.max_n_sense = max_n_sense = config.max_n_sense
        self.g_len, self.c_len_f, self.c_len_b = g_len, c_len_f, c_len_b = config.g_len, config.c_len_f, config.c_len_b
        # config for model
        self.embedding_size = embedding_size = config.embedding_size
        self.hidden_size = hidden_size = config.hidden_size
        self.context_dim = context_dim = 2 * hidden_size  # bi-lstm
        # config for train
        self.lr_start = lr_start = config.lr_start  # 0.2
        lr_decay_factor = 0.96
        lr_min = 0.01

        self.is_training = tf.placeholder(tf.bool, name='is_training')  # 不能直接用if is_training
        self.inputs = tf.placeholder(tf.int32, shape=[batch_size, c_len_f + c_len_b + 1], name='inputs')
        self.poss = tf.placeholder(tf.int32, shape=[batch_size, c_len_f + c_len_b + 1], name='poss')
        self.target_ids = tf.placeholder(tf.int32, shape=[batch_size], name='target_ids')
        self.sense_ids = tf.placeholder(tf.int32, shape=[batch_size], name='sense_ids')


        # 备用（如果不用到词典知识，这儿可能暂时用不到）
        self.glosses = tf.placeholder(tf.int32, shape=[batch_size, max_n_sense, g_len], name='glosses')
        self.glosses_lenth = tf.placeholder(tf.int32, shape=[batch_size, max_n_sense], name='glosses_lenth')
        self.sense_mask = tf.placeholder(tf.float32, shape=[batch_size, max_n_sense, context_dim], name='sense_mask')
        # add for PUNGAN
        self.fake_s = tf.placeholder(tf.float32, shape=[batch_size], name='fakes')
        self.real_s = tf.placeholder(tf.float32, shape=[batch_size], name='reals')
        self.labeled_s = tf.placeholder(tf.float32, shape=[batch_size], name='labeleds')
        # end for PUNGAN
        tot_n_senses = sum(n_senses_from_target_id.values())
        tot_n_target_words = len(n_senses_from_target_id)
        n_senses_sorted_by_target_id = [n_senses_from_target_id[target_id] for target_id
                                        in range(len(n_senses_from_target_id))]
        n_senses_sorted_by_target_id_tf = tf.constant(n_senses_sorted_by_target_id, tf.int32)

        # In[4]: np.cumsum(np.append([0],[1,2,3]))
        # Out[4]: array([0, 1, 3, 6]) * 2 = array([0, 2, 6, 12])
        # all is one dimension: [total_target_word * context_dim]
        # W_starts: [0, s1*h, ..., s_(n-1)*h] : si is sense 排列顺序, n is number of target word, and h is hidden size

        _W_starts = (np.cumsum(np.append([0], n_senses_sorted_by_target_id)) * context_dim)[:-1]
        _W_lenghts = np.array(n_senses_sorted_by_target_id) * context_dim
        W_starts = tf.constant(_W_starts, tf.int32)
        W_lengths = tf.constant(_W_lenghts, tf.int32)
        _b_starts = (np.cumsum(np.append([0], n_senses_sorted_by_target_id)))[:-1]
        _b_lengths = np.array(n_senses_sorted_by_target_id)
        b_starts = tf.constant(_b_starts, tf.int32)
        b_lengths = tf.constant(_b_lengths, tf.int32)

        with tf.variable_scope('target_params', initializer=global_initializer):
            W_targets = tf.get_variable('W_targets', [tot_n_senses * context_dim], dtype=tf.float32)
            b_target = tf.get_variable('b_target', [tot_n_senses], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
            m_ratio = tf.get_variable('r_target', [tot_n_target_words], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.5))  # memory得分所占比重，每次更新后限制在[0，1]之间

        self.keep_prob = keep_prob = tf.cond(tf.equal(self.is_training, tf.constant(True)),
                                             lambda: tf.constant(config.keep_prob),
                                             lambda: tf.constant(1.0))  # val or test: 1.0 means no dropout

        loss = tf.Variable(0., trainable=False, name='total_loss')
        d_loss_class = tf.Variable(0., trainable=False, name='d_loss_class')
        d_loss_real = tf.Variable(0., trainable=False, name='d_loss_real')
        d_loss_fake = tf.Variable(0., trainable=False, name='d_loss_fake')
        n_correct = tf.Variable(0, trainable=False, name='n_correct')
        self.global_step = global_step = tf.Variable(0, trainable=False)
        self.predictions = tf.Variable(tf.zeros([batch_size], dtype=tf.int32), trainable=False)
        self.correct = tf.Variable(tf.zeros([batch_size], dtype=tf.int32), trainable=False)
        self.reward = tf.Variable(tf.zeros([batch_size, max_n_sense], dtype=tf.float32), trainable=False)


        def lstm_cell(num_units):
            # cell = rnn.NASCell(num_units)
            if config.rnn_type == 'GRU':
                cell = rnn.GRUCell(num_units, kernel_initializer=lstm_initializer)
            else:  # default
                # cell = rnn.BasicLSTMCell(num_units, forget_bias= config.forget_bias)
                cell = rnn.LSTMCell(num_units, initializer=lstm_initializer, forget_bias=config.forget_bias)

            if config.dropout:
                if tf.__version__ == '1.2.0' and config.state_dropout:
                    cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob, state_keep_prob=keep_prob)
                # 经测试： state_dropout=keep_prob提升效果明显，variational_recurrent=False对结果来说更好
                else:
                    print('~~~NO USING：state_ DropoutWrapper')
                    cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
                    # 经测试： state_dropout=keep_prob提升效果明显，variational_recurrent=False对结果来说更好
            return cell

        with tf.device('/cpu:0'):
            with tf.variable_scope('word_emb', initializer=global_initializer):
                if config.use_pre_trained_emb:
                    word_emb = tf.get_variable('word_emb', initializer=init_word_vecs, trainable=config.train_word_emb)
                else:
                    word_emb = tf.get_variable('word_emb', shape=[config.vocab_size, embedding_size], trainable=True)
#            with tf.variable_scope('pos_emb'):
#                pos_emb = tf.get_variable('pos_emb', shape=[len(config.pos), config.pos_dim], trainable=True)

        with tf.variable_scope('context', initializer=lstm_initializer):
            c_cell_fw = lstm_cell(hidden_size)  # forward
            c_cell_bw = lstm_cell(hidden_size)  # backward
            c_word_embedding = tf.nn.embedding_lookup(word_emb, self.inputs)  # [batch_size, c_len, dim]
            if config.has_pos:
                inputs_pos = tf.nn.embedding_lookup(pos_emb, self.poss)  # [batch_size, c_len, pos_dim]
                inputs_c = tf.concat([c_word_embedding, inputs_pos], axis=2)
            else:
                inputs_c = c_word_embedding
            inputs_c_ = tf.nn.dropout(inputs_c, keep_prob)
            c_outputs, c_final_state = tf.nn.bidirectional_dynamic_rnn(c_cell_fw,
                                                                       c_cell_bw,
                                                                       inputs_c_,
                                                                       time_major=False,
                                                                       dtype=tf.float32)
            c_lstm_enc = tf.concat([c_outputs[0][:, c_len_f - 1, :], c_outputs[1][:, c_len_f + 1, :]], 1)
            c_encoding = tf.concat(c_outputs, 2)  # [batch_size, c_len, 2*hidden_size]
            # c_reshape = tf.reshape([-1, 2*hidden_size])

        # prediction
        with tf.variable_scope('score'):
            c_hidden_state = tf.split(c_lstm_enc, batch_size, 0)

            target_ids = tf.split(self.target_ids, batch_size, 0)
            sense_ids = tf.split(self.sense_ids, batch_size, 0)
            labeled_s = tf.split(self.labeled_s, batch_size, 0)
            real_s = tf.split(self.real_s, batch_size, 0)
            fake_s = tf.split(self.fake_s, batch_size, 0)
            # m_ratio = tf.split(tf.clip_by_value(m_ratio, 0, 1), tot_n_target_words, 0) # the ratio of memory
            self.m_ratio = m_ratio = tf.clip_by_value(m_ratio, 0, 1)  # the ratio of cosine similarity based on knowledge

            # make predictions for all instances in batch
            for i in range(batch_size):
                target_id = target_ids[i]
                sense_id = sense_ids[i]
                labeled = labeled_s[i]
                real = real_s[i]
                fake = fake_s[i]
                n_sense = tf.squeeze(tf.slice(n_senses_sorted_by_target_id_tf, target_id, [1]))  # target word 义项个数
                c_i = c_hidden_state[i]   # a_ = tf.slice(hidden_state, [i, 0], [1, 4*hidden_size])  # [1, 4*hidden_size]

                # add a fully-connected layer
                # one = tf.constant(1, tf.int32, [1])
                W = tf.slice(W_targets, tf.slice(W_starts, target_id, [1]), tf.slice(W_lengths, target_id, [1]))
                W = tf.reshape(W, [n_sense, 2 * hidden_size])  # [n_sense, 2*hidden_size] # 为了让静态图编译通过
                b = tf.slice(b_target, tf.slice(b_starts, target_id, [1]), tf.slice(b_lengths, target_id, [1]))
                logits = tf.matmul(c_i, W, False, True) + b  # [1, n_sense]
                # add for PUNGAN
                real_class_logits = logits
                fake_class_logits = 0.
                mx = tf.reduce_max(real_class_logits, 1, keep_dims=True)
                stable_real_class_logits = real_class_logits - mx
                gan_logits = tf.log(tf.reduce_sum(tf.exp(stable_real_class_logits), 1)) + tf.squeeze(mx) - fake_class_logits
                # end for PUNGAN
                logits = tf.nn.softmax(logits)
                zero_pad = tf.zeros([1, max_n_sense - n_sense], dtype=tf.float32)
                self.reward = tf.scatter_update(self.reward, tf.constant(i, shape=[1]), tf.concat([logits, zero_pad], axis=1))
                # update corresponding results in the batch_size list
                predicted_sense = tf.arg_max(logits, 1, name='prediction')

                predicted_sense = tf.cast(predicted_sense, tf.int32)  # tf.arg_max defaut return int64
                self.predictions = tf.scatter_update(self.predictions, tf.constant(i, shape=[1]), predicted_sense)

                # tf.equal return bool, tf.cast:bool=>
                n_correct += tf.squeeze(tf.cast(tf.equal(sense_id, predicted_sense), tf.int32))
                self.correct = tf.scatter_update(self.correct, tf.constant(i, shape=[1]),
                                                 tf.cast(tf.equal(sense_id, predicted_sense), tf.int32))
                # loss calculate
                # add for PUNGAN
                d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits,
                                                labels=tf.ones_like(gan_logits)))
                d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits,
                                                labels=tf.zeros_like(gan_logits)))
                d_loss_class = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=sense_id))
                loss += tf.multiply(labeled, d_loss_class) + tf.multiply(real, d_loss_real) + tf.multiply(fake, d_loss_fake)
                # end for PUNGAN
                # if i == batch_size - 1:
                #     tf.summary.histogram('logits', logits)
                #     tf.summary.histogram('W_targets', W_targets)
                #     tf.summary.histogram('b_target', b_target)

        with tf.variable_scope('prediction'):
            self.loss_op = tf.div(loss, batch_size)
            self.accuracy_op = tf.div(tf.cast(n_correct, tf.float32), batch_size)

            # Summaries
            # tf.summary.scalar('loss', self.loss_op)  # 每个batch采集一次
            # tf.summary.scalar('accuracy', self.accuracy_op)
            # self.summary_op = tf.summary.merge_all()

        with tf.variable_scope('train'):
            print 'Trainable Variables'
            tvars = tf.trainable_variables()
            for tvar in tvars:
                print tvar.name
            # Gradients
            if config.decay_lr:
                self.decay_lr = tf.train.exponential_decay(lr_start, global_step, 60, lr_decay_factor)
                self.lr = tf.maximum(lr_min, self.decay_lr)  # 衰减学习率
                optimizer = tf.train.MomentumOptimizer(self.lr, config.momentum)  # 确定优化方法
            else:  # 自适应学习率的方法
                self.lr = tf.Variable(lr_start, trainable=False)

                if config.optimizer_type == 'Adadelta':
                    optimizer = tf.train.AdadeltaOptimizer(self.lr)
                elif config.optimizer_type == 'Adagrad':
                    optimizer = tf.train.AdagradOptimizer(self.lr)
                # elif config.optimizer_type == "AMSGrad":
                #     optimizer = AMSGrad(self.lr)
                else:
                    optimizer = tf.train.AdamOptimizer(self.lr)  # AdamOptimizer自身有学习率衰减

            # Update Parameters
            if config.clip_gradients:  # 计算梯度并剪裁
                print('~~~Using clip')
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_op, tvars), config.max_grad_norm)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
            else:   # 不裁剪梯度，快一些
                print('~~~No using clip')
                self.train_op = optimizer.minimize(loss, global_step=global_step)
