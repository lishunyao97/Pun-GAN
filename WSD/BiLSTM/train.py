# -*- coding: utf-8 -*-
"""
 @version: python2.7
 @author: luofuli
 @time: 2018/4/14 19:26
"""

import glob
import os
import sys
import time
import tensorflow as tf
import pickle
import random
import heapq
# print sys.path  # 一般是从该文件下的所在的目录中查询，所以找不到MemNN和utils、postprocessing这些目录
# sys.path.insert(0, "..")  # 保证下面的import 其他文件夹的类成功
PUNGAN_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 因为需要在pun generation文件夹下运行，所以把相对路径改为绝对路径
from utils.data import *
from utils.glove import *
from utils import path
from utils import store_result, score
from config import BiLSTM_Config
from nltk.corpus import wordnet as wn
import model

_path = path.WSD_path()
config = BiLSTM_Config()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def feed_dict(model, batch_data, is_training=True):
    [xf, xb, xfb, pf, pb, pfb, target_ids, sense_ids, instance_ids, glosses_ids, synsets_ids,
     glosses_lenth, synsets_lenth, sense_mask, fakes, reals, labeleds] = batch_data

    feeds = {
        model.is_training: is_training,  # 控制dropout比例
        model.inputs: xfb,
        model.poss: pfb,
        model.target_ids: target_ids,
        model.sense_ids: sense_ids,
        model.glosses: glosses_ids, # 用不到gloss
        model.glosses_lenth: glosses_lenth, # 用不到gloss
        model.sense_mask: sense_mask,
        model.fake_s: fakes,
        model.real_s: reals,
        model.labeled_s: labeleds
    }
    return feeds


def evaluate(eval_data, model, step, dict_data,word_to_id ,session, target_id_to_word, target_sense_to_id, target_id_to_sense_id_to_sense, global_t0, back_off_result):
    total_loss = []
    total_acc = []
    total_pred = []
    total_correct = []

    for batch_id, batch_data in enumerate(
            batch_generator(True, config.batch_size, eval_data, dict_data, word_to_id['<pad>'],
                            config.c_len_f, config.c_len_b, pad_last_batch=True)):

        feeds = feed_dict(model, batch_data, is_training=False)
        acc, loss, pred, correct = session.run([model.accuracy_op, model.loss_op, model.predictions, model.correct],
                                               feeds)
        total_loss.append(loss)
        total_acc.append(acc)

        # 映射到真正的答案
        for i, predicted_sense_id in enumerate(pred):
            if batch_id * config.batch_size + i < len(eval_data):
                instance_id = batch_data[8][i]  # 8 is index of instance_ids
                predicted_sense = target_id_to_sense_id_to_sense[batch_data[6][i]][predicted_sense_id]
                total_pred.append([instance_id, predicted_sense])
                total_correct.append([instance_id, correct[i]])

    # 先定义summary，后面再add也可以 summary.value.add(tag=, simple_value=)
    # tag='accuracy' 保证与训练时候的variable name一致，这样子才能在同一个图上
    prefix = re.findall("^(.*/)", model.accuracy_op.name)
    prefix = prefix[0] if len(prefix) == 1 else ''
    # summary_acc_average = tf.Summary(value=[tf.Summary.Value(tag=prefix + 'accuracy', simple_value=np.mean(total_acc))])
    # summary_val_writer.add_summary(summary_acc_average, global_step=step)
    # summary_loss_average = tf.Summary(value=[tf.Summary.Value(tag=prefix + 'loss', simple_value=np.mean(total_loss))])
    # summary_val_writer.add_summary(summary_loss_average, global_step=step)

    if config.show_true_result:
        # 两种计算方法，1.打分函数
        tmp_result_path = PUNGAN_ROOT_PATH + '/WSD/tmp/results/tmp-%s.txt' % global_t0
        score.write_result(total_pred, back_off_result, tmp_result_path, print_logs=False)
        p_true, r_true, val_true1 = score.score_one(tmp_result_path,
                                                        gold_key_path=_path.ALL_WORDS_TEST_KEY_PATH.format('ALL'),
                                                        unsocre_dataset=_path.ALL_WORDS_VAL_DATASET)

        # 2.模型正确率 + back_off_result正确率。第2个有偏差，因为有一些sense不在测试集合里，就被划为0类，造成模型正确率偏高，
        # 同时由于测量不到multi-sense，所以结果偏低，最终结果偏低偏高不知道呀不知道）
        # attempt_n, correct_n = score.score_for_back_off_result(config.back_off_type)
        # correct_list = [item[1] for item in total_correct]
        # val_true2 = (sum(correct_list) + correct_n) / (len(correct_list) + attempt_n) # total_correct避免最后一个batch的影响
        val_true = [val_true1, p_true, r_true, 0]
    else:
        val_true = [0, 0]

    return np.mean(total_acc), np.mean(total_loss), total_pred, total_correct, val_true

def evaluate_pungan(eval_data,dict_data,word_to_id, model, session, target_id_to_word, target_sense_to_id):
    print('enter eval pungan')
    total_loss = []
    total_acc = []
    total_pred = []
    total_correct = []

    total_target_word = []
    total_senses = []
    total_reward = []
    for batch_id, batch_data in enumerate(batch_generator(False, config.batch_size, eval_data, dict_data, word_to_id['<pad>'],
                            config.c_len_f, config.c_len_b, pad_last_batch=True)):

        feeds = feed_dict(model, batch_data, is_training=False)

        acc, loss, pred, correct, reward = session.run([model.accuracy_op, model.loss_op, model.predictions, model.correct, model.reward], feeds)


        target_word = [target_id_to_word[i] for i in batch_data[6]]
        senses = [target_sense_to_id[i] for i in batch_data[6]]
        print('target_word length',len(target_word))
        print('senses length',len(senses))
        print('reward length',len(reward))
        total_target_word.extend(target_word)
        total_senses.extend(senses)
        total_reward.extend(reward)

    return total_target_word, total_senses, total_reward

def test(model, test_data,dict_data,word_to_id,session, target_id_to_word, target_sense_to_id, test_data_index, senses_input, result_path=None):
    print('enter test')
    target_word, senses, reward = evaluate_pungan(test_data, dict_data, word_to_id, model,session, target_id_to_word, target_sense_to_id)
    print('[target_word]',target_word)
    print('[senses]',senses)
    print('target word length', len(target_word))
    print('senses lenght', len(senses))
    print('reward length', len(reward))
    print('test_data_index', len(test_data_index))
    selected_sense_pair = []
    for i, pair in enumerate(senses_input):
        if i in test_data_index:
            selected_sense_pair.append(pair)
    print('[selected_sense_pair]', selected_sense_pair)
    return_reward = []
    reward_detail = []
    recall5 = []
    for i in range(len(selected_sense_pair)):
        # 此处计算reward
        sense_pair = selected_sense_pair[i]
        # synset = wn.lemma_from_key(sense_pair[0]).synset()
        # s = synset.name()
        # targetw = '#'.join(s.split('.')[:2])
        # if targetw not in target_word:
        #     continue
        possible_senses = senses[i]
        cnt_senses = len(possible_senses)
        answer1, answer2 = sense_pair[0], sense_pair[1]
        print('possible_senses', possible_senses)
        print('answer1', answer1)
        print('answer2', answer2)
        tmp_reward = reward[i].tolist()[:cnt_senses]
        if answer1 in possible_senses and answer2 in possible_senses:
            answer1_p, answer2_p = tmp_reward[possible_senses[answer1]], tmp_reward[possible_senses[answer2]]
            # calcu_reward = alpha * (1 / (abs(answer1_p - answer2_p) + 1)) + (1-alpha) * (answer1_p + answer2_p) - 1.0
            calcu_reward = (answer1_p + answer2_p)/(abs(answer1_p - answer2_p) + 1)
        else:
            calcu_reward = -1
        return_reward.append(calcu_reward)
        if answer1 in possible_senses and answer2 in possible_senses:
            if cnt_senses <= 2:
                recall5.append(1.0)
            else:
                max_num_index_list = map(tmp_reward.index, heapq.nlargest(2, tmp_reward))
                rectmp = 0.0
                if possible_senses[answer1] in max_num_index_list:
                    rectmp += 0.5
                if possible_senses[answer2] in max_num_index_list:
                    rectmp += 0.5
                recall5.append(rectmp)
            reward_detail.append((answer1_p, answer2_p))
        else:
            reward_detail.append((-1, -1))
            recall5.append(-1)
        #print('calcu_reward', calcu_reward)
    return return_reward, reward_detail, recall5

def train(model, train_data, val_data, dict_data, word_to_id, session, target_id_to_word, target_sense_to_id, target_id_to_sense_id_to_sense, global_t0, back_off_result, saver, model_save_prefix, tag, train_dataset, val_dataset):
    total_batch = 0  # Same to global step, 不过global step需要添加到ops然后返回得到，两种方式都OK
    n_batch = len(train_data) / config.batch_size
    best = {
        'acc_val': 0.0,
        'acc_train': 0.0,
        'i': 0,
    }

    for i in range(1, config.n_epochs + 1):
        print '::: EPOCH: %d :::' % i
        for batch_id, batch_data in enumerate(
                batch_generator(True, config.batch_size, train_data, dict_data, word_to_id['<pad>'],
                                config.c_len_f, config.c_len_b, pad_last_batch=False)):
            t1 = time.time()
            total_batch += 1

            feeds = feed_dict(model, batch_data, is_training=True)

            if config.store_run_time and (total_batch % (n_batch * config.run_time_epoch) == 0 or total_batch == 1):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                acc_train, loss_train, correct, pred, step, lr, _ = \
                    session.run([model.accuracy_op, model.loss_op, model.correct, model.predictions, model.global_step,
                                 model.lr, model.train_op], feeds,
                                options=run_options,
                                run_metadata=run_metadata)
                print('Add run metadata at step:%s' % step)

            else:
                acc_train, loss_train, correct, pred, step, lr, _ = \
                    session.run([model.accuracy_op, model.loss_op, model.correct, model.predictions, model.global_step,
                                 model.lr, model.train_op], feeds)

            if step % config.evaluate_gap == 0:  # 每print_gap轮输出在训练集和验证集上的性能
                acc_val, loss_val, pred_val, total_correct, val_true = evaluate(val_data, model, step,dict_data,word_to_id, session, target_id_to_word, target_sense_to_id, target_id_to_sense_id_to_sense,global_t0,back_off_result)
                if val_true[0] > best['acc_val']:   # acc_val => val_true[0]换成真正的acc_val
                # if acc_val > best['acc_val']:
                    best['i'] = step
                    best['acc_val'] = val_true[0]
                    # best['acc_val'] = acc_val
                    best['acc_train'] = acc_train
                    prob_best_result = [pred_val[:]]
                    best_result = pred_val[:]  # 避免后面result.extend影响prob_best_result
                    best_correct = total_correct[:]

                    if best['acc_val'] >= config.sota_score:  # 存储sota的model
                        saver.save(session, sota_model_save_prefix,
                                   global_step=global_t0)  # path: model_save_prefix-global_step

                    if config.save_best_model:  # 存储普通的model需要判断
                        saver.save(session, model_save_prefix,
                                   global_step=global_t0)  # path: model_save_prefix-global_step
                else:
                    try:
                        prob_best_result.append(pred_val[:])
                    except:
                        prob_best_result = [pred_val[:]]
                    prob_best_result = prob_best_result[-100:]  # 最多留100个

                if config.print_batch:
                    print('Epoch: %s,step:%s (batch:%s/%s)' % (i, step, batch_id + 1, n_batch))
                    msg = '%s--->\tacc:%.5f\tloss:%.5f\tlr:%.5f'
                    print(msg % ('Train', acc_train, loss_train, lr))
                    print(msg % ('Test', acc_val, loss_val, lr))
                    if config.show_true_result:
                        print('True--->\tac1:%.5f(P:%.5f, R:%.5f)\tac2::%.5f\tbest:%.5f at step %d' %
                              (val_true[0], val_true[1], val_true[2],val_true[-1], best['acc_val'], best['i']))

                    if step == config.evaluate_gap:
                        print ('%d step used time: %d' % (config.evaluate_gap, time.time() - t1))
                        print('Time per EPOCH (estimate): %.3f hour' %
                              ((time.time() - t1) * n_batch / (config.evaluate_gap * 3600)))

                if loss_train > 100:
                    score.store_params(PUNGAN_ROOT_PATH + '/WSD/tmp/big_loss_param.txt', params=config.changed_config)
                    return 0

            # early stop
            if (step - best['i']) / n_batch >= config.min_no_improvement:
                print('**>> Early stop <<**')
                store_result.save_result_and_score(tag, train_dataset, val_dataset, back_off_result, best_result,
                                                   prob_best_result, best, config, PUNGAN_ROOT_PATH + '/WSD/tmp/CAN_score.txt', print_logs=True)
                # score.write_result(best_correct, [], path='../tmp/results/correct.txt')
                print_best(best)
                return 0

    store_result.save_result_and_score(tag, train_dataset, val_dataset, back_off_result, best_result, prob_best_result,
                                       best, config, PUNGAN_ROOT_PATH + '/WSD/tmp/CAN_score.txt', print_logs=True)
    print_best(best)

def train_rl(model, train_data, val_data, dict_data, word_to_id, session, target_id_to_word, target_sense_to_id, target_id_to_sense_id_to_sense, global_t0, back_off_result, saver, model_save_prefix, tag, train_dataset, val_dataset):
    print('train_rl len train_data', len(train_data))
    n_batch = len(train_data) / config.batch_size
    best = {
        'acc_val': 0.0,
        'acc_train': 0.0,
        'i': 0,
    }

    for i in range(1, 2):
        print '::: EPOCH: %d :::' % i
        for batch_id, batch_data in enumerate(
                batch_generator(True, config.batch_size, train_data, dict_data, word_to_id['<pad>'],
                                config.c_len_f, config.c_len_b, pad_last_batch=False)):
            t1 = time.time()
            feeds = feed_dict(model, batch_data, is_training=True)
            acc_train, loss_train, correct, pred, step, lr, _ = session.run([model.accuracy_op, model.loss_op, model.correct, model.predictions, model.global_step, model.lr, model.train_op], feeds)
            saver.save(session, model_save_prefix, global_step=global_t0)  # path: model_save_prefix-global_step

            acc_val, loss_val, pred_val, total_correct, val_true = evaluate(val_data, model, step,dict_data,word_to_id, session, target_id_to_word, target_sense_to_id, target_id_to_sense_id_to_sense,global_t0,back_off_result)
            best['i'] = step
            best['acc_val'] = val_true[0]
            best['acc_train'] = acc_train
            prob_best_result = [pred_val[:]]
            best_result = pred_val[:]  # 避免后面result.extend影响prob_best_result
            best_correct = total_correct[:]
            if config.print_batch:
                print('Epoch: %s,step:%s (batch:%s/%s)' % (i, step, batch_id + 1, n_batch))
                msg = '%s--->\tacc:%.5f\tloss:%.5f\tlr:%.5f'
                print(msg % ('Train', acc_train, loss_train, lr))
                print(msg % ('Test', acc_val, loss_val, lr))
                with open(PUNGAN_ROOT_PATH + '/acc_val.txt', 'a+') as fw:
                    fw.write('acc_val:%.5f at step %d\n' % (val_true[0], best['i']))
                if config.show_true_result:
                    print('True--->\tac1:%.5f(P:%.5f, R:%.5f)\tac2::%.5f\tbest:%.5f at step %d' %
                          (val_true[0], val_true[1], val_true[2],val_true[-1], best['acc_val'], best['i']))

                if step == config.evaluate_gap:
                    print ('%d step used time: %d' % (config.evaluate_gap, time.time() - t1))
                    print('Time per EPOCH (estimate): %.3f hour' %
                           ((time.time() - t1) * n_batch / (config.evaluate_gap * 3600)))
    with open(PUNGAN_ROOT_PATH + '/acc_val.txt', 'a+') as fw:
        fw.write('\n')


def print_best(best):
    n_batch = len(train_data) / config.batch_size

    print('-------------------------------------')
    print('Best train accuracy: %.4f' % best['acc_train'])
    print('Best val   accuracy: %.4f' % best['acc_val'])
    print('Step: %d   (Epoch: %d)' % (best['i'], best['i'] / n_batch))
    if best['acc_val'] > config.sota_score:
        print('Save Best model in: %s-%s' % (sota_model_save_prefix, global_t0))
    else:
        print('Save Best model in: %s-%s' % (model_save_prefix, global_t0))
    print('-------------------------------------')



def run_main(test_data, senses_input, wiki_unlabel, concatenate_file):
    global_t0 = int(time.time())
    Model = model.BiLSTM

    print('python train.py [x]')  # x可有可无，具体看下面的代码，没有的情况就进入debug参数组模式
    '''
    if len(sys.argv) > 1:
        run_type = int(sys.argv[1])
        if run_type == -1:
            print('Debug model!!')
            config.degug_model()   # 快速debug
            global_t0 = -1
        elif run_type == 0:
            print('Show Detail model')
            config.detailed_show()  # 存储模型+log
        elif 100 < run_type <= 500:
            print('Grid search model')
            config.get_grid_search_i(run_type)    # 网格搜索，细调参
        else:
            config.random_config()  # random search to find best parameters
    else:   # 直接
    '''
    config.degug_model()
    global_t0 = 0

    tag = '%s-%s' % (global_t0, config.name)
    print "time+model.name: " + tag
    print('Changed config')
    print config.changed_config
    print('All config')
    print vars(config)

    # Load data
    # ======================================================================================================================

    print('Loading all-words task data...')
    train_dataset = _path.ALL_WORDS_TRAIN_DATASET[0]  # ['semcor', 'semcor+omsti']
    print 'train_dataset: ' + train_dataset
    val_dataset = _path.ALL_WORDS_TEST_DATASET[
        0]  # ['ALL','senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015']
    config.has_pos = False  # all-words task has labeled with pos
    print 'val_dataset: ' + val_dataset
    test_dataset = _path.ALL_WORDS_TEST_DATASET[0]  # ALL
    print 'test_dataset: ' + test_dataset
    # print('Load lexical sample task data...')
    # dataset_i = 1 #int(sys.argv[1]) #    ## ['senseval2_LS', 'senseval3_LS']
    # train_dataset = _path.LS_DATASET[dataset_i]
    # val_dataset = _path.LS_DATASET[dataset_i]
    # config.has_pos = False  # lexical example task don't have pos
    # print 'val_dataset: '+val_dataset

    train_data_labeled = load_train_data(train_dataset)
    train_data_unlabeled = wiki_unlabel
    train_data_fake = concatenate_file
    train_data = train_data_labeled + train_data_unlabeled + train_data_fake
    val_data = load_val_data(val_dataset)
    #test_data = load_test_data(test_dataset) # 需要修改
    print 'train_data[0]', train_data[0]
    print 'test_data[0]', test_data[0]
    print 'test_data len', len(test_data)
    print 'Dataset size (train/test): %d / %d' % (len(train_data), len(val_data))

    back_off_result = []
    if train_dataset in _path.ALL_WORDS_TRAIN_DATASET:
        val_data_lenth_pre = len(val_data)
        train_data, val_data, target_words, back_off_result = \
            data_postprocessing(train_dataset, val_dataset, train_data, val_data, config.back_off_type)
        test_data, test_data_index, _ = test_data_postprocessing(train_dataset, target_words, test_data)
        print 'Dataset size (train/test): %d / %d' % (len(train_data), len(val_data))
        print '***Using back-off instance: %d' % (len(back_off_result))
        missed = val_data_lenth_pre - (len(val_data) + len(back_off_result))
        print '***Missing instance(not in MFS/FS): %d/%d = %.3f' % (
            (missed, val_data_lenth_pre, float(missed) / val_data_lenth_pre))

    # ===build vocab utils
    word_to_id = build_vocab(train_data)
    # with open(PUNGAN_ROOT_PATH + '/WSD/BiLSTM/word_to_id.pickle','rb') as f:
    #     word_to_id = pickle.load(f)
    config.vocab_size = len(word_to_id)
    target_word_to_id, target_sense_to_id, n_words, n_senses_from_target_id = build_sense_ids(train_data)
    print('target_word_to_id.len',len(target_word_to_id))
    print('target_word_to_id', target_word_to_id)
    print('target_sense_to_id.len',len(target_sense_to_id))
    print('target_sense_to_id', target_sense_to_id)
    print 'Vocabulary size: %d' % len(word_to_id)
    print 'Target word: %d' % len(target_word_to_id)
    tot_n_senses = sum(n_senses_from_target_id.values())
    print 'Avg n senses per target word: ' + str(float(tot_n_senses) / len(n_senses_from_target_id))
    with open(PUNGAN_ROOT_PATH + '/WSD/tmp/target_word.txt', 'w') as f:
        for word, id in target_word_to_id.items():
            f.write('{}\t{}\n'.format(word, id))

    # ===make numericR
    # train_data_labeled = convert_to_numeric(train_data_labeled, word_to_id, target_word_to_id,
    #                                 target_sense_to_id, n_senses_from_target_id, data_type="labeled")
    # train_data_unlabeled = convert_to_numeric(train_data_unlabeled, word_to_id, target_word_to_id,
    #                                 target_sense_to_id, n_senses_from_target_id, data_type="unlabeled")
    # train_data_fake = convert_to_numeric(train_data_fake, word_to_id, target_word_to_id,
    #                                     target_sense_to_id, n_senses_from_target_id, data_type="fake")
    # train_data = train_data_labeled # for pretrain
    #train_data = train_data_labeled + train_data_unlabeled + train_data_fake # 1:1:1 for RL
    # val_data = convert_to_numeric(val_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id, ignore_sense_not_in_train=True)

    print('test_data.shape',len(test_data))
    test_data, ids_pungan = convert_to_numeric_pungan(test_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id, ignore_sense_not_in_train=False)
    print('test_data.shape',len(test_data))
    target_id_to_word = {id: word for (word, id) in target_word_to_id.iteritems()}
    target_id_to_sense_id_to_sense = [{sense_id: sense for (sense, sense_id) in sense_to_id.iteritems()} for
                                      (target_id, sense_to_id) in enumerate(target_sense_to_id)]

    # get dic and make numeric
    # train_target_words = target_word_to_id.keys()  # 两者都可以 都是表示train和test重合部分的词
    gloss_dict, synset_dict = load_dictionary(train_dataset, target_word_to_id, config.expand_type)
    mask_size = 2 * config.hidden_size
    dict_data, config.max_n_sense = bulid_dictionary_id(gloss_dict, synset_dict, target_sense_to_id, word_to_id,
                                                        word_to_id['<pad>'], mask_size, config.g_len,
                                                        is_empty=config.gloss_is_empty)
    if config.use_pre_trained_emb:
        print('Load pre-trained word embedding...')
        init_emb = fill_with_gloves(word_to_id, config.embedding_size, config.vocab_name)
    else:
        init_emb = None

    # Training
    # ======================================================================================================================

    # === Initial model
    model0 = Model(config, n_senses_from_target_id, init_emb)

    tf.constant([1, 2], name='constant-test')

    # === Create session
    # session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as session:  # limit gpu memory; don't pre-allocate memory; allocate as-needed
        session.run(tf.global_variables_initializer())

        # === Start training

        print("Configuring TensorBoard and Saver")
        '''
        # ===Configuring TensorBoard(Summary)
        # train summary
        train_log_dir = PUNGAN_ROOT_PATH + '/WSD/tmp/tf.log/CAN/train-%s' % global_t0  # 注意log的文件夹不能取名为：tensorboard
        print('Train log dir' + train_log_dir)
        if os.path.exists(train_log_dir):
            for old_file in glob.glob(train_log_dir + '/*'):  # 重新训练时， 自动将该文件夹下的内容删除，避免图覆盖
                print 'Remove: ' + old_file
                os.remove(old_file)

        # val summary
        val_log_dir = PUNGAN_ROOT_PATH + '/WSD/tmp/tf.log/CAN/val-%s' % global_t0
        print('Val log dir' + val_log_dir)
        if os.path.exists(val_log_dir):
            for old_file in glob.glob(val_log_dir + '/*'):  # 重新训练时， 自动将该文件夹下的内容删除，避免图覆盖
                print 'Remove: ' + old_file
                os.remove(old_file)

        summary_train_writer = tf.summary.FileWriter(train_log_dir, graph=tf.get_default_graph())
        summary_val_writer = tf.summary.FileWriter(val_log_dir)
        '''
        print('Configuring model saver')
        # === Configuring model saver
        # 不用判断dir是否存在，saver会自建
        model_save_prefix = PUNGAN_ROOT_PATH + '/WSD/tmp/model/' + val_dataset + '/best'  # save model path 的前缀，因为后面还需要加入global step或者其他标识
        sota_model_save_prefix = PUNGAN_ROOT_PATH + '/WSD/tmp/model/SOTA/best'  # save model path 的前缀，因为后面还需要加入global step或者其他标识

        saver = tf.train.Saver()  # default: save all the global variables
        if config.warm_start:
            ckpt = tf.train.get_checkpoint_state(PUNGAN_ROOT_PATH + '/WSD/tmp/model/' + val_dataset + '/')
            if ckpt and ckpt.model_checkpoint_path:
                with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
                    saver.restore(session, ckpt.model_checkpoint_path)


        # store config (optional)
        #pickle.dump(config, open(PUNGAN_ROOT_PATH + '/WSD/tmp/model/conf.pkl', 'w'))
        return_reward, reward_detail, recall5 = test(model0, test_data, dict_data, word_to_id,session,target_id_to_word,target_sense_to_id, ids_pungan, senses_input)
        #train(model0, train_data,val_data,dict_data, word_to_id,session,target_id_to_word,target_sense_to_id,summary_train_writer,summary_val_writer,target_id_to_sense_id_to_sense,global_t0,back_off_result,saver,model_save_prefix)


    return return_reward, ids_pungan, reward_detail, recall5

def run_main_pretrain(wiki_unlabel, concatenate_file):
    global_t0 = int(time.time())
    Model = model.BiLSTM

    print('python train.py [x]')  # x可有可无，具体看下面的代码，没有的情况就进入debug参数组模式
    '''
    if len(sys.argv) > 1:
        run_type = int(sys.argv[1])
        if run_type == -1:
            print('Debug model!!')
            config.degug_model()   # 快速debug
            global_t0 = -1
        elif run_type == 0:
            print('Show Detail model')
            config.detailed_show()  # 存储模型+log
        elif 100 < run_type <= 500:
            print('Grid search model')
            config.get_grid_search_i(run_type)    # 网格搜索，细调参
        else:
            config.random_config()  # random search to find best parameters
    else:   # 直接
    '''
    config.degug_model()
    global_t0 = 0

    tag = '%s-%s' % (global_t0, config.name)
    print "time+model.name: " + tag
    print('Changed config')
    print config.changed_config
    print('All config')
    print vars(config)

    # Load data
    # ======================================================================================================================

    print('Loading all-words task data...')
    train_dataset = _path.ALL_WORDS_TRAIN_DATASET[0]  # ['semcor', 'semcor+omsti']
    print 'train_dataset: ' + train_dataset
    val_dataset = _path.ALL_WORDS_TEST_DATASET[
        0]  # ['ALL','senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015']
    config.has_pos = False  # all-words task has labeled with pos
    print 'val_dataset: ' + val_dataset
    test_dataset = _path.ALL_WORDS_TEST_DATASET[0]  # ALL
    print 'test_dataset: ' + test_dataset
    # print('Load lexical sample task data...')
    # dataset_i = 1 #int(sys.argv[1]) #    ## ['senseval2_LS', 'senseval3_LS']
    # train_dataset = _path.LS_DATASET[dataset_i]
    # val_dataset = _path.LS_DATASET[dataset_i]
    # config.has_pos = False  # lexical example task don't have pos
    # print 'val_dataset: '+val_dataset

    train_data_labeled = load_train_data(train_dataset)
    train_data_unlabeled = wiki_unlabel
    train_data_fake = concatenate_file
    train_data = train_data_labeled + train_data_unlabeled + train_data_fake
    val_data = load_val_data(val_dataset)
    test_data = load_test_data(test_dataset) # 需要修改
    print 'train_data[0]', train_data[0]
    print 'test_data[0]', test_data[0]
    print 'test_data len', len(test_data)
    print 'Dataset size (train/test): %d / %d' % (len(train_data), len(val_data))

    back_off_result = []
    if train_dataset in _path.ALL_WORDS_TRAIN_DATASET:
        val_data_lenth_pre = len(val_data)
        train_data, val_data, target_words, back_off_result = \
            data_postprocessing(train_dataset, val_dataset, train_data, val_data, config.back_off_type)
        test_data, test_data_index, _ = test_data_postprocessing(train_dataset, target_words, test_data)
        print 'Dataset size (train/test): %d / %d' % (len(train_data), len(val_data))
        print '***Using back-off instance: %d' % (len(back_off_result))
        missed = val_data_lenth_pre - (len(val_data) + len(back_off_result))
        print '***Missing instance(not in MFS/FS): %d/%d = %.3f' % (
            (missed, val_data_lenth_pre, float(missed) / val_data_lenth_pre))

    # ===build vocab utils
    word_to_id = build_vocab(train_data)
    with open(PUNGAN_ROOT_PATH + '/WSD/BiLSTM/word_to_id.pickle','wb') as f:
        pickle.dump(word_to_id, f)
    config.vocab_size = len(word_to_id)
    target_word_to_id, target_sense_to_id, n_words, n_senses_from_target_id = build_sense_ids(train_data)
    print('target_word_to_id.len',len(target_word_to_id))
    print('target_sense_to_id.len',len(target_sense_to_id))
    print 'Vocabulary size: %d' % len(word_to_id)
    print 'Target word: %d' % len(target_word_to_id)
    tot_n_senses = sum(n_senses_from_target_id.values())
    print 'Avg n senses per target word: ' + str(float(tot_n_senses) / len(n_senses_from_target_id))
    with open(PUNGAN_ROOT_PATH + '/WSD/tmp/target_word.txt', 'w') as f:
        for word, id in target_word_to_id.items():
            f.write('{}\t{}\n'.format(word, id))

    # ===make numericR
    train_data_labeled = convert_to_numeric(train_data_labeled, word_to_id, target_word_to_id,
                                    target_sense_to_id, n_senses_from_target_id, data_type="labeled")
    train_data_unlabeled = convert_to_numeric(train_data_unlabeled, word_to_id, target_word_to_id,
                                    target_sense_to_id, n_senses_from_target_id, data_type="unlabeled")
    train_data_fake = convert_to_numeric(train_data_fake, word_to_id, target_word_to_id,
                                        target_sense_to_id, n_senses_from_target_id, data_type="fake")
    train_data = train_data_labeled
    #train_data = train_data_labeled + train_data_unlabeled + train_data_fake # 1:1:1 for RL
    val_data = convert_to_numeric(val_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id, ignore_sense_not_in_train=True)

    print('test_data.shape',len(test_data))
    # test_data = convert_to_numeric(test_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id, ignore_sense_not_in_train=False)
    # print('test_data.shape',len(test_data))
    target_id_to_word = {id: word for (word, id) in target_word_to_id.iteritems()}
    target_id_to_sense_id_to_sense = [{sense_id: sense for (sense, sense_id) in sense_to_id.iteritems()} for
                                      (target_id, sense_to_id) in enumerate(target_sense_to_id)]

    # get dic and make numeric
    # train_target_words = target_word_to_id.keys()  # 两者都可以 都是表示train和test重合部分的词
    gloss_dict, synset_dict = load_dictionary(train_dataset, target_word_to_id, config.expand_type)
    mask_size = 2 * config.hidden_size
    dict_data, config.max_n_sense = bulid_dictionary_id(gloss_dict, synset_dict, target_sense_to_id, word_to_id,
                                                        word_to_id['<pad>'], mask_size, config.g_len,
                                                        is_empty=config.gloss_is_empty)
    if config.use_pre_trained_emb:
        print('Load pre-trained word embedding...')
        init_emb = fill_with_gloves(word_to_id, config.embedding_size, config.vocab_name)
    else:
        init_emb = None

    # Training
    # ======================================================================================================================

    # === Initial model
    model0 = Model(config, n_senses_from_target_id, init_emb)

    tf.constant([1, 2], name='constant-test')

    # === Create session
    # session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as session:  # limit gpu memory; don't pre-allocate memory; allocate as-needed
        session.run(tf.global_variables_initializer())

        # === Start training

        print("Configuring TensorBoard and Saver")
        '''
        # ===Configuring TensorBoard(Summary)
        # train summary
        train_log_dir = PUNGAN_ROOT_PATH + '/WSD/tmp/tf.log/CAN/train-%s' % global_t0  # 注意log的文件夹不能取名为：tensorboard
        print('Train log dir' + train_log_dir)
        if os.path.exists(train_log_dir):
            for old_file in glob.glob(train_log_dir + '/*'):  # 重新训练时， 自动将该文件夹下的内容删除，避免图覆盖
                print 'Remove: ' + old_file
                os.remove(old_file)

        # val summary
        val_log_dir = PUNGAN_ROOT_PATH + '/WSD/tmp/tf.log/CAN/val-%s' % global_t0
        print('Val log dir' + val_log_dir)
        if os.path.exists(val_log_dir):
            for old_file in glob.glob(val_log_dir + '/*'):  # 重新训练时， 自动将该文件夹下的内容删除，避免图覆盖
                print 'Remove: ' + old_file
                os.remove(old_file)

        summary_train_writer = tf.summary.FileWriter(train_log_dir, graph=tf.get_default_graph())
        summary_val_writer = tf.summary.FileWriter(val_log_dir)
        '''
        print('Configuring model saver')
        # === Configuring model saver
        # 不用判断dir是否存在，saver会自建
        model_save_prefix = PUNGAN_ROOT_PATH + '/WSD/tmp/model/' + val_dataset + '/best'  # save model path 的前缀，因为后面还需要加入global step或者其他标识
        sota_model_save_prefix = PUNGAN_ROOT_PATH + '/WSD/tmp/model/SOTA/best'  # save model path 的前缀，因为后面还需要加入global step或者其他标识

        saver = tf.train.Saver()  # default: save all the global variables
        # if config.warm_start:
        #     ckpt = tf.train.get_checkpoint_state(PUNGAN_ROOT_PATH + '/WSD/tmp/model/' + val_dataset + '/')
        #     if ckpt and ckpt.model_checkpoint_path:
        #         with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        #             saver.restore(session, ckpt.model_checkpoint_path)


        # store config (optional)
        #pickle.dump(config, open(PUNGAN_ROOT_PATH + '/WSD/tmp/model/conf.pkl', 'w'))
        #return_reward = test(model0, test_data, dict_data, word_to_id, session, target_id_to_word,target_sense_to_id, test_data_index, senses_input)
        train(model0, train_data, val_data, dict_data, word_to_id, session, target_id_to_word,
              target_sense_to_id, target_id_to_sense_id_to_sense,
              global_t0, back_off_result, saver, model_save_prefix, tag, train_dataset, val_dataset)

def run_main_rltrain(wiki_unlabel, concatenate_file):
    global_t0 = int(time.time())
    Model = model.BiLSTM

    print('python train.py [x]')  # x可有可无，具体看下面的代码，没有的情况就进入debug参数组模式
    '''
    if len(sys.argv) > 1:
        run_type = int(sys.argv[1])
        if run_type == -1:
            print('Debug model!!')
            config.degug_model()   # 快速debug
            global_t0 = -1
        elif run_type == 0:
            print('Show Detail model')
            config.detailed_show()  # 存储模型+log
        elif 100 < run_type <= 500:
            print('Grid search model')
            config.get_grid_search_i(run_type)    # 网格搜索，细调参
        else:
            config.random_config()  # random search to find best parameters
    else:   # 直接
    '''
    config.degug_model()
    global_t0 = 0

    tag = '%s-%s' % (global_t0, config.name)
    print "time+model.name: " + tag
    print('Changed config')
    print config.changed_config
    print('All config')
    print vars(config)

    # Load data
    # ======================================================================================================================

    print('Loading all-words task data...')
    train_dataset = _path.ALL_WORDS_TRAIN_DATASET[0]  # ['semcor', 'semcor+omsti']
    print 'train_dataset: ' + train_dataset
    val_dataset = _path.ALL_WORDS_TEST_DATASET[
        0]  # ['ALL','senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015']
    config.has_pos = False  # all-words task has labeled with pos
    print 'val_dataset: ' + val_dataset
    test_dataset = _path.ALL_WORDS_TEST_DATASET[0]  # ALL
    print 'test_dataset: ' + test_dataset
    # print('Load lexical sample task data...')
    # dataset_i = 1 #int(sys.argv[1]) #    ## ['senseval2_LS', 'senseval3_LS']
    # train_dataset = _path.LS_DATASET[dataset_i]
    # val_dataset = _path.LS_DATASET[dataset_i]
    # config.has_pos = False  # lexical example task don't have pos
    # print 'val_dataset: '+val_dataset

    train_data_labeled = load_train_data(train_dataset)
    train_data_unlabeled = wiki_unlabel
    train_data_fake = concatenate_file
    train_data = train_data_labeled + train_data_unlabeled + train_data_fake
    val_data = load_val_data(val_dataset)
    test_data = load_test_data(test_dataset) # 需要修改
    print 'train_data[0]', train_data[0]
    print 'test_data[0]', test_data[0]
    print 'test_data len', len(test_data)
    print 'Dataset size (train/test): %d / %d' % (len(train_data), len(val_data))

    back_off_result = []
    if train_dataset in _path.ALL_WORDS_TRAIN_DATASET:
        val_data_lenth_pre = len(val_data)
        train_data, val_data, target_words, back_off_result = \
            data_postprocessing(train_dataset, val_dataset, train_data, val_data, config.back_off_type)
        test_data, test_data_index, _ = test_data_postprocessing(train_dataset, target_words, test_data)
        print 'Dataset size (train/test): %d / %d' % (len(train_data), len(val_data))
        print '***Using back-off instance: %d' % (len(back_off_result))
        missed = val_data_lenth_pre - (len(val_data) + len(back_off_result))
        print '***Missing instance(not in MFS/FS): %d/%d = %.3f' % (
            (missed, val_data_lenth_pre, float(missed) / val_data_lenth_pre))

    # ===build vocab utils
    # word_to_id = build_vocab(train_data)
    with open(PUNGAN_ROOT_PATH + '/WSD/BiLSTM/word_to_id.pickle','rb') as f:
        word_to_id = pickle.load(f)
    config.vocab_size = len(word_to_id)
    target_word_to_id, target_sense_to_id, n_words, n_senses_from_target_id = build_sense_ids(train_data)
    print('target_word_to_id.len',len(target_word_to_id))
    print('target_sense_to_id.len',len(target_sense_to_id))
    print 'Vocabulary size: %d' % len(word_to_id)
    print 'Target word: %d' % len(target_word_to_id)
    tot_n_senses = sum(n_senses_from_target_id.values())
    print 'Avg n senses per target word: ' + str(float(tot_n_senses) / len(n_senses_from_target_id))
    with open(PUNGAN_ROOT_PATH + '/WSD/tmp/target_word.txt', 'w') as f:
        for word, id in target_word_to_id.items():
            f.write('{}\t{}\n'.format(word, id))

    # ===make numericR
    train_data_labeled = convert_to_numeric(train_data_labeled, word_to_id, target_word_to_id,
                                    target_sense_to_id, n_senses_from_target_id, data_type="labeled")
    train_data_unlabeled = convert_to_numeric(train_data_unlabeled, word_to_id, target_word_to_id,
                                    target_sense_to_id, n_senses_from_target_id, data_type="unlabeled")
    train_data_fake = convert_to_numeric(train_data_fake, word_to_id, target_word_to_id,
                                        target_sense_to_id, n_senses_from_target_id, data_type="fake")
    #train_data = train_data_labeled # for pretrain
    random.shuffle(train_data_labeled)
    random.shuffle(train_data_unlabeled)
    train_data = train_data_labeled[:500] + train_data_unlabeled[:500] + train_data_fake # 1:1:1 for RL
    val_data = convert_to_numeric(val_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id, ignore_sense_not_in_train=True)

    print('test_data.shape',len(test_data))
    # test_data = convert_to_numeric(test_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id, ignore_sense_not_in_train=False)
    # print('test_data.shape',len(test_data))
    target_id_to_word = {id: word for (word, id) in target_word_to_id.iteritems()}
    target_id_to_sense_id_to_sense = [{sense_id: sense for (sense, sense_id) in sense_to_id.iteritems()} for
                                      (target_id, sense_to_id) in enumerate(target_sense_to_id)]

    # get dic and make numeric
    # train_target_words = target_word_to_id.keys()  # 两者都可以 都是表示train和test重合部分的词
    gloss_dict, synset_dict = load_dictionary(train_dataset, target_word_to_id, config.expand_type)
    mask_size = 2 * config.hidden_size
    dict_data, config.max_n_sense = bulid_dictionary_id(gloss_dict, synset_dict, target_sense_to_id, word_to_id,
                                                        word_to_id['<pad>'], mask_size, config.g_len,
                                                        is_empty=config.gloss_is_empty)
    if config.use_pre_trained_emb:
        print('Load pre-trained word embedding...')
        init_emb = fill_with_gloves(word_to_id, config.embedding_size, config.vocab_name)
    else:
        init_emb = None

    # Training
    # ======================================================================================================================

    # === Initial model
    model0 = Model(config, n_senses_from_target_id, init_emb)

    tf.constant([1, 2], name='constant-test')

    # === Create session
    # session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as session:  # limit gpu memory; don't pre-allocate memory; allocate as-needed
        session.run(tf.global_variables_initializer())

        # === Start training

        print("Configuring TensorBoard and Saver")
        '''
        # ===Configuring TensorBoard(Summary)
        # train summary
        train_log_dir = PUNGAN_ROOT_PATH + '/WSD/tmp/tf.log/CAN/train-%s' % global_t0  # 注意log的文件夹不能取名为：tensorboard
        print('Train log dir' + train_log_dir)
        if os.path.exists(train_log_dir):
            for old_file in glob.glob(train_log_dir + '/*'):  # 重新训练时， 自动将该文件夹下的内容删除，避免图覆盖
                print 'Remove: ' + old_file
                os.remove(old_file)

        # val summary
        val_log_dir = PUNGAN_ROOT_PATH + '/WSD/tmp/tf.log/CAN/val-%s' % global_t0
        print('Val log dir' + val_log_dir)
        if os.path.exists(val_log_dir):
            for old_file in glob.glob(val_log_dir + '/*'):  # 重新训练时， 自动将该文件夹下的内容删除，避免图覆盖
                print 'Remove: ' + old_file
                os.remove(old_file)

        summary_train_writer = tf.summary.FileWriter(train_log_dir, graph=tf.get_default_graph())
        summary_val_writer = tf.summary.FileWriter(val_log_dir)
        '''
        print('Configuring model saver')
        # === Configuring model saver
        # 不用判断dir是否存在，saver会自建
        model_save_prefix = PUNGAN_ROOT_PATH + '/WSD/tmp/model/' + val_dataset + '/best'  # save model path 的前缀，因为后面还需要加入global step或者其他标识
        sota_model_save_prefix = PUNGAN_ROOT_PATH + '/WSD/tmp/model/SOTA/best'  # save model path 的前缀，因为后面还需要加入global step或者其他标识

        saver = tf.train.Saver()  # default: save all the global variables
        if config.warm_start:
            ckpt = tf.train.get_checkpoint_state(PUNGAN_ROOT_PATH + '/WSD/tmp/model/' + val_dataset + '/')
            if ckpt and ckpt.model_checkpoint_path:
                with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
                    saver.restore(session, ckpt.model_checkpoint_path)


        # store config (optional)
        #pickle.dump(config, open(PUNGAN_ROOT_PATH + '/WSD/tmp/model/conf.pkl', 'w'))
        #return_reward = test(model0, test_data, dict_data, word_to_id, session, target_id_to_word,target_sense_to_id, test_data_index, senses_input)
        train_rl(model0, train_data, val_data, dict_data, word_to_id, session, target_id_to_word,
              target_sense_to_id, target_id_to_sense_id_to_sense,
              global_t0, back_off_result, saver, model_save_prefix, tag, train_dataset, val_dataset)
def wordnet_process(filename):
        '''
        {'target_word': u'picture#n', 'target_sense': None, 'id': None, 'context': [u'<unk>', u'is', u'the', '<target>', u'of', u'the', u'<unk>', u'and', u'the', u'<unk>'], 'poss': ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.']}
        '''
        dict_list = []
        with open(filename) as f1:
            for i, line in enumerate(f1):
                wordlist = line.strip().split()
                dict = {}
                target_id = -1
                target = ''
                for id, word in enumerate(wordlist):
                    if '%' in word:
                        name = wn.lemma_from_key(word).synset().name().encode('utf-8').split('.')
                        target_id = id
                        # target = name[0]+'#'+name[1]
                        target = word[:word.index('%')]+'#'+name[1]
                        break
                dict['target_word'] = target
                dict['target_sense'] = None
                dict['id'] = None
                dict['context'] = []
                for id, word in enumerate(wordlist):
                    if id == target_id:
                        dict['context'].append('<target>')
                    else:
                        dict['context'].append(word)
                dict['poss'] = ['.' for i in range(0, len(wordlist))]
                dict_list.append(dict)
        with open(filename+'.pickle', 'wb') as fp:
            pickle.dump(dict_list, fp)
if __name__ == "__main__":
    if sys.argv[1] == 'pretrain':
        # 2. PRE-TRAIN DISCRIMINATOR
        # wordnet_process() # sentences generated by Generator
        wordnet_process(PUNGAN_ROOT_PATH + '/Pun_Generation/data/sample_2548')
        with open(PUNGAN_ROOT_PATH + '/WSD/data/All_Words_WSD/Training_Corpora/wiki_unlabel.pickle', 'rb') as f:
            wiki_unlabel = pickle.load(f)
        with open(PUNGAN_ROOT_PATH + '/Pun_Generation/data/sample_2548.pickle', 'rb') as f:
            concatenate_vocab = pickle.load(f)
        run_main_pretrain(wiki_unlabel, concatenate_vocab)
    elif sys.argv[1] == 'rltrain':
        # 3. RL TRAIN DISCRIMINATOR
        wordnet_process(PUNGAN_ROOT_PATH + '/Pun_Generation/code/backward_model_path/concatenate_file') # sentences generated by Generator
        with open(PUNGAN_ROOT_PATH + '/WSD/data/All_Words_WSD/Training_Corpora/wiki_unlabel.pickle', 'rb') as f:
            wiki_unlabel = pickle.load(f)
        with open(PUNGAN_ROOT_PATH + '/WSD/data/All_Words_WSD/Training_Corpora/concatenate_file.pickle', 'rb') as f:
            concatenate_file = pickle.load(f)
        run_main_rltrain(wiki_unlabel, concatenate_file)
    else:
        # 1. TEST IN RL
        with open(sys.argv[1], 'rb') as f:
            dict = pickle.load(f)
        wsd_input = dict['wsd_input']
        senses_input = dict['senses_input']
        with open(PUNGAN_ROOT_PATH + '/WSD/data/All_Words_WSD/Training_Corpora/wiki_unlabel.pickle', 'rb') as f:
            wiki_unlabel = pickle.load(f)
        wordnet_process(PUNGAN_ROOT_PATH + '/Pun_Generation/data/sample_2548')
        with open(PUNGAN_ROOT_PATH + '/Pun_Generation/data/sample_2548.pickle', 'rb') as f:
            concatenate_file = pickle.load(f)
        return_reward, test_data_index, reward_detail, recall5 = run_main(wsd_input, senses_input, wiki_unlabel, concatenate_file)
        out_dict = {'return_reward': return_reward, 'test_data_index': test_data_index, 'reward_detail': reward_detail, 'recall5': recall5}
        with open(sys.argv[1].replace("input","output"), 'wb') as fw:
            pickle.dump(out_dict, fw)
