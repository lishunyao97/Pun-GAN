# -*- coding: utf-8 -*-  
"""  
 @version: python2.7 
 @author: luofuli 
 @time: 2018/4/1 19:56
"""

import os
import sys
# print sys.path  # 一般是从该文件下的所在的目录中查询，所以找不到MemNN和utils、postprocessing这些目录
# sys.path.insert(0, "..")  # 保证下面的import 其他文件夹的类成功

import path
import score
#from postprocessing import ensemble

_path = path.WSD_path()


# 保存结果文件以及得分文件
def save_result_and_score(tag, train_dataset, val_dataset, back_off_result, best_result, prob_best_result,
                          best, config=None, score_path='../tmp/params_score.txt', print_logs=True):
    # 用best_i而不是最近的存储的i
    if train_dataset in _path.LS_DATASET:
        gold_key_path = _path.LS_TEST_KEY_PATH.format(val_dataset)
    else:
        gold_key_path = _path.ALL_WORDS_TEST_KEY_PATH.format(val_dataset)

    if not os.path.exists('../tmp/results'):
        os.makedirs('../tmp/results')

    best_result_path = '../tmp/results/{0}-best-{1}-{2}'.format(tag, str(best['i']), val_dataset)
    print('best_result_path: ' + best_result_path)
    score.write_result(best_result, back_off_result, best_result_path)
    f1s = None
    if val_dataset == _path.ALL_WORDS_TEST_DATASET[0]:
        f1s, pos_ps = score.score_all(best_result_path, gold_key_path, print_logs=print_logs, logs_level=1)
    else:
        _, _, f1 = score.score_one(best_result_path, gold_key_path)

    prob_best_path = '../tmp/results/{0}-prob-{1}'.format(tag, val_dataset)
#    ensemble_result_path = '../tmp/results/{0}-ensemble-{1}'.format(tag, val_dataset)
#    f1b, b_i = ensemble.score_for_prob(prob_best_result, back_off_result,
#                                       gold_key_path, print_logs=False)  # luofuli add @2018/2/1
#    print('Best prob F1:%s\t(step: %d)' % (f1b, best['i'] + b_i + 1))
#    ensemble_result = ensemble.vote(prob_best_result, back_off_result, prob_best_path, ensemble_result_path)
#    f1s_ = None
#    if val_dataset == _path.ALL_WORDS_TEST_DATASET[0]:
#        f1s_, pos_ps_ = score.score_all(ensemble_result_path, gold_key_path, print_logs=print_logs, logs_level=1)
#    else:
#        _, _, f1_ = score.score_one(ensemble_result_path, gold_key_path)
#
#    print('Writing param score in path:%s, tag: %s' % (score_path, tag))
#    try:
#        old = open(score_path).read()
#    except Exception:
#        old = ''
#
#    if f1s and f1s_:
#        with open(score_path, 'w') as f:
#            f.write(
#                '%s%s\tbest\t%.1f\t%.1f' % (old, tag, best['acc_train'] * 100, best['acc_val'] * 100))
#            for f1 in f1s:
#                f.write('\t%.1f' % (f1 * 100))
#            for p in pos_ps:
#                f.write('\t%.1f' % (p * 100))
#        score.store_params(score_path, val_dataset, config)
#        old = open(score_path).read()
#        with open(score_path, 'w') as f:
#            f.write('%s%s\tensemble\t%.1f\t%.1f' % (
#                old, tag, best['acc_train'] * 100, best['acc_val'] * 100))
#            for f1_ in f1s_:
#                f.write('\t%.1f' % (f1_ * 100))
#            for p_ in pos_ps_:
#                f.write('\t%.1f' % (p_ * 100))
#        score.store_params(score_path, val_dataset, config)
#    else:
#        with open(score_path, 'w') as f:
#            f.write('%s%s\t%.1f\t%.1f' % (old, tag, best['acc_train'] * 100, best['acc_val'] * 100))
#            f.write('\t%.1f\t%.1f' % (f1 * 100, f1_ * 100))
#        score.store_params(score_path, val_dataset, config)


def check_correct(correct_path, result_path, gold_key_path, id_index=0, unsocre_dataset='semeval2007'):
    id_to_key = {}
    for line in open(gold_key_path):
        line = line.split()  # defaut \s=[ \f\n\r\t\v]
        id = line[id_index]
        key = line[id_index + 1]
        id_to_key[id] = key

    true_ = {}
    with open(result_path) as f2:
        for line in f2.readlines():
            id = line.split()[id_index]
            sense = line.split()[id_index + 1]
            if sense == id_to_key.get(id):
                tag = 1
            else:
                tag = 0
            true_[id] = tag

    import numpy as np

    with open(correct_path) as f1:
        correct = []
        correct_has_se7 = []
        error = []
        error_id = []
        lines1 = f1.readlines()
        for i, line1 in enumerate(lines1):
            id = line1.split()[id_index]
            tag = int(line1.split()[id_index+1])
            if tag != true_[id]:
                error.append(tag)
                error_id.append(id)
                # print ('error line: %s' % line1)
            else:
                correct_has_se7.append(tag)
                if unsocre_dataset not in id:
                    correct.append(tag)
    print('Using back-off: %s' % (len(id_to_key) - len(lines1)))
    print('Total score: %s' % (np.mean(correct)))
    print('Total score(has se7): %s' % (np.mean(correct_has_se7)))
    print('Total score(has se7 + error): %s' % np.mean(error + correct_has_se7))
    print('error number: %s' % len(error))

    return error_id


def write_result(results, back_off_result, path, print_logs=True):
    if print_logs:
        print('Writing to file:%s' % path)
    new_results = results + back_off_result
    new_results = sorted(new_results, key=lambda a: a[0])
    with open(path, 'w') as file:
        for instance_id, predicted_sense in new_results:
            file.write('%s %s\n' % (instance_id, predicted_sense))


if __name__ == "__main__":
    e1 = check_correct(correct_path='../tmp/results/correct-1.txt',
                  result_path='../tmp/results/0CAN-best-1-ALL-1',
                  gold_key_path=_path.ALL_WORDS_TEST_KEY_PATH.format('ALL'))

    e2 = check_correct(correct_path='../tmp/results/correct-2.txt',
                  result_path='../tmp/results/0CAN-best-1-ALL-2',
                  gold_key_path=_path.ALL_WORDS_TEST_KEY_PATH.format('ALL'))

    e3 = check_correct(correct_path='../tmp/results/correct.txt',
                       result_path='../tmp/results/0CAN-best-1-ALL',
                       gold_key_path=_path.ALL_WORDS_TEST_KEY_PATH.format('ALL'))

    print((set(e1) & set(e2)))
    print((set(e1) & set(e3)))
    print((set(e2) & set(e3)))

    print(u'最后找出来原因了，为什么模型run的val_acc比真实测出来的值低，原因主要是：'
          u' MFS 的结果没有包含在其中，FS: 2081/2370 = 87.8%的正确率，后面在train.py里增加了一个函数'
          u'此外，出来的值有部分是测试集的答案不在训练集中的，也就是test_instance_sensekey_not_in_train，总共有312个，占比4.3%。'
          u'这部分在测试的时候被统一归为了0类，所以导致部分预测到0类，实际上是假性提高了模型的正确率（这部分与多歧义互相+-吧）'
          u'最后的结论就是，代码没有问题，哪儿都没有问题，唯一需要考虑的就是，要不要预测sense不在训练集这种情况？因为不预测肯定会提高F1值'
          u'For 原本正确率上升和召回率不变，所以F1上升'
          )

