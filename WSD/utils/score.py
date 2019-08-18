# -*- coding: utf-8 -*-

import os
import pickle
import re
import sys

import lxml.etree as et

from utils.path import *

_path = WSD_path()


def print_int(ds):
    s = ''
    for d in ds:
        if d < 1:
            s += '\t%.1f' % (d * 100)
        else:
            s += '\t%.1f' % (d)
    return s


def get_AW_pos_map(data_path):
    context = et.iterparse(data_path, tag='instance')
    id_pos_map = {}
    for event, inst in context:
        id = inst.get('id')
        if _path.ALL_WORDS_VAL_DATASET not in id:  # semeval2007 is validation datax
            pos = inst.get('pos')
            id_pos_map[id] = pos
    return id_pos_map


def get_LS_pos_map(data_path):
    i = 0


def split_gold_key_by_pos(gold_key_path, id_pos_map, id_index):
    poss = set(id_pos_map.values())
    for pos in poss:
        data = []
        for line in open(gold_key_path):
            ans_id = line.split()[id_index]
            if _path.ALL_WORDS_VAL_DATASET not in ans_id:  # semeval2007 is validation datax
                p = id_pos_map[ans_id]
                if p == pos:
                    data.append(line)
        open(gold_key_path + '.' + pos, 'w').write(''.join(data))


def score_for_pos(dataset, answer_path, key_path, id_index=0, print_logs=False, multi_sense=True):
    """
    分析不同postag的结果
    """
    if dataset in _path.LS_DATASET:
        id_pos_map = get_LS_pos_map(_path.LS_VAL_PATH.format(dataset))
    else:
        id_pos_map = get_AW_pos_map(_path.ALL_WORDS_TEST_PATH.format(dataset))

    split_gold_key_by_pos(key_path, id_pos_map, id_index)

    pos_group = {}
    for i, line in enumerate(open(answer_path)):
        ans_id = line.split()[id_index]
        if _path.ALL_WORDS_VAL_DATASET not in ans_id:  # semeval2007 is validation datax
            pos = id_pos_map[ans_id]
            if pos not in pos_group:
                pos_group[pos] = [line]
            else:
                pos_group[pos].append(line)
    ps = []
    poss = []
    for pos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
        if pos in pos_group.keys():
            lines = pos_group[pos]
            ans_path = answer_path + '.' + pos
            open(ans_path, 'w').write(''.join(lines))
            p, r, f1 = score_one(ans_path, key_path + '.' + pos, print_logs=False)
            os.remove(ans_path)
            ps.append(f1)
            poss.append(pos)
        else:
            ps.append(0)
            poss.append(pos)
    p, r, f1 = score_one(answer_path, key_path, print_logs=False, unsocre_dataset=_path.ALL_WORDS_VAL_DATASET,
                         multi_sense=multi_sense)
    poss.append('ALL')
    ps.append(f1)

    if print_logs:
        print('POS_Score\t' + '\t'.join(poss))
        ims = [71.9, 56.6, 75.9, 84.7, 70.1]
        emnlp17 = [71.5, 57.5, 75.0, 83.8, 69.9]
        print('IMS+emb=' + print_int(ims))
        print('EMNLP17=' + print_int(emnlp17))
        if p == r:
            print('MemNN+-=' + print_int(ps))
        else:
            print('MemNN+-=' + print_int(ps)),
            print('(P:%.1f  R:%.1f)' % (p*100, r*100))

    return ps


def get_pos_distribution(path):
    print path
    id_pos_map = get_AW_pos_map(path)
    m = ['NOUN', 'VERB', 'ADJ', 'ADV']
    count = [0, 0, 0, 0]
    for id, pos in id_pos_map.iteritems():
        count[m.index(pos)] += 1
    print(count)


def score_all(answer_path, gold_key_path, id_index=None, print_logs=True, logs_level=2, multi_sense=True):
    """
    id index refers to the column number of instance id
    id_index==0: (id, key) of each line
    id_index==1: (word, id, key) of each line
    """
    # this method is not always right
    if id_index is None:
        line = open(answer_path).readline()
        if len(line.split()) >= 3:
            id_index = 1
            if print_logs and logs_level > 1:
                print('Warning: Detect file format: (word, id, key). '),
                print('You can specifies format at the end of cmd 0: (id, key)  1:(word, id, key) ')
        else:
            id_index = 0
            if print_logs and logs_level > 1:
                print('Warning: File format should be: (id, key). '),
                print('You can specifies format at the end of cmd 0: (id, key)  1:(word, id, key) ')

    print('Score for all dataset')
    if not multi_sense:
        print(u'!!强制匹配第一个sense')

    all_id_to_key = {}
    for line in open(gold_key_path):
        line = line.split()  # defaut \s=[ \f\n\r\t\v]
        id0 = line[id_index]
        dataset_i = _path.ALL_WORDS_TEST_DATASET.index(id0.split('.')[0])
        id = '.'.join(id0.split('.')[1:])
        # print id
        # exit(-1)
        key = line[id_index + 1:]
        if dataset_i not in all_id_to_key.keys():
            all_id_to_key[dataset_i] = {id: key}
        else:
            all_id_to_key[dataset_i][id] = key

    ps = []
    rs = []
    f1s = []
    for dataset_i, id_to_key in sorted(all_id_to_key.items()):  # ensure the order is senseval2,3,7,13,15
        dataset = _path.ALL_WORDS_TEST_DATASET[dataset_i]
        if print_logs and logs_level > 1:
            print 'Score:%s' % dataset
        ok = 0
        notok = 0
        tried = 0
        ans_ids = []
        for i, line in enumerate(open(answer_path)):
            line = line.split()
            ds = line[id_index].split('.')[0]
            if dataset != ds:
                continue
            ans_id = '.'.join(line[id_index].split('.')[1:])
            if multi_sense:  # support multi-sense score
                ans_key = line[id_index + 1:]
                if ans_id in id_to_key:
                    if ans_id in ans_ids:
                        if print_logs and logs_level > 1:
                            print('%s: Unable to score answer for "%s" (line %s): Already scored.' % (
                                dataset, ans_id, i))

                    # Handling multiple answers in system
                    else:
                        tried += 1
                        ans_ids.append(ans_id)
                        n_key_ok = len(set(ans_key) & set(id_to_key[ans_id]))
                        n_key_notok = len(ans_key) - n_key_ok
                        ok += float(n_key_ok) / len(ans_key)  # if system is multi-key, mean it
                        notok += float(n_key_notok) / len(ans_key)
                else:
                    if print_logs and logs_level > 1:
                        print('%s: Unable to score answer for "%s" (line %s): Excluded.' % (dataset, ans_id, i))
            else:
                ans_key = line[id_index + 1]
                if ans_key == id_to_key.get(ans_id)[0]:
                    ok += 1.0
                else:
                    notok += 1.0
        total = len(id_to_key.keys())
        p = ok / (ok + notok) if ok + notok else 0
        r = ok / total
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0
        ps.append(p)
        rs.append(r)
        f1s.append(f1)

    ims = [72.2, 70.4, 62.6, 65.9, 71.5]
    emnlp17 = [72.0, 69.1, 64.8, 66.9, 71.5]
    if print_logs:
        print('Dataset-\tSE2\tSE3\tSE7\tSE13\tSE15')
        print('IMS+emb=' + print_int(ims))
        print('EMNLP17=' + print_int(emnlp17))
        if ps != rs:
            print('MemNN-P=' + print_int(ps))
            print('MemNN-R=' + print_int(rs))
            print('MemNN-F1=' + print_int(f1s) + ' # P!=R!=F!')
        else:
            print('MemNN+-=' + print_int(f1s) + ' # P=R=F1')

    pos_ps = score_for_pos('ALL', answer_path, gold_key_path, id_index, print_logs, multi_sense)

    return f1s, pos_ps


def score_one(answer_path, gold_key_path, id_index=None, store_path=None, print_logs=False, unsocre_dataset='YY',
              logs_level=2, multi_sense=True):
    """
    id index refers to the column number of instance id
    id_index==0: (id, key) of each line
    id_index==1: (word, id, key) of each line
    """
    # this method is not always right
    if print_logs and unsocre_dataset != 'YY':
        print '!!!unsocre_dataset: ', unsocre_dataset

    if id_index is None:
        line = open(answer_path).readlines()[0]
        if len(line.split()) >= 3:
            id_index = 1
            if print_logs and logs_level > 1:
                print('Warning: Detect file format: (word, id, key). '),
                print('You can specifies format at the end of cmd 0: (id, key)  1:(word, id, key) ')
        else:
            id_index = 0
            if print_logs and logs_level > 1:
                print('Warning: File format should be: (id, key). '),
                print('You can specifies format at the end of cmd 0: (id, key)  1:(word, id, key) ')

    id_to_key = {}
    for line in open(gold_key_path):
        line = line.split()  # defaut \s=[ \f\n\r\t\v]
        id = line[id_index]
        if unsocre_dataset not in id:
            key = line[id_index + 1:]
            id_to_key[id] = key

    ok = 0
    notok = 0
    tried = 0
    ans_ids = []
    for i, line in enumerate(open(answer_path)):
        line = line.split()
        ans_id = line[id_index]
        if unsocre_dataset in ans_id:  # 去除掉一个dataset
            continue

        ans_key = line[id_index + 1:]
        if multi_sense:
            if ans_id in id_to_key:
                if ans_id in ans_ids:
                    if print_logs and logs_level > 1:
                        print('Unable to score answer for "%s" (line %s): Already scored.' % (ans_id, i))

                # Handling multiple answers in system
                else:
                    tried += 1
                    ans_ids.append(ans_id)
                    n_key_ok = len(list(set(ans_key) & set(id_to_key[ans_id])))
                    n_key_notok = len(ans_key) - n_key_ok
                    ok += float(n_key_ok) / len(ans_key)  # if system is multi-key, mean it
                    notok += float(n_key_notok) / len(ans_key)
            else:
                if print_logs and logs_level > 1:
                    print('Unable to score answer for "%s" (line %s): Excluded.' % (ans_id, i))
        else:
            ans_key = line[id_index + 1]
            if ans_key == id_to_key.get(ans_id)[0]:
                ok += 1.0
            else:
                notok += 1.0

    total = len(id_to_key.keys())
    p = ok / (ok + notok)
    r = ok / total
    if p + r > 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0

    if store_path:
        dataset = re.findall('-(.*?)$', answer_path)[-1]
        save_prefix = dataset + '/'
        print('Store score at %s' % (store_path))
        store_result(f1, store_path, save_prefix)

    if print_logs:
        print('P=\t%.1f\t(%.2f correct of %.2f attempt)' % (p * 100, ok, ok + notok))
        print('R=\t%.1f\t(%.2f correct of %.2f in total)' % (r * 100, ok, total))
        print('F1=\t%.1f\t(2*P*R/(P+R))' % (f1 * 100))

    return p, r, f1


def store_result(f1_score, dst_path, save_prefix):
    try:
        conf_path = '../tmp/model/' + save_prefix + 'conf.pkl'
        conf = pickle.load(open(conf_path))
        if not conf:
            conf = {}
            print '\nWARNING: no conf.pkl file found at %s!\n' % conf_path
    except Exception:
        conf = {}
        print '\nWARNING: no conf.pkl file found at %s!\n' % conf_path

    if os.path.exists(dst_path):
        old_result = open(dst_path).read()
    else:
        old_result = ''
    with open(dst_path, 'w') as f:
        f.write(old_result)
        f.write('%.1f' % (f1_score * 100))
        for key, value in conf.items():
            f.write('\t{}:{}'.format(key, value))
        f.write('\n')


def store_params(dst_path, dataset='', params=None):
    if params is None:
        save_prefix = dataset + '/'
        try:
            conf_path = '../tmp/model/' + save_prefix + 'conf.pkl'
            conf = pickle.load(open(conf_path))
            if not conf:
                conf = {}
                print '\nWARNING: no conf.pkl file found at %s!\n' % conf_path
        except Exception:
            conf = {}
            print '\nWARNING: no conf.pkl file found at %s!\n' % conf_path
    else:
        if type(params) is dict:
            conf = params
        else:
            conf = vars(params)

    if os.path.exists(dst_path):
        old_result = open(dst_path).read()
    else:
        old_result = ''
    with open(dst_path, 'w') as f:
        f.write(old_result)
        for key, value in conf.items():
            f.write('\t{}:{}'.format(key, value))
        f.write('\n')


def write_result(results, back_off_result, path, print_logs=True):
    if print_logs:
        print('Writing to file:%s' % path)
    new_results = results + back_off_result
    new_results = sorted(new_results, key=lambda a: a[0])
    with open(path, 'w') as file:
        for instance_id, predicted_sense in new_results:
            file.write('%s %s\n' % (instance_id, predicted_sense))


def score_for_back_off_result(back_off_type, print_logs=False):
    back_off_result_path = _path.BACK_OFF_RESULT_PATH.format(back_off_type)
    p, r, f1 = score_one(back_off_result_path, gold_key_path=_path.ALL_WORDS_TEST_KEY_PATH.format('ALL'),
                         print_logs=print_logs)
    attempt = len(open(back_off_result_path).readlines())
    correct = p * attempt
    return attempt, correct   # FS: 2081/2370 = 87.8


if __name__ == '__main__':
    # dataset = re.findall('-(.*?)$', sys.argv[1])[-1]
    # save_prefix = dataset + '/'
    # score_path = './params_score.txt'

    if len(sys.argv) == 3:
        score_one(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        score_one(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        print('Error!! Usage: python score.py answer_path gold_key_path [id column index]')

    ### 这几行测试了我们自己写的打分函数是否有问题,经过测试，完全没有问题
    system1 = _path.MFS_PATH.format('semcor')
    system2 = _path.ALL_WORDS_BASE_PATH + 'Output_Systems_ALL/IMSemb-s_semcor.key'
    # score_all(system1, _path.ALL_WORDS_TEST_KEY_PATH.format('ALL'), 0)

    # pss = []
    # ts = [1511351658,1511360115,1511378061,1511319633,1511335626,1511423501,1511395987,1511360115,1511413941,1511378061,1511395987,1511423501,1511335626,1511351658,1511413941,1511319633]
    # ts = [1510409468 ,1510498691 ,1510498691 ,1510409468 ,1510476307 ,1510476307 ,1510584282 ,1510539342 ,1510539342 ,1510584282 ,1510612001 ,1510612001 ,1510598197 ,1510598197 ]
    # rts = []
    # for f in os.listdir('../tmp/results'):
    #     t = int(re.findall('(\d+)-',f)[0])
    #     if t in ts and 'prob' not in f:
    #         if 'best' in f:
    #             rts.append([t,'best'])
    #         else:
    #             rts.append([t, 'ensemble'])
    #         ps = score_for_pos('ALL','../tmp/results/' + f, _path.ALL_WORDS_TEST_KEY_PATH.format('ALL'),0)
    #         pss.append(ps)
    # for i, ps in enumerate(pss):
    #     print '%d\t%s' % (rts[i][0],rts[i][1]),
    #     for p in ps:
    #         print '%.1f\t'%(p*100),
    #     print

    # for dataset in _path.ALL_WORDS_TEST_DATASET[1:]:
    #     path = _path.ALL_WORDS_TEST_PATH.format(dataset)
    #     get_pos_distribution(path)
    # get_pos_distribution(_path.ALL_WORDS_TRAIN_PATH.format('semcor'))

    # score_for_pos('ALL', _path.WNFS_PATH, _path.ALL_WORDS_TEST_KEY_PATH.format('ALL'))
    # p, r, f1 = score_one(_path.WNFS_PATH, _path.ALL_WORDS_TEST_KEY_PATH.format('ALL'), unsocre_dataset=_path.ALL_WORDS_VAL_DATASET)
    # print p, r, f1
    # p, r, f1 = score_one(_path.WNFS_PATH, _path.ALL_WORDS_TEST_KEY_PATH.format('ALL'))
    # print p, r, f1

    score_all('../tmp/results/0CAN-best-1-ALL', _path.ALL_WORDS_TEST_KEY_PATH.format('ALL'),
              print_logs=True, multi_sense=False)
    score_all('../tmp/results/0CAN-best-1-ALL', _path.ALL_WORDS_TEST_KEY_PATH.format('ALL'),
              print_logs=True, multi_sense=True)

    # print score_for_back_off_result('FS')
