# -*- coding: utf-8 -*-
import os

import lxml.etree as et
import math
import numpy as np
import collections
import re
import nltk.stem.porter as porter
from nltk.stem.wordnet import WordNetLemmatizer
from itertools import groupby
import random
from bs4 import BeautifulSoup
from bs4 import NavigableString
import pickle
from utils import path
from nltk.corpus.reader.wordnet import WordNetCorpusReader

_path = path.WSD_path()
PUNGAN_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#wn = WordNetCorpusReader(_path.WORDNET_PATH, '.*')  # 这种方式就会有函数补全
#print('wordnet version %s: %s' % (wn.get_version(), _path.WORDNET_PATH))

wi = -1  # for test
sj = -1  # for test

path_words_notin_vocab = PUNGAN_ROOT_PATH + '/WSD/tmp/words_notin_vocab.txt'

pos_dic = {
    'ADJ': u'a',
    'ADV': u'r',
    'NOUN': u'n',
    'VERB': u'v', }

POS_LIST = pos_dic.values()  # ['a', 'r', 'n', 'v']


def clean_context(ctx_in, has_target=False):
    replace_target = re.compile("<head.*?>.*</head>")
    replace_newline = re.compile("\n")
    replace_dot = re.compile("\.")
    replace_cite = re.compile("'")
    replace_frac = re.compile("[\d]*frac[\d]+")
    replace_num = re.compile("\s\d+\s")
    rm_context_tag = re.compile('<.{0,1}context>')
    rm_cit_tag = re.compile('\[[eb]quo\]')
    # rm_markup = re.compile('\[.+?\]')    # 会造成[]里面的文字丢失，比如semcor的d006.s007.t002
    rm_misc = re.compile("[\[\]\$`()%/,\.:;-]")

    ctx = replace_newline.sub(' ', ctx_in)  # (' <eop> ', ctx)
    if not has_target:
        ctx = replace_target.sub(' <target> ', ctx)

    ctx = replace_dot.sub(' ', ctx)  # .sub(' <eos> ', ctx)
    ctx = replace_cite.sub(' ', ctx)  # .sub(' <cite> ', ctx)
    ctx = replace_frac.sub(' <frac> ', ctx)
    ctx = replace_num.sub(' <number> ', ctx)
    ctx = rm_cit_tag.sub(' ', ctx)
    ctx = rm_context_tag.sub('', ctx)
    # ctx = rm_markup.sub('', ctx)
    ctx = rm_misc.sub('', ctx)

    return ctx


def split_context(ctx):
    # word_list = re.split(', | +|\? |! |: |; ', ctx.lower())
    word_list = [word for word in re.split('`|, | +|\? |! |: |; |\(|\)|_|,|\.|"|“|”|\'|\'', ctx.lower()) if word]
    # word_list = [word for word in re.split(' ', ctx.lower()) if word] # 这样子会把带inferences,的词分不开
    return word_list  # [stemmer.stem(word) for word in word_list]


def one_hot_encode(length, target):
    y = np.zeros(length, dtype=np.float32)
    y[target] = 1.
    return y


def load_train_data(dataset):
    if dataset in _path.LS_DATASET:
        return load_lexical_sample_data(_path.LS_TRAIN_PATH.format(dataset), True)
    elif dataset in _path.ALL_WORDS_TRAIN_DATASET:
        return load_all_words_data(_path.ALL_WORDS_TRAIN_PATH.format(dataset),
                                   _path.ALL_WORDS_TRAIN_KEY_PATH.format(dataset),
                                   _path.ALL_WORDS_DIC_PATH.format(dataset), True)
    else:
        raise ValueError('%s, %s. Provided: %s' % (
            ','.join(_path.LS_DATASET), ','.join(_path.ALL_WORDS_TRAIN_DATASET), dataset))


def load_val_data(dataset):
    if dataset in _path.LS_DATASET:
        return load_lexical_sample_data(_path.LS_VAL_PATH.format(dataset), True)  # 反正P、U都会预测错，所以在这儿放弃这些，提高正确率
    elif dataset in _path.ALL_WORDS_TEST_DATASET:
        return load_all_words_data(_path.ALL_WORDS_TEST_PATH.format(dataset),
                                   _path.ALL_WORDS_TEST_KEY_PATH.format(dataset), None, False)
    else:
        raise ValueError('%s, %s. Provided: %s' % (
            ','.join(_path.LS_DATASET), ','.join(_path.ALL_WORDS_TEST_DATASET), dataset))


def load_test_data(dataset):
    if dataset in _path.LS_DATASET:
        return load_lexical_sample_data(_path.LS_TEST_PATH.format(dataset), False)
    elif dataset in _path.ALL_WORDS_TEST_DATASET:
        return load_all_words_data(_path.ALL_WORDS_TEST_PATH.format(dataset), False)
    else:
        raise ValueError('%s, %s. Provided: %s' % (
            ','.join(_path.LS_DATASET), ','.join(_path.ALL_WORDS_TEST_DATASET), dataset))


def load_plain_test_data(file_path, index_place="end", target_words=None):
    """
    load plain txt test data with multi-sense word index.
    :param file_path: path of the input file.
    :param index_place: multi-sense word index. "end" means index is append in the end,
    "start" means index is insert in the start.
    :return: a list of data
    """
    data = []
    with open(file_path) as f:
        for id, line in enumerate(f):
            line = line.strip()
            if index_place == "end":
                context = re.sub("\d$", "", line)
                index = int(re.findall("(\d)$", line)[0])
            else:
                context = re.sub("^\d ", "", line)
                index = int(re.findall("^(\d) ", line)[0])
            context = context.split(' ')
            target_word = context[index] + "#v"
            # target_word = WordNetLemmatizer().lemmatize(target_word)
            if target_words is not None and target_word not in target_words:
                # print(target_word)
                continue
            context[index] = '<target>'
            pos_list = ['<pad>'] * len(context)
            x = {
                'id': str(id) + '-' + str(index),
                'context': context,
                'target_sense': None,
                'target_word': target_word,
                'poss': pos_list,
            }
            data.append(x)
    return data


def load_lexical_sample_data(path, is_training=None):
    data = []
    doc = BeautifulSoup(open(path), 'xml')
    instances = doc.find_all('instance')

    for instance in instances:
        answer = None
        context = None
        for child in instance.children:
            if isinstance(child, NavigableString):
                continue
            if child.name == 'answer':
                senseid = child.get('senseid')
                if senseid == 'P' or senseid == 'U':
                    pass  # ignore in wsd_bilstm-salomons
                # else:    # 记录最后一个非P Usense
                elif not answer:  # 记录最前面一个非P U的 sense
                    answer = senseid
            elif child.name == 'context':
                context = child.prettify()
            else:
                print(child.name)
                print(instance.text)
                raise ValueError('unknown child tag to instance')

        # if valid
        if (is_training and answer and context) or (not is_training and context):
            context = clean_context(context)
            context = split_context(context)
            lemma = instance.get('id').split('.')[0]
            pos = instance.get('id').split('.')[1]
            if pos in POS_LIST:
                word = lemma + '#' + pos
            else:
                word = lemma  # SE2 don't have and don't need pos
            pos_list = ['<pad>'] * len(context)
            x = {
                'id': instance.get('id'),
                'context': context,
                'target_sense': answer,  # todo support multiple answers?
                'target_word': word,
                'poss': pos_list,
            }

            data.append(x)

    return data


def load_all_words_data(data_path, key_path=None, dic_path=None, is_training=None):
    word_count_info = {}
    if dic_path:
        soup = BeautifulSoup(open(dic_path), 'lxml')
        for lexelt_tag in soup.find_all('lexelt'):
            lemma = lexelt_tag['item']
            sense_count_wn = int(lexelt_tag['sence_count_wn'])
            sense_count_corpus = int(lexelt_tag['sense_count_corpus'])
            word_count_info[lemma] = [sense_count_wn, sense_count_corpus]

    id_to_sensekey = {}
    if key_path:
        for line in open(key_path).readlines():
            id = line.split()[0]
            sensekey = line.split()[1]  # 取第一个义项
            id_to_sensekey[id] = sensekey

    context = et.iterparse(data_path, tag='sentence')

    data = []
    poss = set()
    for event, elem in context:
        sent_list = []
        pos_list = []
        for child in elem:
            word = child.get('lemma').lower()  # 有一些lemma也是大写，造成在glove中查不到，比如<wf lemma="Heights" pos="NOUN">Heights</wf>
            # word = child.get('lemma')
            # word = child.text
            sent_list.append(word)
            pos = child.get('pos')
            pos_list.append(pos)
            poss.add(pos)

        i = -1
        for child in elem:
            if child.tag == 'wf':
                i += 1
            elif child.tag == 'instance':
                i += 1
                id = child.get('id')
                lemma = child.get('lemma')
                pos = child.get('pos')
                word = lemma + '#' + pos_dic[pos]
                if key_path:
                    sensekey = id_to_sensekey[id]
                else:
                    sensekey = None
                if is_training:
                    # 单义词或者在训练集中只是出现一次的词语丢弃
                    if word_count_info[word][0] == 1 or word_count_info[word][1] == 1:
                        continue  ## wuwuwuwu,之前写成break，踩了好久的坑啊！！！

                context = sent_list[:]
                context[i] = '<target>'
                # context = ' '.join(context)
                # context = clean_context(context, has_target=True)
                x = {
                    'id': id,
                    'context': context,  # lfl: all words task is list
                    'target_sense': sensekey,  # todo support multiple answers?
                    # 'target_word': lemma+'#'+pos_dic[pos],
                    'target_word': word,
                    'poss': pos_list,
                }
                data.append(x)

    if is_training:
        poss_list = ['<pad>', '<eos>'] + list(sorted(poss))
        print 'Wirting to tmp/pos_dic.pkl:' + ' '.join(poss_list)
        poss_map = dict(zip(poss_list, range(len(poss_list))))
        with open(PUNGAN_ROOT_PATH + '/WSD/tmp/pos_dic.pkl', 'wb') as f:
            pickle.dump((poss_map), f)

    return data


def get_lexelts(dataset):
    items = []
    path = _path.LS_TRAIN_PATH.format(dataset)
    parser = et.XMLParser(dtd_validation=True)
    doc = et.parse(path, parser)
    instances = doc.findall('.//lexelt')

    for instance in instances:
        items.append(instance.get('item'))

    return items


def target_to_lexelt_map(target_words, lexelts):
    # assert len(target_words) == len(lexelts)

    res = {}
    for lexelt in lexelts:
        base = lexelt.split('.')[0]
        res[base] = lexelt
    if 'colourless' in res:
        res['colorless'] = res['colourless']  # senseval2 errors

    return res


def build_sense_ids_for_all(data):
    counter = collections.Counter()
    for elem in data:
        counter.update([elem['answer']])

    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    senses, _ = list(zip(*count_pairs))
    sense_to_id = dict(zip(senses, range(len(senses))))

    return sense_to_id


def build_sense_ids(data):
    words = set()
    word_to_senses = {}
    for elem in data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']
        if target_sense is None:
            continue
        if target_word not in words:
            words.add(target_word)
            word_to_senses.update({target_word: [target_sense]})

        else:
            if target_sense not in word_to_senses[target_word]:
                word_to_senses[target_word].append(target_sense)

    words = list(words)
    target_word_to_id = dict(zip(words, range(len(words))))
    target_sense_to_id = [dict(zip(word_to_senses[word], range(len(word_to_senses[word])))) for word in words]

    n_senses_from_word_id = dict([(target_word_to_id[word], len(word_to_senses[word])) for word in words])
    return target_word_to_id, target_sense_to_id, len(words), n_senses_from_word_id


def build_vocab(data):
    """
    :param data: list of dicts containing attribute 'context'
    :return: a dict with words as key and ids as value
    """
    counter = collections.Counter()
    for elem in data:
        counter.update(elem['context'])  # context 里面就有<target>
        counter.update([elem['target_word']])  # 带 pos 标记的

    try:   # 为什么要加入path_words_notin_vocab？没想通
        words_notin_vocab = open(path_words_notin_vocab).read().split('\n')
        counter.update(words_notin_vocab)
    except Exception as e:
        print e

    # remove infrequent words
    min_freq = 1
    filtered = [item for item in counter.items() if item[1] >= min_freq]

    count_pairs = sorted(filtered, key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    # add_words = ['<pad>', '<dropped>', '<head>', '<tail>'] + list(words)
    print('old:35758, new list(words) length is', len(list(words)))
    add_words = ['<pad>', '<eos>'] + list(words)#[:185899]#[:35758]#
    print('len(add_words)', len(add_words))
    word_to_id = dict(zip(add_words, range(len(add_words))))
    print('pad id:%s' % (word_to_id['<pad>']))
    print('eos id:%s' % (word_to_id['<eos>']))  # split_context 会去掉<>， 所以这儿没有<>
    return word_to_id


def store_notin_vocab_words(words_notin_vocab, clean=False):
    if clean:
        old = []
    else:
        try:
            old = open(path_words_notin_vocab).read()
            old = old.split('\n')
        except Exception as e:
            old = []

    ws = []
    for word in words_notin_vocab:
        try:
            word.decode('ascii')
            ws.append(word)
        except Exception as e:
            continue
    new = set(ws + old)
    open(path_words_notin_vocab, 'w').write('\n'.join(new))

def convert_to_numeric_pungan(data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id,
                       ignore_sense_not_in_train=False, data_type='labeled'):
    n_senses_sorted_by_target_id = [n_senses_from_target_id[target_id] for target_id in
                                    range(len(n_senses_from_target_id))]
    starts = (np.cumsum(np.append([0], n_senses_sorted_by_target_id)))[:-1]

    words_notin_vocab = []

    def get_tot_id(target_id, sense_id):
        return starts[target_id] + sense_id

    with open(PUNGAN_ROOT_PATH + '/WSD/tmp/pos_dic.pkl', 'rb') as f:
        pos_to_id = pickle.load(f)

    all_data = []
    ids_pungan = []
    target_tag_id = word_to_id['<target>']
    test_instance_sensekey_not_in_train = []
    for insi, instance in enumerate(data):
        # if insi < 10:
        #     print instance['target_word']


        words = instance['context']
        poss = instance['poss']
        # words = ['<head>'] + words + ['<tail>']  ## lfl:add head and tail for each sentence
        # poss = ['HEAD'] + poss + ['TAIL']
        assert len(poss) == len(words)
        ctx_ints = []
        pos_ints = []
        for i, word in enumerate(words):
            if word in word_to_id:
                ctx_ints.append(word_to_id[word])
                pos_ints.append(pos_to_id[poss[i]])
            elif len(word) > 0:
                words_notin_vocab.append(word)
        # print('ctx_ints', ctx_ints)
        if target_tag_id not in ctx_ints:
            continue
        stop_idx = ctx_ints.index(target_tag_id)
        xf = np.array(ctx_ints[:stop_idx], dtype=np.int32)
        pf = np.array(pos_ints[:stop_idx], dtype=np.int32)
        xb = np.array(ctx_ints[stop_idx + 1:], dtype=np.int32)[::-1]  # [::-1] 表示逆序
        pb = np.array(pos_ints[stop_idx + 1:], dtype=np.int32)[::-1]

        instance_id = instance['id']
        target_word = instance['target_word']
        target_sense = instance['target_sense']
        if target_word not in target_word_to_id:
            continue
        target_id = target_word_to_id[target_word]
        senses = target_sense_to_id[target_id]
        if target_sense in senses:
            sense_id = senses[target_sense]
        else:
            test_instance_sensekey_not_in_train.append([instance_id, target_sense])
            if ignore_sense_not_in_train:  # 忽略不在训练集的义项（直接不进行预测，结果是正确率上升，召回率不变）
                continue
            else:
                sense_id = 0  # 把测试集的key不在训练集的都分为第0类吧
        instance = Instance()
        instance.id = instance_id
        instance.xf = xf
        instance.xb = xb
        instance.pf = pf
        instance.pb = pb
        instance.fake = 1.0 if data_type == 'fake' else 0.
        instance.labeled = 1.0 if data_type == 'labeled' else 0.
        instance.real = 1.0 if data_type in ['labeled', 'unlabeled'] else 0.
        instance.target_word_id = word_to_id[target_word]
        instance.target_pos_id = pos_ints[stop_idx]
        instance.target_id = target_id
        instance.sense_id = sense_id
        instance.one_hot_labels = one_hot_encode(n_senses_from_target_id[target_id], sense_id)
        # instance.one_hot_labels = one_hot_encode(tot_n_senses, get_tot_id(target_id, sense_id))

        all_data.append(instance)
        ids_pungan.append(insi)
    tmp_lenth = len(test_instance_sensekey_not_in_train)
    if tmp_lenth:  ## training set lenth =0
        print('###test_instance_sensekey_not_in_train: %s' % (tmp_lenth))

    store_notin_vocab_words(words_notin_vocab)

    return all_data, ids_pungan

def convert_to_numeric(data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id,
                       ignore_sense_not_in_train=False, data_type='labeled'):
    n_senses_sorted_by_target_id = [n_senses_from_target_id[target_id] for target_id in
                                    range(len(n_senses_from_target_id))]
    starts = (np.cumsum(np.append([0], n_senses_sorted_by_target_id)))[:-1]

    words_notin_vocab = []

    def get_tot_id(target_id, sense_id):
        return starts[target_id] + sense_id

    with open(PUNGAN_ROOT_PATH + '/WSD/tmp/pos_dic.pkl', 'rb') as f:
        pos_to_id = pickle.load(f)

    all_data = []

    target_tag_id = word_to_id['<target>']
    test_instance_sensekey_not_in_train = []
    for insi, instance in enumerate(data):
        # if insi < 10:
        #     print instance['target_word']


        words = instance['context']
        poss = instance['poss']
        # words = ['<head>'] + words + ['<tail>']  ## lfl:add head and tail for each sentence
        # poss = ['HEAD'] + poss + ['TAIL']
        assert len(poss) == len(words)
        ctx_ints = []
        pos_ints = []
        for i, word in enumerate(words):
            if word in word_to_id:
                ctx_ints.append(word_to_id[word])
                pos_ints.append(pos_to_id[poss[i]])
            elif len(word) > 0:
                words_notin_vocab.append(word)
        # print('ctx_ints', ctx_ints)
        if target_tag_id not in ctx_ints:
            continue
        stop_idx = ctx_ints.index(target_tag_id)
        xf = np.array(ctx_ints[:stop_idx], dtype=np.int32)
        pf = np.array(pos_ints[:stop_idx], dtype=np.int32)
        xb = np.array(ctx_ints[stop_idx + 1:], dtype=np.int32)[::-1]  # [::-1] 表示逆序
        pb = np.array(pos_ints[stop_idx + 1:], dtype=np.int32)[::-1]

        instance_id = instance['id']
        target_word = instance['target_word']
        target_sense = instance['target_sense']
        if target_word not in target_word_to_id:
            continue
        target_id = target_word_to_id[target_word]
        senses = target_sense_to_id[target_id]
        if target_sense in senses:
            sense_id = senses[target_sense]
        else:
            test_instance_sensekey_not_in_train.append([instance_id, target_sense])
            if ignore_sense_not_in_train:  # 忽略不在训练集的义项（直接不进行预测，结果是正确率上升，召回率不变）
                continue
            else:
                sense_id = 0  # 把测试集的key不在训练集的都分为第0类吧
        instance = Instance()
        instance.id = instance_id
        instance.xf = xf
        instance.xb = xb
        instance.pf = pf
        instance.pb = pb
        instance.fake = 1.0 if data_type == 'fake' else 0.
        instance.labeled = 1.0 if data_type == 'labeled' else 0.
        instance.real = 1.0 if data_type in ['labeled', 'unlabeled'] else 0.
        instance.target_word_id = word_to_id[target_word]
        instance.target_pos_id = pos_ints[stop_idx]
        instance.target_id = target_id
        instance.sense_id = sense_id
        instance.one_hot_labels = one_hot_encode(n_senses_from_target_id[target_id], sense_id)
        # instance.one_hot_labels = one_hot_encode(tot_n_senses, get_tot_id(target_id, sense_id))

        all_data.append(instance)

    tmp_lenth = len(test_instance_sensekey_not_in_train)
    if tmp_lenth:  ## training set lenth =0
        print('###test_instance_sensekey_not_in_train: %s' % (tmp_lenth))

    store_notin_vocab_words(words_notin_vocab)

    return all_data


def group_by_target(ndata):
    res = {}
    for key, group in groupby(ndata, lambda inst: inst.target_id):
        res.update({key: list(group)})
    return res


def split_grouped(data, frac, min=None):
    assert frac > 0.
    assert frac < .5
    l = {}
    r = {}
    for target_id, instances in data.iteritems():
        # instances = [inst for inst in instances]
        random.shuffle(instances)  # optional
        n = len(instances)
        n_r = int(frac * n)
        if min and n_r < min:
            n_r = min
        n_l = n - n_r

        l[target_id] = instances[:n_l]
        r[target_id] = instances[-n_r:]

    return l, r


def batchify_grouped(gdata, n_step_f, n_step_b, pad_id, n_senses_from_target_id):
    res = {}
    for target_id, instances in gdata.iteritems():
        batch_size = len(instances)
        xfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        xbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        xfs.fill(pad_id)
        xbs.fill(pad_id)

        # x forward backward
        for j in range(batch_size):
            n_to_use_f = min(n_step_f, len(instances[j].xf))
            n_to_use_b = min(n_step_b, len(instances[j].xb))
            # print(instances[j].xb)
            xfs[j, -n_to_use_f:] = instances[j].xf[-n_to_use_f:]
            xbs[j, -n_to_use_b:] = instances[j].xb[-n_to_use_b:]

        # labels
        labels = np.zeros([batch_size, n_senses_from_target_id[target_id]], np.float32)
        for j in range(batch_size):
            labels[j, instances[j].sense_id] = 1.

        res[target_id] = (xfs, xbs, labels)

    return res


class Instance:
    pass


def batch_generator(is_training, batch_size, data, dict_data, pad_id, n_step_f, n_step_b, pad_last_batch=False):
    data_len = len(data)
    n_batches_float = data_len / float(batch_size)
    n_batches = int(math.ceil(n_batches_float)) if pad_last_batch else int(n_batches_float)
    if is_training:
        random.shuffle(data)

    for i in range(n_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]

        # context word
        xfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        xbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        xfbs = np.zeros([batch_size, n_step_f + n_step_b + 1], dtype=np.int32)
        xfs.fill(pad_id)
        xbs.fill(pad_id)
        xfbs.fill(pad_id)

        # context pos
        pfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        pbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        pfbs = np.zeros([batch_size, n_step_f + n_step_b + 1], dtype=np.int32)  # 0 is pad for pos, no need pad_id

        # x forward backward
        for j in range(batch_size):
            if i * batch_size + j < data_len:

                n_to_use_f = min(n_step_f, len(batch[j].xf))
                n_to_use_b = min(n_step_b, len(batch[j].xb))
                if n_to_use_f:
                    xfs[j, -n_to_use_f:] = batch[j].xf[-n_to_use_f:]
                    pfs[j, -n_to_use_f:] = batch[j].pf[-n_to_use_f:]
                if n_to_use_b:
                    xbs[j, -n_to_use_b:] = batch[j].xb[-n_to_use_b:]
                    pbs[j, -n_to_use_b:] = batch[j].pb[-n_to_use_b:]

                xfbs[j] = np.concatenate((xfs[j], [batch[j].target_word_id], xbs[j][::-1]), axis=0)
                pfbs[j] = np.concatenate((pfs[j], [batch[j].target_pos_id], pbs[j][::-1]), axis=0)

        # id
        instance_ids = [inst.id for inst in batch]

        # labels
        target_ids = [inst.target_id for inst in batch]
        sense_ids = [inst.sense_id for inst in batch]
        fakes = [inst.fake for inst in batch]
        reals = [inst.real for inst in batch]
        labeleds = [inst.labeled for inst in batch]
        if len(target_ids) < batch_size:  # padding
            n_pad = batch_size - len(target_ids)
            # print('Batch padding size: %d'%(n_pad))
            target_ids += [0] * n_pad
            sense_ids += [0] * n_pad
            fakes += [0.] * n_pad
            reals += [0.] * n_pad
            labeleds += [0.] * n_pad
            instance_ids += [0] * n_pad  # instance_ids += [''] * n_pad

        target_ids = np.array(target_ids, dtype=np.int32)
        sense_ids = np.array(sense_ids, dtype=np.int32)
        fakes = np.array(fakes, dtype=np.float32)
        reals = np.array(reals, dtype=np.float32)
        labeleds = np.array(labeleds, dtype=np.float32)
        # one_hot_labels = np.vstack([inst.one_hot_labels for inst in batch])

        glosses_ids = [dict_data[0][target_ids[i]] for i in
                       range(batch_size)]  # [batch_size, max_n_sense, max_gloss_words]
        synsets_ids = [dict_data[1][target_ids[i]] for i in range(batch_size)]
        glosses_lenth = [dict_data[2][target_ids[i]] for i in range(batch_size)]
        synsets_lenth = [dict_data[3][target_ids[i]] for i in range(batch_size)]
        sense_mask = [dict_data[4][target_ids[i]] for i in range(batch_size)]

        for i, target_id in enumerate(target_ids):
            if target_id == wi and sense_ids[i] == sj:
                j = sense_ids[i]
                id_to_word = {}
                for word in word_to_id:
                    id_to_word[word_to_id[word]] = word
                print '!!!gloss No.!!! ' + str(j)
                g = [id_to_word[w] for w in glosses_ids[i][j]]
                print ' '.join(g)
                print('gloss lenth', glosses_lenth[i][j])

        yield (xfs, xbs, xfbs, pfs, pbs, pfbs, target_ids, sense_ids, instance_ids, glosses_ids, synsets_ids, glosses_lenth,
        synsets_lenth, sense_mask, fakes, reals, labeleds)


def batch_generator_hyp(is_training, batch_size, data, dict_data, pad_id, n_step_f, n_step_b, n_hyper, n_hypo,
                        pad_last_batch=False):
    data_len = len(data)
    n_batches_float = data_len / float(batch_size)
    n_batches = int(math.ceil(n_batches_float)) if pad_last_batch else int(n_batches_float)

    if is_training:
        random.shuffle(data)

    for i in range(n_batches):
        batch = data[i * batch_size:(i + 1) * batch_size]

        # context word
        xfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        xbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        xfbs = np.zeros([batch_size, n_step_f + n_step_b + 1], dtype=np.int32)
        xfs.fill(pad_id)
        xbs.fill(pad_id)
        xfbs.fill(pad_id)

        # context pos
        pfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        pbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        pfbs = np.zeros([batch_size, n_step_f + n_step_b + 1], dtype=np.int32)  # 0 is pad for pos, no need pad_id

        # x forward backward
        for j in range(batch_size):
            if i * batch_size + j < data_len:
                n_to_use_f = min(n_step_f, len(batch[j].xf))
                n_to_use_b = min(n_step_b, len(batch[j].xb))
                if n_to_use_f:
                    xfs[j, -n_to_use_f:] = batch[j].xf[-n_to_use_f:]
                    pfs[j, -n_to_use_f:] = batch[j].pf[-n_to_use_f:]
                if n_to_use_b:
                    xbs[j, -n_to_use_b:] = batch[j].xb[-n_to_use_b:]
                    pbs[j, -n_to_use_b:] = batch[j].pb[-n_to_use_b:]
                xfbs[j] = np.concatenate((xfs[j], [batch[j].target_word_id], xbs[j][::-1]), axis=0)
                pfbs[j] = np.concatenate((pfs[j], [batch[j].target_pos_id], pbs[j][::-1]), axis=0)

        # id
        instance_ids = [inst.id for inst in batch]

        # labels
        target_ids = [inst.target_id for inst in batch]
        sense_ids = [inst.sense_id for inst in batch]

        if len(target_ids) < batch_size:  # padding
            n_pad = batch_size - len(target_ids)
            # print('Batch padding size: %d'%(n_pad))
            target_ids += [0] * n_pad
            sense_ids += [0] * n_pad
            instance_ids += [0] * n_pad  # instance_ids += [''] * n_pad

        target_ids = np.array(target_ids, dtype=np.int32)
        sense_ids = np.array(sense_ids, dtype=np.int32)
        # one_hot_labels = np.vstack([inst.one_hot_labels for inst in batch])

        # [gloss_to_id, gloss_lenth, sense_mask, hyper_lenth, hypo_lenth]
        glosses_ids = [dict_data[0][target_ids[i]] for i in
                       range(batch_size)]  # [batch_size, max_n_sense, n_hyp, max_gloss_words]
        glosses_lenth = [dict_data[1][target_ids[i]] for i in range(batch_size)]  # [batch_size, max_n_sense]
        sense_mask = [dict_data[2][target_ids[i]] for i in range(batch_size)]  # [batch_size, max_n_sense, mask_size]
        hyper_lenth = [dict_data[3][target_ids[i]] for i in range(batch_size)]  # [batch_size, max_n_sense]
        hypo_lenth = [dict_data[4][target_ids[i]] for i in range(batch_size)]  # [batch_size, max_n_sense]

        #  注意train.py的时候一定要注释掉这行代码，因为会有 NameError: global name 'word_to_id' is not defined
        # for i, target_id in enumerate(target_ids):
        #     if target_id == 607 and sense_ids[i] == 2:
        #         j = sense_ids[i]
        #         id_to_word = {}
        #         for word in word_to_id:
        #             id_to_word[word_to_id[word]] = word
        #         print '!!!gloss hyper lenth: %s '%hyper_lenth[i][j]
        #         for k in range(hyper_lenth[i][j]):
        #             kk = n_hyper-hyper_lenth[i][j] + k
        #             g = [id_to_word[w] for w in glosses_ids[i][j][kk]]
        #             print('hyper '+ str(n_hyper-kk) +' '.join(g))
        #         print '!!!gloss hypo lenth: %s'%hypo_lenth[i][j]
        #         for k in range(hypo_lenth[i][j]):
        #             kk = n_hyper + 1 + k
        #             g = [id_to_word[w] for w in glosses_ids[i][j][kk]]
        #             print('hypo '+ str(k+1) +' '.join(g))
        #         print('gloss lenth',glosses_lenth[i][j])

        yield (xfs, xbs, xfbs, pfs, pbs, pfbs, target_ids, sense_ids, instance_ids, glosses_ids,
               glosses_lenth, sense_mask, hyper_lenth, hypo_lenth)


# get gloss and synset from dictionary.xml
# luofuli add @2017/7/28
def load_dictionary(dataset, target_words=None, expand_type=0, n_hyper=3, n_hypo=3):
    gloss_dic = {}
    synset_dic = {}

    print('&n_hyper:%s\t&n_hypo：%s' % (n_hyper, n_hypo))

    if dataset in _path.LS_DATASET:
        dic_path = _path.LS_DIC_PATH.format(dataset)
        target_words = None  # Lexical task don't need target words filter
    elif dataset in _path.ALL_WORDS_TRAIN_DATASET:
        dic_path = _path.ALL_WORDS_DIC_PATH.format(dataset)
    else:
        raise ValueError(
            '%s or %s. Provided: %s' % (','.join(_path.LS_DATASET), ','.join(_path.ALL_WORDS_TRAIN_DATASET), dataset))

    soup = BeautifulSoup(open(dic_path), 'lxml')
    all_sense_tag = soup.find_all('sense')
    for sense_tag in all_sense_tag:
        id = sense_tag['id']
        key = id  # all-words
        # key = id.replace("-", "'")  # senseval2_LS README EG:pull_in_one-s_horns%2:32:00::
        synset = sense_tag['synset']
        gloss = sense_tag['gloss']
        if expand_type in [1, 2, 3]:
            gloss = expand_gloss(key, expand_type)
        elif expand_type == 4:
            gloss = expand_gloss_list(key, n_hyper, n_hypo)
        if target_words:  ## 全词的任务并不是加载训练集的所有词典，而是加载训练集和测试集重合的target_word的词典
            target_word = sense_tag.parent['item']
            if target_word in target_words:
                synset_dic[id] = synset
                gloss_dic[id] = gloss
        else:  ## Lexical example task
            synset_dic[id] = synset
            gloss_dic[id] = gloss
    return gloss_dic, synset_dic


def expand_gloss(key, expand_type):
    try:
        lemma = wn.lemma_from_key(key)
    except Exception as e:
        print e
        print key
        exit(-1)
    synset = lemma.synset()
    if expand_type == 1:  # 'hyper':
        h = list(synset.closure(lambda s: s.hypernyms(), 3))
    elif expand_type == 2:
        h = list(synset.closure(lambda s: s.hyponyms(), 3))  # 反正会cut掉一部分，所以深度为3
    else:
        h1 = list(synset.closure(lambda s: s.hypernyms(), 3))
        h2 = list(synset.closure(lambda s: s.hyponyms(), 3))
        h2.reverse()
        h = h1 + h2

    glosses = [synset.definition()]
    for i, s in enumerate(h):
        glosses.append(s.definition())
    if expand_type == 3 and h != []:
        glosses.append(synset.definition())  # target->上->下->target
    # glosses.reverse()  # 影响截取最大长度，增加了这行这个记得把 words = words[:max_gloss_words] 改为 words = words[-max_gloss_words:]

    r = ' <eos> '.join(glosses)
    return r


def expand_gloss1(key, expand_type):
    lemma = wn.lemma_from_key(key)
    if expand_type == 1:  # 'hyper':
        h = list(lemma.closure(lambda s: s.hypernyms(), 2))
    else:
        h = list(lemma.closure(lambda s: s.hyponyms(), 2))

    if len(h) == 0:
        glosses = []
    else:
        glosses = [s.synset().definition() for s in h]
    glosses.insert(0, lemma.synset().definition())

    return ' eos '.join(glosses)


def expand_gloss_list(key, n_hyper, n_hypo):
    lemma = wn.lemma_from_key(key)
    synset = lemma.synset()
    hyper = list(synset.closure(lambda s: s.hypernyms(), n_hyper))[:n_hyper]
    hypo = list(synset.closure(lambda s: s.hyponyms(), n_hypo))[:n_hypo]
    # hyper.reverse()   # No need

    glosses = [''] * (n_hyper + n_hypo + 1)  # hyper3 hyper2 hyper1 target hypo1 hypo2 hypo3
    for i, s in enumerate(hyper):
        glosses[n_hyper - 1 - i] = s.definition()  # reverse is here
    glosses[n_hyper] = synset.definition()
    for i, s in enumerate(hypo):
        glosses[n_hyper + 1 + i] = s.definition()  # no reverse

    return glosses  # lenth: [n_hypo+1+n_hyper]


# make initial sense id(in dataset) to new sense id, and make numeric for gloss defination and synset words
# add @luofuli 2017/7/28
def bulid_dictionary_id(gloss_dict, synset_dict, target_sense_to_id, word_to_id, pad_id, mask_size, max_gloss_words=100,
                        max_synset_words=5, is_empty=False):
    t_max_gloss_words = max([len(split_context(g)) for g in gloss_dict.values()])
    print('initial max_gloss_words: %s' % (t_max_gloss_words))
    t_max_synset_words = max([len(split_context(s)) for s in synset_dict.values()])
    print('initial max_synset_words: %s' % (t_max_synset_words))
    n_target_words = len(target_sense_to_id)
    print('n_target_words: %s' % n_target_words)
    max_n_sense = max([len(sense_to_ids) for sense_to_ids in target_sense_to_id])
    print('max_n_sense %d' % (max_n_sense))
    gloss_to_id = np.zeros([n_target_words, max_n_sense, max_gloss_words], dtype=np.int32)
    synset_to_id = np.zeros([n_target_words, max_n_sense, max_synset_words], dtype=np.int32)
    gloss_to_id.fill(pad_id)
    synset_to_id.fill(pad_id)

    words_notin_vocab = []

    gloss_lenth = np.zeros([n_target_words, max_n_sense], dtype=np.int32)  # 每个gloss包含的word数目
    synset_lenth = np.zeros([n_target_words, max_n_sense], dtype=np.int32)  # 同上
    sense_mask = np.zeros([n_target_words, max_n_sense, mask_size], dtype=np.int32)  # 为了使得不是sense的部分失效
    for i, sense_to_ids in enumerate(target_sense_to_id):  # target_sense_to_id 是按照target_word的target_word_to_id中的id排序的
        if i % 500 == 0:
            print("Bulid dictionary: %s/%s" % (i, len(target_sense_to_id)))
        # for j, id0 in enumerate(sense_to_ids):  # old bug, dict sense order is not the sense_id,sense_id是按照在文章中出现的先后顺序，而dict.keys()的顺序不是这

        for id0 in sense_to_ids:  # id0 is the initial id in dataset
            j = sense_to_ids[id0]
            gloss_words = split_context(gloss_dict[id0])
            synset_words = split_context(synset_dict[id0])
            sense_mask[i][j][:] = 1
            words = []
            for word in gloss_words:
                if word in word_to_id:
                    words.append(word_to_id[word])
                elif len(word) > 0:
                    words_notin_vocab.append(word)

            if not is_empty:
                words = words[:max_gloss_words]  # 去掉多余的gloss words
            else:
                # random: test gloss in memory network work or not
                # words = np.random.random_integers(len(word_to_id)-1, size=(len(words)))
                # all <pad>=0: test gloss in memory network work or not
                words = [word_to_id['<pad>']] * max_gloss_words

            if len(words) > 0:
                gloss_to_id[i, j, :len(words)] = words  # pad in the end
                gloss_lenth[i][j] = len(words)

            # if i == wi and j == sj:
            #     print '!!!gloss: {}  sense:{}'.format(str(j), id0)
            #     id_to_word = {}
            #     for word in word_to_id:
            #         id_to_word[word_to_id[word]] = word
            #     g = [id_to_word[w] for w in words]
            #     print(' '.join(g))

            words = []
            for word in synset_words:
                if word in word_to_id:
                    words.append(word_to_id[word])
                elif len(word) > 0:
                    words_notin_vocab.append(word)
            if len(words) > 0:
                # words = np.random.random_integers(len(word_to_id)-1, size=(len(words)))   ## test memory network work or not
                words = words[:max_synset_words]
                synset_to_id[i, j, :len(words)] = words  ## pad in the end
                synset_lenth[i][j] = len(words)
    store_notin_vocab_words(words_notin_vocab)
    return [gloss_to_id, synset_to_id, gloss_lenth, synset_lenth, sense_mask], max_n_sense


# 为扩增后的gloss list
# add @luofuli 2018/1/16
def bulid_dictionary_id_hyp(gloss_dict, target_sense_to_id, word_to_id, pad_id, mask_size, max_gloss_words, n_hyper,
                            n_hypo, is_empty=False):
    n_target_words = len(target_sense_to_id)
    n_hy = n_hyper + n_hypo + 1
    print('n_target_words: %s' % n_target_words)
    max_n_sense = max([len(sense_to_ids) for sense_to_ids in target_sense_to_id])
    print('max_n_sense %d' % (max_n_sense))
    gloss_to_id = np.zeros([n_target_words, max_n_sense, n_hy, max_gloss_words], dtype=np.int32)
    gloss_to_id.fill(pad_id)

    words_notin_vocab = []

    gloss_lenth = np.zeros([n_target_words, max_n_sense, n_hy], dtype=np.int32)  ## 每个gloss包含的word数目
    hyper_lenth = np.zeros([n_target_words, max_n_sense], dtype=np.int32)
    hypo_lenth = np.zeros([n_target_words, max_n_sense], dtype=np.int32)
    sense_mask = np.zeros([n_target_words, max_n_sense, mask_size], dtype=np.int32)  ## 为了使得不是sense的部分失效
    for i, sense_to_ids in enumerate(target_sense_to_id):  # target_sense_to_id 是按照target_word的target_word_to_id中的id排序的
        if i % 500 == 0:
            print("Bulid dictionary: %s/%s" % (i, len(target_sense_to_id)))
        ## for j, id0 in enumerate(sense_to_ids):  # old bug, dict sense order is not the sense_id,sense_id是按照在文章中出现的先后顺序，而dict.keys()的顺序不是这

        for senseid in sense_to_ids:  # senseid is like have%2:40:00::
            j = sense_to_ids[senseid]
            gloss_list = gloss_dict[senseid]
            sense_mask[i][j][:] = 1
            for k, gloss in enumerate(gloss_list):
                if gloss == ['']:
                    continue

                gloss_words = split_context(gloss)
                words = []
                for word in gloss_words:
                    if word in word_to_id:
                        words.append(word_to_id[word])
                    elif len(word) > 0:
                        words_notin_vocab.append(word)

                if not is_empty:
                    words = words[:max_gloss_words]  # 去掉多余的gloss words
                else:
                    # random: test gloss in memory network work or not
                    # words = np.random.random_integers(len(word_to_id)-1, size=(len(words)))
                    # all <pad>=0: test gloss in memory network work or not
                    words = [word_to_id['<pad>']] * max_gloss_words

                if len(words) > 0:
                    gloss_to_id[i, j, k, :len(words)] = words  ## pad in the end
                    gloss_lenth[i][j][k] = len(words)
                    hyper_lenth[i][j] = max(hyper_lenth[i][j], n_hyper - k + 1)  # +1 is gloss of the target word
                    hypo_lenth[i][j] = max(hypo_lenth[i][j], k - n_hyper + 1)

                # if i == wi and j == sj and k == n_hyper:
                #     print '!!!gloss: {}  sense:{}'.format(str(j), senseid)
                #     id_to_word = {}
                #     for word in word_to_id:
                #         id_to_word[word_to_id[word]] = word
                #     g = [id_to_word[w] for w in words]
                #     print ' '.join(g)
    store_notin_vocab_words(words_notin_vocab)
    return [gloss_to_id, gloss_lenth, sense_mask, hyper_lenth, hypo_lenth], max_n_sense


def data_postprocessing_for_validation(val_data, filtered_word_to_sense=None):
    new_val_data = []
    for elem in val_data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']
        if target_word in filtered_word_to_sense and target_sense in filtered_word_to_sense[target_word]:
            new_val_data.append(elem)
    return new_val_data


def test_data_postprocessing(train_dataset, train_target_words, test_data, back_off_type='MFS'):
    key_path = None
    if back_off_type == 'MFS':
        key_path = _path.MFS_PATH.format(train_dataset)
    elif back_off_type == 'FS':
        key_path = _path.WNFS_PATH

    id_key_map = {}
    if key_path:
        for line in open(key_path):
            id = line.split()[0]  # defaut \s=[ \f\n\r\t\v]
            key = line.split()[1]
            id_key_map[id] = key
    print('test_data_postprocessing test data length is', len(test_data))
    print('train_target_words length is', len(train_target_words))
    back_off_result = []
    new_test_data = []
    test_data_index = []
    for index, d in enumerate(test_data):
        if d['target_word'] in train_target_words:
            new_test_data.append(d)
            test_data_index.append(index)
        else:
            id = d['id']
            if id in id_key_map:
                back_off_result.append([id, id_key_map[id]])
    print('test_data_postprocessing new test data length is', len(new_test_data))
    return new_test_data, test_data_index, back_off_result


def data_postprocessing(train_dataset, test_dataset, train_data, test_data, back_off_type='MFS'):
    train_target_words = set()
    for elem in train_data:
        target_word = elem['target_word']
        train_target_words.add(target_word)

    test_target_words = set()
    for elem in test_data:
        target_word = elem['target_word']
        test_target_words.add(target_word)

    target_words = train_target_words | test_target_words  # 求交集-》求并集

    new_train_data = []
    for elem in train_data:
        target_word = elem['target_word']
        if target_word in target_words:
            new_train_data.append(elem)

    mfs_key_path = _path.MFS_PATH.format(train_dataset)
    fs_key_path = _path.WNFS_PATH

    mfs_id_key_map = {}
    for line in open(mfs_key_path):
        id = line.split()[0]
        key = line.split()[1]  # defaut \s=[ \f\n\r\t\v]
        mfs_id_key_map[id] = key
    fs_id_key_map = {}
    for line in open(fs_key_path):
        id = line.split()[0]
        key = line.split()[1]  # defaut \s=[ \f\n\r\t\v]
        fs_id_key_map[id] = key

    back_off_result = []
    new_test_data = []
    mfs_using_fs_info = 0
    target_word_back_off = set()
    all_target_words = set()
    for elem in test_data:
        target_word = elem['target_word']
        all_target_words.add(target_word)
        if target_word in target_words:
            new_test_data.append(elem)
        else:
            target_word_back_off.add(target_word)
            if test_dataset != _path.ALL_WORDS_TEST_DATASET[0]:  # ALL dataset id format has dataset name
                id = test_dataset + '.' + elem['id']
            else:
                id = elem['id']
            if back_off_type == 'FS':
                back_off_result.append([elem['id'], fs_id_key_map[id]])
            if back_off_type == 'MFS':  # dataset MFS may not cover all-words
                if id in mfs_id_key_map:
                    back_off_result.append([elem['id'], mfs_id_key_map[id]])
                else:  ## 不在MFS用wn-first-sense补充
                    mfs_using_fs_info += 1
                    back_off_result.append([elem['id'], fs_id_key_map[id]])

    print('***MFS Using wordnet information instance number:%d ' % (mfs_using_fs_info))
    print('***Using back off target words: %s/%s' % (len(target_word_back_off), len(all_target_words)))

    back_off_result_path = _path.BACK_OFF_RESULT_PATH.format(back_off_type)
    print('***Writing to back_off_'
          'results to file:%s' % back_off_result_path)
    with open(back_off_result_path, 'w') as f:
        for instance_id, predicted_sense in back_off_result:
            f.write('%s %s\n' % (instance_id, predicted_sense))

    return new_train_data, new_test_data, target_words, back_off_result


if __name__ == '__main__':

    train_dataset = _path.ALL_WORDS_TRAIN_DATASET[0]
    print 'train_dataset: ' + train_dataset
    test_dataset = _path.ALL_WORDS_TEST_DATASET[0]
    print 'test_dataset: ' + test_dataset

    # dataset_i = 0
    # train_dataset = _path.LS_DATASET[dataset_i]
    # test_dataset = _path.LS_DATASET[dataset_i]
    # print 'test_dataset: ' + test_dataset

    # load dateset
    train_data = load_train_data(train_dataset)
    test_data = load_test_data(test_dataset)
    val_data = load_val_data(test_dataset)

    back_off_result = []
    if train_dataset in _path.ALL_WORDS_TRAIN_DATASET:
        val_data_lenth_pre = len(val_data)
        train_data, val_data, target_words, back_off_result = data_postprocessing(train_dataset, test_dataset,
                                                                                  train_data, val_data,
                                                                                  back_off_type='FS')
        print 'Dataset size (train/test): %d / %d' % (len(train_data), len(val_data))
        print '***Using back-off instance: %d' % (len(back_off_result))
        missed = val_data_lenth_pre - (len(val_data) + len(back_off_result))
        print '***Missing instance(not in MFS/FS): %d/%d = %.3f' % (
            (missed, val_data_lenth_pre, float(missed) / val_data_lenth_pre))

    # build vocab utils
    word_to_id = build_vocab(train_data)
    target_word_to_id, target_sense_to_id, n_words, n_senses_from_target_id = build_sense_ids(train_data)
    print 'Vocabulary size: %d' % len(word_to_id)
    with open(PUNGAN_ROOT_PATH + '/WSD/tmp/words_vocab.txt', 'w') as f:
        print('Writing in words_vocab.txt')
        for word, id in word_to_id.items():
            f.write('{}\t{}\n'.format(word.encode('utf-8'), id))
    print 'Target word: %d' % len(target_word_to_id)
    tot_n_senses = sum(n_senses_from_target_id.values())
    print 'Avg n senses per target word: ' + str(float(tot_n_senses) / len(n_senses_from_target_id))
    with open(PUNGAN_ROOT_PATH + '/WSD/tmp/target_word.txt', 'w') as f:
        for word, id in target_word_to_id.items():
            f.write('{}\t{}\n'.format(word, id))

    train_data = convert_to_numeric(train_data, word_to_id, target_word_to_id, target_sense_to_id,
                                    n_senses_from_target_id)
    val_data = convert_to_numeric(val_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id)

    id_to_word = {}
    for word in word_to_id:
        id_to_word[word_to_id[word]] = word

    with open(PUNGAN_ROOT_PATH + '/WSD/tmp/pos_dic.pkl', 'rb') as f:
        pos_to_id = pickle.load(f)

    id_to_pos = {}
    for pos in pos_to_id:
        id_to_pos[pos_to_id[pos]] = pos

    # for i in range(5):
    #     print id_to_word[train_data[i].target_word_id]

    ii = 3
    for i in range(len(train_data[ii].xf)):
        # print('%s/%s ' % (id_to_word[train_data[ii].xf[i]], id_to_pos[train_data[ii].pf[i]])),
        print('%s ' % (id_to_word[train_data[ii].xf[i]])),
    print(id_to_word[train_data[ii].target_word_id], id_to_pos[train_data[ii].target_pos_id])

    # load dictionary
    # try_key = 'art%1:04:00::'   # SE2, SE3: 'ask%2:32:05::'
    # try_word = 'art'  # ask#v
    try_key = 'divide%2:31:00::'  # ALL_WORDS
    try_word = 'divide#v'  #
    # 下面是为了测试我的gloss是否与真正的sense一一对应，用这个发现了一个bug： for j, id0 in enumerate(sense_to_ids):
    wi = -1
    for word, i in target_word_to_id.items():
        if word == try_word:
            wi = i
            print('###main: target_word: %s' % word)
            # print('wi: %s'%(wi))
            break
    sj = -1
    for sense, sense_id in target_sense_to_id[wi].items():
        if sense == try_key:
            sj = sense_id
            print('%s-th sense: %s' % (sj, sense))
    print('wi:%s\tsj:%s' % (wi, sj))

    train_target_words = target_word_to_id.keys()
    gloss_dict, synset_dict = load_dictionary(train_dataset, train_target_words, expand_type=3)
    print('gloss_dict[%s]: \n%s' % (try_key, gloss_dict[try_key]))

    print('end loading gloss_dict')
    dict_data, max_n_sense = bulid_dictionary_id(gloss_dict, synset_dict, target_sense_to_id, word_to_id,
                                                 word_to_id['<pad>'],
                                                 mask_size=100, max_gloss_words=100, max_synset_words=5)

    gloss = []
    for id in dict_data[0][wi][sj]:
        gloss.append(id_to_word[id])
    print "###main: gloss: %s" % ' '.join(gloss)

    for i, batch in enumerate(
            batch_generator(True, 300, val_data, dict_data, word_to_id['<pad>'], 10, 10, pad_last_batch=True)):
        if i % 100 == 0:
            print('@@@test_data: batch %s' % i)
            xfbs = batch[2]
            xfbs0 = batch[2][0, :].tolist()
            print("xfbs.shape:", xfbs.shape)
            for i in range(len(xfbs0)):
                print('%s ' % (id_to_word[xfbs0[i]])),
            print('\n')

    '''
    测试expand gloss 层次结构 部分的代码的问题
    '''
    # 下面是为了测试我的gloss是否与真正的sense一一对应
    for word, i in target_word_to_id.items():
        if word == try_word:
            wi = i
            print('###main: target_word: %s' % word)
            # print('wi: %s'%(wi))
            break
    sj = -1
    for sense, sense_id in target_sense_to_id[wi].items():
        if sense == try_key:
            sj = sense_id
            print('%s-th sense: %s' % (sj, sense))

    print('***Expand hyp gloss')
    gloss_dict, synset_dict = load_dictionary(train_dataset, train_target_words, expand_type=4, n_hyper=4, n_hypo=4)
    print('hyper of %s' % try_key)
    for k, g in enumerate(gloss_dict[try_key]):
        print(k, g)

    print('end loading gloss_dict')
    dict_data, max_n_sense = bulid_dictionary_id_hyp(gloss_dict, target_sense_to_id, word_to_id, word_to_id['<pad>'],
                                                     mask_size=100, max_gloss_words=100, n_hyper=4, n_hypo=4)

    for g in dict_data[0][wi][sj]:
        gloss = []
        for id in g:
            gloss.append(id_to_word[id])
        print "gloss: %s" % ' '.join(gloss)

    for i, batch in enumerate(batch_generator_hyp(True, 300, train_data, dict_data, word_to_id['<pad>'], 10, 10,
                                                  n_hyper=4, n_hypo=4, pad_last_batch=False)):

        xf, xb, xfb, pf, pb, pfb, target_ids, sense_ids, instance_ids, \
        glosses_ids, glosses_lenth, sense_mask, hyper_lenth, hypo_lenth = batch

        if i == 1:
            print('@@@test_data: batch %s' % i)
            xfbs = batch[2]
            xfbs0 = batch[2][0, :].tolist()
            print("xfbs.shape:", xfbs.shape)
            for i in range(len(xfbs0)):
                print('%s ' % (id_to_word[xfbs0[i]])),
            print('\n')

            print(xfb.shape)
            print(xfb.dtype)
            print(target_ids.shape)
            print(target_ids.dtype)
