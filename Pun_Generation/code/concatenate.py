import random
PUNGAN_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
source_sents = []
with open(PUNGAN_ROOT_PATH + '/Pun_Generation/code/backward_model_path/dealt_first_part_file') as f:
    for line in f:
        source_sents.append(line.strip())
forward = []
with open(PUNGAN_ROOT_PATH + '/Pun_Generation_Forward/code/forward_model_path/second_part_file') as f:
    for line in f:
        forward.append(line.strip())
forward_split = [forward[i:i + 32] for i in range(0, len(forward), 32)]
with open(PUNGAN_ROOT_PATH + '/Pun_Generation/code/backward_model_path/concatenate_file','w') as fw:
    remain = []
    if len(forward_split)%2 == 0 and len(forward_split[-1]) == 32:
        unit = len(forward_split)/2
    elif len(forward_split)%2 == 1:
        unit = len(forward_split)/2
        remain = forward_split[-1]
    elif len(forward_split)%2 == 0:
        unit = len(forward_split)/2 - 1
        remain = forward_split[-2] + forward_split[-1]
    for i in range(unit):
        for j in range(32):
            l1 = forward[i * 64 + j].split()
            l1 = source_sents[i * 64 + j * 2].split() + l1
            fw.write(' '.join(l1[:-1])+'\n')
            l2 = forward[i * 64 + 32 + j].split()
            l2 = source_sents[i * 64 + j * 2 + 1].split() + l2
            fw.write(' '.join(l2[:-1])+'\n')
    if remain:
        id = unit * 64
        for sent in remain[:len(remain)/2]:
            l1 = sent.split()
            l1 = source_sents[id].split() + l1
            fw.write(' '.join(l1[:-1])+'\n')
            l2 = sent.split()
            l2 = source_sents[id + 1].split() + l2
            fw.write(' '.join(l2[:-1])+'\n')
            id += 2
# randomly choose 10 candidate sentences for human evaluation
result = []
chosen = []
with open(PUNGAN_ROOT_PATH + '/Pun_Generation/code/backward_model_path/concatenate_file') as f:
    for line in f:
        result.append(line)
    result_split = [result[i:i + 2] for i in range(0, len(result), 2)]
    id_list = []
    for i in range(50):
        id = random.randint(0, len(result_split)-1)
        while id in id_list:
            id = random.randint(0, len(result_split)-1)
        id_list.append(id)
        # print('len(result_split)', len(result_split))
        # print('id', id)
        chosen.extend(result_split[id])
with open(PUNGAN_ROOT_PATH + '/inf_human.txt', 'w') as fw:
    for i in chosen:
        fw.write(i)
