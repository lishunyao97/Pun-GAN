import sys
PUNGAN_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
target_words = []
with open(PUNGAN_ROOT_PATH + '/Pun_Generation/data/sample_'+sys.argv[1]) as f:
    for line in f:
        target_words.append(line.strip())
backward = []
with open(PUNGAN_ROOT_PATH + '/Pun_Generation/code/backward_model_path/first_part_file') as f:
    for line in f:
        backward.append(line.strip())
backward_split = [backward[i:i + 32] for i in range(0, len(backward), 32)]

with open(PUNGAN_ROOT_PATH + '/Pun_Generation/code/backward_model_path/dealt_first_part_file','w') as fw:
    remain = []
    if len(backward_split)%2 == 0 and len(backward_split[-1]) == 32:
        unit = len(backward_split)/2
    elif len(backward_split)%2 == 1:
        unit = len(backward_split)/2
        remain = backward_split[-1]
    elif len(backward_split)%2 == 0:
        unit = len(backward_split)/2 - 1
        remain = backward_split[-2] + backward_split[-1]
    for i in range(unit):
        for j in range(32):
            l1 = backward[i * 64 + j].split()
            l1.reverse()
            l1.append(target_words[i * 64 + j * 2])
            fw.write(' '.join(l1[1:])+'\n')
            l2 = backward[i * 64 + 32 + j].split()
            l2.reverse()
            l2.append(target_words[i * 64 + j * 2 + 1])
            fw.write(' '.join(l2[1:])+'\n')
    if remain:
        id = unit * 64
        for sent in remain[:len(remain)/2]:
            l1 = sent.split()
            l1.reverse()
            l1.append(target_words[id])
            fw.write(' '.join(l1[1:])+'\n')
            l2 = sent.split()
            l2.reverse()
            l2.append(target_words[id + 1])
            fw.write(' '.join(l2[1:])+'\n')
            id += 2
