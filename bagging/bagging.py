TEST_CNT = 49076
cate_voter = [['', {}, {}, {}] for j in range(TEST_CNT)]
with open('gru_word_out.txt', 'r') as fp:
    fp.readline()
    for i, line in enumerate(fp.readlines()):
        line = line.strip().split('\t')
        cate_voter[i][0] = line[0]

def update_voter(text_path):
    with open(text_path, 'r') as fp:
        fp.readline()
        for i, line in enumerate(fp.readlines()):
            line = line.strip().split('\t')
            cate1, cate2, cate3 = line[1], line[2], line[3]
            if cate1 not in cate_voter[i][1]:
                cate_voter[i][1][cate1] = 0
            cate_voter[i][1][cate1] += 1
            if cate2 not in cate_voter[i][2]:
                cate_voter[i][2][cate2] = 0
            cate_voter[i][2][cate2] += 1
            if cate3 not in cate_voter[i][3]:
                cate_voter[i][3][cate3] = 0
            cate_voter[i][3][cate3] += 1

def update_voter_(text_path):
    with open(text_path, 'r') as fp:
        fp.readline()
        for i, line in enumerate(fp.readlines()):
            line = line.strip().split('\t')
            cate2, cate3 = line[2], line[3]
            if cate2 not in cate_voter[i][2]:
                cate_voter[i][2][cate2] = 0
            cate_voter[i][2][cate2] += 1
            if cate3 not in cate_voter[i][3]:
                cate_voter[i][3][cate3] = 0
            cate_voter[i][3][cate3] += 1

def get_result():
    for i in range(TEST_CNT):
        cate_voter[i][1] = max(cate_voter[i][1], key=cate_voter[i][1].get)
        cate_voter[i][2] = max(cate_voter[i][2], key=cate_voter[i][2].get)
        cate_voter[i][3] = max(cate_voter[i][3], key=cate_voter[i][3].get)


def save_result(save_path):
    with open(save_path, 'w') as fp:
        fp.write('id\tcate1\tcate2\tcate3\n')
        for info in cate_voter:
            fp.write('{}\t{}\t{}\t{}\n'.format(info[0], info[1], info[2], info[3]))

#update_voter('cnn_word_out.txt')
update_voter('cnn345_wordchar_out.txt')
update_voter('bagging_ordered.txt')
update_voter('cnn_wordchar_out.txt')
update_voter('submit_ordered.txt')
update_voter('submit_ordered_weight.txt')
update_voter('gru_word_out.txt')
update_voter('lstm_word_out.txt')
update_voter('swem_wordchar_out.txt')

get_result()

save_result('final_result.txt')
