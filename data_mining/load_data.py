def load_data(file):
    data = []
    with open(file, 'r') as fp:
        fp.readline()
        for line in fp.readlines():
            line_list = line.strip().split('\t')
            line_list[1:5] = [x.split(',') for x in line_list[1:5]]
            if len(line_list) == 8:
                line_list[5:8] = [int(x) for x in line_list[5:8]]
            data.append(line_list)
    return data