import pickle
import os

def load_data(data_file):
    data = []
    with open(data_file, 'r') as fp:
        fp.readline()
        for line in fp.readlines():
            line_list = line.strip().split('\t')
            line_list[1:5] = [x.split(',') for x in line_list[1:5]]
            if len(line_list) == 8:
                line_list[5:8] = [int(x) for x in line_list[5:8]]
            data.append(line_list)
    return data


def load_leveled_data(data_file, save=True, out_file=None, update=False):
    """
    Get data with catogary of every level

    Args:
        data_file (str): the .txt file which is storing data
        save (bool, optional): save or not (default: True)
        out_file (str, optional): path to save and load the .pkl file if save is True (default: data_file[:-4]+'.pkl')
        update (bool, optional): if True, the returned data will be calculated again (default: False)

    Return:
        leveled_data (dict): data with catogary of every level
            {
                level1_1:
                {
                    level2_1:
                    {
                        level3_1: [item_id, title_characters, title_words, description_characters, description_words]
                        ...
                    }
                    ...
                }
                ...
            }
        item_ids (list): list of item ids
    """
    if out_file == None:
        out_file = data_file[:-4]+'.pkl'
    if os.path.exists(out_file) and not update:
        return pickle.load(open(out_file, 'rb'))
    else:
        data = load_data(data_file)
        leveled_data = {}
        item_ids = []
        for line in data:
            level1, level2, level3 = line[5:8]
            if level1 not in leveled_data:
                leveled_data[level1] = {}
            if level2 not in leveled_data[level1]:
                leveled_data[level1][level2] = {}
            if level3 not in leveled_data[level1][level2]:
                leveled_data[level1][level2][level3] = []
            leveled_data[level1][level2][level3].append(line)
            item_ids.append(line[0])
        if save:
            with open(out_file, 'wb') as fp:
                pickle.dump([leveled_data, item_ids], fp)
        return leveled_data, item_ids


def get_class_data(leveled_data, classes):
    """
    Get data of classes

    Args:
        leveled_data (): returned value of load_leveled_data
        classes (list or tuple): list storing the classes you want

    Return:
        [
            [item_id, title_chars, title_words, descrip_chars, descrip_words, level1, level2, level3],
            ...
        ]

    Example:
        get_class_data(leveled_data, [1, 2])  # get the data of class 1 in level1 and class 2 level2
        get_class_data(leveled_data, [])  # get the data of all classes
    """
    if len(classes) == 0:
        if isinstance(leveled_data, dict):
            result = []
            for k in leveled_data:
                result.extend(get_class_data(leveled_data[k], []))
            return result
        else:
            return leveled_data
    else:
        return get_class_data(leveled_data[classes[0]], classes[1:])


if __name__ == '__main__':
    leveled_data = load_leveled_data('../data/train_a.txt')
    class_data = get_class_data(leveled_data, [3])
    print(len(class_data), class_data[0])