import csv
import pickle

label_file = '/Users/mike/Downloads/object-detection-crowdai/labels_crowdai.csv'
train_file = '/Users/mike/Downloads/object-detection-crowdai/labels_train.csv'
val_file = '/Users/mike/Downloads/object-detection-crowdai/labels_val.csv'
pickle_file = '/Users/mike/PycharmProjects/object_detection/dataset/name2id.pickle'
LABEL = 5
LIMIT = 64585


def save_pickle():
    name_set = set()
    name2id = {}

    with open(label_file) as fh:
        csv_reader = csv.reader(fh)
        next(csv_reader)  # Skip head
        for row in csv_reader:
            name_set.add(row[LABEL])

    name_list = list(name_set)
    idx = 1
    for i in name_list:
        name2id[i] = idx
        idx += 1

    with open(pickle_file, 'wb') as fh:
        pickle.dump(name2id, fh)


def convert_name2id():
    with open(pickle_file, 'rb') as fh:
        name2id = pickle.load(fh)

    i = 1
    with open(label_file) as label_fh, open(train_file, 'w') as train_fh, open(val_file, 'w') as val_fh:
        csv_reader = csv.reader(label_fh)
        next(csv_reader)
        for row in label_fh:
            row = row.split(',')
            row[LABEL] = str(name2id[row[LABEL]])
            if i < LIMIT:
                train_fh.write(','.join(row))
            else:
                val_fh.write(','.join(row))
            i += 1


if __name__ == '__main__':
    # save_pickle()
    convert_name2id()
