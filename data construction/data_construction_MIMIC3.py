import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader


class Reader(object):
    def __init__(self, dataset_dir, listfile=None):
        self._dataset_dir = dataset_dir
        self._current_index = 0
        if listfile is None:
            listfile_path = os.path.join(dataset_dir, "listfile.csv")
        else:
            listfile_path = listfile
        with open(listfile_path, "r") as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self._data = self._data[1:]

    def get_number_of_examples(self):
        return len(self._data)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)


class DeteriorationReader(Reader):
    def __init__(self, dataset_dir, listfile=None, period_length=48.0):
        Reader.__init__(self, dataset_dir, listfile)
        self._data = [line.split(',') for line in self._data]
        self._data = [(x, int(y)) for (x, y) in self._data]
        self._period_length = period_length

    def _read_timeseries(self, ts_filename):
        ret = []
        with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                if float(mas[0]) > self._period_length:
                    continue
                ret.append(np.array(mas))
        if ret == []:
            return (ret, header)
        final_ret = np.stack(ret)
        return (final_ret, header)

    def read_example(self, index):
        if index < 0 or index >= len(self._data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

        name = self._data[index][0]
        t = self._period_length
        y = self._data[index][1]
        (X, header) = self._read_timeseries(name)

        return {"X": X,
                "t": t,
                "y": y,
                "header": header,
                "name": name}


def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if v == []:
                break
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data


def load_data(reader, normalizer, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]

    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    whole_data = (np.array(data), labels)
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def str2float(data):
    new_data = []
    for i in range(data.shape[0]):
        new_data_tmp = []
        for j in range(data.shape[1]):
            if data[i][j] == '':
                new_data_tmp.append(-1.0)
            elif j == 3 or j == 4 or j == 6:
                num = data[i][j].split(' ', 1)[0]
                if num in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    num_data = float(num)
                elif data[i][j] == '1.0 ET/Trach' or data[i][j] == 'ET/Trach':
                    num_data = float(2.0)
                elif num == 'None':
                    num_data = -1.0
                elif num == 'Spontaneously':
                    num_data = 4.0
                elif num == 'To Speech':
                    num_data = 3.0
                elif num == 'To Pain':
                    num_data = 2.0
                elif num == 'No':
                    num_data = 1.0
                elif num == 'Obeys':
                    num_data = 6.0
                elif num == 'Localizes':
                    num_data = 5.0
                elif num == 'Oriented':
                    num_data = 5.0
                elif num == 'Inapprop':
                    num_data = 3.0
                elif num == 'Confused':
                    num_data = 4.0
                elif num == 'Abnorm':
                    if data[i][j] == 'Abnorm extensn':
                        num_data = 2.0
                    else:
                        num_data = 3.0
                elif num == 'Abnormal':
                    if data[i][j] == 'Abnormal extension':
                        num_data = 2.0
                    else:
                        num_data = 3.0
                elif num == 'Flex-withdraws':
                    num_data = 4.0
                elif num == 'To':
                    if data[i][j] == 'To Speech':
                        num_data = 3.0
                    else:
                        num_data = 2.0
                new_data_tmp.append(float(num_data))
            else:
                new_data_tmp.append(float(data[i][j]))

        new_data.append(np.array(new_data_tmp))

    return np.array(new_data)


class VisitSequenceWithLabelDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels, ts, names, num_features, reverse=False):
        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")

        self.labels = labels
        self.seqs = seqs
        self.ts = ts
        self.names = names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index], self.ts[index], self.names[index]


def visit_collate_fn(batch):
    batch_seq, batch_label, batch_ts, batch_name = zip(*batch)

    num_features = batch_seq[0].shape[1]
    seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
    max_length = 200

    sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))

    sorted_padded_seqs = []
    sorted_labels = []
    sorted_ts = []
    sorted_names = []
    sorted_lengths_new = []

    k = 0
    for i in sorted_indices:
        length = batch_seq[i].shape[0]

        if length < max_length:
            padded = np.concatenate(
                (batch_seq[i], np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
            padded_ts = np.concatenate(
                (batch_ts[i], np.zeros((max_length - length,), dtype=np.float32)), axis=0)
        elif length == max_length:
            padded = batch_seq[i]
            padded_ts = batch_ts[i]
        else:
            k = k + 1
            continue

        sorted_padded_seqs.append(padded)
        sorted_labels.append(batch_label[i])
        sorted_ts.append(padded_ts)
        sorted_names.append(batch_name[i])
        sorted_lengths_new.append(sorted_lengths[k])
        k = k + 1

    seq_tensor = np.stack(sorted_padded_seqs, axis=0)
    label_tensor = torch.FloatTensor(sorted_labels)
    ts_tensor = np.stack(sorted_ts, axis=0)

    return torch.from_numpy(seq_tensor), label_tensor, sorted_lengths_new, torch.from_numpy(ts_tensor), list(
        sorted_names)


def data_construction(data_path, demo_path):
    small_part = False
    time_length = 48.0

    train_reader = DeteriorationReader(dataset_dir=os.path.join(data_path, 'train'),
                                             listfile=os.path.join(data_path, 'train_listfile.csv'),
                                             period_length=time_length)

    val_reader = DeteriorationReader(dataset_dir=os.path.join(data_path, 'train'),
                                           listfile=os.path.join(data_path, 'val_listfile.csv'),
                                           period_length=time_length)

    test_reader = DeteriorationReader(dataset_dir=os.path.join(data_path, 'test'),
                                            listfile=os.path.join(data_path, 'test_listfile.csv'),
                                            period_length=time_length)

    train_raw = load_data(train_reader, None, small_part, return_names=True)
    val_raw = load_data(val_reader, None, small_part, return_names=True)
    test_raw = load_data(test_reader, None, small_part, return_names=True)

    train_raw_new = np.array([str2float(train_raw['data'][0][i][:,1:]) for i in range(len(train_raw['data'][0]))])
    val_raw_new = np.array([str2float(val_raw['data'][0][i][:,1:]) for i in range(len(val_raw['data'][0]))])
    test_raw_new = np.array([str2float(test_raw['data'][0][i][:,1:]) for i in range(len(test_raw['data'][0]))])
    train_raw_ts = np.array([train_raw['data'][0][i][:,0].astype(float) for i in range(len(train_raw['data'][0]))])
    val_raw_ts = np.array([val_raw['data'][0][i][:,0].astype(float) for i in range(len(val_raw['data'][0]))])
    test_raw_ts = np.array([test_raw['data'][0][i][:,0].astype(float) for i in range(len(test_raw['data'][0]))])

    train_set = VisitSequenceWithLabelDataset(train_raw_new, train_raw['data'][1], train_raw_ts, train_raw['names'], 17)
    valid_set = VisitSequenceWithLabelDataset(val_raw_new, val_raw['data'][1], val_raw_ts, val_raw['names'], 17)
    test_set = VisitSequenceWithLabelDataset(test_raw_new, test_raw['data'][1], test_raw_ts, test_raw['names'], 17)

    train_loader = DataLoader(dataset=train_set, batch_size=256, shuffle=False, collate_fn=visit_collate_fn,num_workers=0)
    valid_loader = DataLoader(dataset=valid_set, batch_size=256, shuffle=False,collate_fn=visit_collate_fn, num_workers=0)
    test_loader = DataLoader(dataset=test_set, batch_size=256, shuffle=False,collate_fn=visit_collate_fn, num_workers=0)

    demographic_data = []
    los_data = []
    idx_list = []

    for id_name in os.listdir(demo_path):
        for demo_file in os.listdir(os.path.join(demo_path, id_name)):
            if demo_file[0:7] != 'episode':
                continue
            cur_file = demo_path + id_name + '/' + demo_file
            with open(cur_file, "r") as tsfile:
                header = tsfile.readline().strip().split(',')
                if header[0] != "Icustay":
                    continue
                cur_data = tsfile.readline().strip().split(',')
                if len(cur_data) == 1:
                    cur_demo = (np.zeros(3) - 1).tolist()
                else:
                    if cur_data[6] == '':
                        continue
                    if cur_data[1] == '':
                        cur_data[1] = -1
                    if cur_data[2] == '':
                        cur_data[2] = -1
                    if cur_data[3] == '':
                        cur_data[3] = -1
                    cur_demo = [int(cur_data[1]), int(cur_data[2]), float(cur_data[3])]
                    cur_los = float(cur_data[6])

                demographic_data.append(cur_demo)
                los_data.append(cur_los)
                idx_list.append(id_name + '_' + demo_file[0:8])

    return train_loader, valid_loader, test_loader, demographic_data, los_data, idx_list
