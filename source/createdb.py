from __future__ import print_function

import os
import pickle
import time
from datetime import datetime

import numpy as np
from medpy.filter import IntensityRangeStandardization
from medpy.io import load

SHAPE = (4, 240, 240, 155)
KERNEL = (33, 33)
CLASSCNT = 5
MODCNT = 4
ROTATE = 4

ORIG_READ_PATH = "./data/"
WRITE_PATH = "./output/"

hl = {'h': 0, 'l': 1, 't': 2}
MODS = {"T1": 0, "T2": 1, "T1c": 2, "Flair": 3, "OT": 4}


def get_img_name_list(path):
    name_list = os.listdir(path)
    result_list_l = {}
    result_list_h = {}
    result_list_t = {}
    for name in name_list:
        temp = name.replace(".nii", "").split(".")
        dataset = temp[0]
        cnt = int(temp[1]) - 1
        mod = MODS[temp[-2].split("_")[-1]]
        if dataset == "l":
            result_list_l[cnt, mod] = name
        elif dataset == "h":
            result_list_h[cnt, mod] = name
        elif dataset == "t":
            result_list_t[cnt, mod] = name
    return np.array([result_list_h, result_list_l, result_list_t])


def get_img(path, name, isgt=False):
    result = load(path + name)[0]
    if not isgt:
        minpxl = np.min(result)
        if minpxl < 0:
            result[result != 0] -= minpxl
    return result


def train_irs(data_list, dataset):
    logthis("Train LGG IRS Started")
    irs = IntensityRangeStandardization()
    imgcnt = len(data_list[hl[dataset]]) // 5
    for i in range(imgcnt):
        for mod in range(MODCNT):
            curimg = get_img(ORIG_READ_PATH, data_list[hl[dataset]][i, mod])
            irs = irs.train([curimg[curimg > 0]])
        print("\rIRS Train", i + 1, "/", imgcnt, end="")
    with open(os.path.join(WRITE_PATH, "intensitymodel.txt"), 'wb') as f:
        pickle.dump(irs, f)


def getlabel(data_list, datasetclass):
    logthis(datasetclass + " Getting label started.")
    total = np.sum([len(data_list[hl[dataset]]) for dataset in datasetclass]) // 5
    label = np.memmap(filename=os.path.join(WRITE_PATH, datasetclass + "_gt.dat"), mode="w+",
                      shape=(total, SHAPE[1], SHAPE[2], SHAPE[3]), dtype=np.int8)
    cnt = 0
    for dataset in datasetclass:
        set_size = len(data_list[hl[dataset]]) // 5
        for i in range(set_size):
            label[cnt] = get_img(ORIG_READ_PATH, data_list[hl[dataset]][i, MODS["OT"]], True)
            cnt += 1
            print("\rLabel", datasetclass, i + 1, "/", set_size, end="")
    return label


def read_from_file(data_list, datasetclass):
    logthis(datasetclass + " Reading training dataset & IRS trainig started")

    fp = np.memmap(os.path.join(WRITE_PATH, datasetclass + "_orig.dat"), dtype=np.float32, mode="w+", shape=(
        np.sum([len(data_list[hl[dataset]]) for dataset in datasetclass]) // 5, SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[3]))
    labels = np.memmap(filename=os.path.join(WRITE_PATH, datasetclass + "_gt.dat"), mode="r",
                       shape=(fp.shape[0], SHAPE[1], SHAPE[2], SHAPE[3]), dtype=np.int8)
    cnt = 0
    with open(os.path.join(WRITE_PATH, "intensitymodel.txt"), 'r') as f:
        trained_model = pickle.load(f)
    labelid = {}
    cut_size = [409, 525, 409, 416, 424]
    for dataset in datasetclass:
        set_size = len(data_list[hl[dataset]]) // 5
        for i in range(set_size):
            for mod in range(MODCNT):
                curimg = get_img(ORIG_READ_PATH, data_list[hl[dataset]][i, mod])

                if mod == 0:
                    dataidx = np.argwhere(curimg == 0).astype(np.uint8)
                    for clscnt in range(CLASSCNT):
                        labelidx = np.argwhere(labels[cnt] == clscnt).astype(np.uint8)
                        if labelidx.shape[0] == 0:
                            labelid[cnt, clscnt] = np.empty((0, 3))
                            continue
                        cumdims = (np.maximum(labelidx.max(), dataidx.max()) + 1) ** np.arange(dataidx.shape[1])
                        labelid[i, clscnt] = np.random.permutation(
                            labelidx[~np.in1d(labelidx.dot(cumdims), dataidx.dot(cumdims))])[:cut_size[clscnt]].astype(
                            np.uint8)
                curimg[curimg > 0] = trained_model.transform(curimg[curimg > 0], surpress_mapping_check=True)
                fp[cnt, mod] = curimg
            cnt += 1
            print("\r" + dataset, "Image Get", i + 1, "/", set_size, end="")

    with open(os.path.join(WRITE_PATH, "labelidx.txt"), 'wb') as f:
        pickle.dump(labelid, f)

    allstd = np.zeros(SHAPE[0], dtype=np.float32)
    for i in range(SHAPE[0]):
        allstd[i] = np.std(fp[:, i, :, :, :])
    allmean = np.mean(fp, axis=(0, 2, 3, 4))
    np.save(os.path.join(WRITE_PATH, datasetclass + "_zmuv.npy"), np.array([allmean, allstd]))
    print("")
    return fp


def temp_get_label(data_list, datasetclass):
    labelid = {}
    cut_size = [1600, 525, 409, 416, 424]

    fp = np.memmap(os.path.join(WRITE_PATH, datasetclass + "_orig.dat"), dtype=np.float32, mode="r", shape=(
        np.sum([len(data_list[hl[dataset]]) for dataset in datasetclass]) // 5, SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[3]))
    labels = np.memmap(filename=os.path.join(WRITE_PATH, datasetclass + "_gt.dat"), mode="r",
                       shape=(fp.shape[0], SHAPE[1], SHAPE[2], SHAPE[3]), dtype=np.int8)
    cnt = 0
    for curimg, curlabel in zip(fp[:, 0, :, :, :], labels):
        dataidx = np.argwhere(curimg == 0).astype(np.uint8)
        for clscnt in range(CLASSCNT):
            labelidx = np.argwhere(curlabel == clscnt).astype(np.uint8)
            if labelidx.shape[0] == 0:
                labelid[cnt, clscnt] = np.empty((0, 3))
                continue
            cumdims = (np.maximum(labelidx.max(), dataidx.max()) + 1) ** np.arange(dataidx.shape[1])
            labelid[cnt, clscnt] = np.random.permutation(
                labelidx[~np.in1d(labelidx.dot(cumdims), dataidx.dot(cumdims))])[:cut_size[clscnt]].astype(
                np.uint8)
        cnt += 1

    with open(os.path.join(WRITE_PATH, "labelidx.txt"), 'wb') as f:
        pickle.dump(labelid, f)


def finalize_input(datas, labels, data_list, datasetclass):
    logthis(datasetclass + " Finalizing input started.")
    total = np.sum([len(data_list[hl[dataset]]) for dataset in datasetclass]) // 5
    if datas is None:
        datas = np.memmap(os.path.join(WRITE_PATH, datasetclass + "_orig.dat"), dtype=np.float32,
                          shape=(total, SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[3]), mode="r")
    if labels is None:
        labels = np.memmap(filename=os.path.join(WRITE_PATH, datasetclass + "_gt.dat"),
                           shape=(total, SHAPE[1], SHAPE[2], SHAPE[3]), dtype=np.int8, mode="r")
    print(datasetclass + " started")

    with open(os.path.join(WRITE_PATH, "labelidx.txt"), 'r') as f:
        labelid = pickle.load(f)
    sample_size = np.sum([value.shape[0] for _, value in labelid.items()])
    fdata = np.memmap(os.path.join(WRITE_PATH, datasetclass + "_train.dat"), dtype=np.float32,
                      shape=(sample_size * ROTATE, KERNEL[0], KERNEL[1], SHAPE[0]), mode="w+")
    flabel = np.memmap(os.path.join(WRITE_PATH, datasetclass + "_train.lbl"), dtype=np.int8,
                       shape=(fdata.shape[0]), mode="w+")

    cnt = 0
    cnt2 = 0
    for data, label in zip(datas, labels):
        zzz = time.time()
        for clscnt in range(CLASSCNT):
            for curid in range(labelid[cnt2, clscnt].shape[0]):
                randid = labelid[cnt2, clscnt][curid]
                w_h = [randid[0] - KERNEL[0] / 2, randid[0] + KERNEL[0] / 2 + 1, randid[1] - KERNEL[1] / 2,
                       randid[1] + KERNEL[1] / 2 + 1]
                tempdata = data[:, w_h[0]:w_h[1], w_h[2]:w_h[3], randid[2]]
                tempdata = np.rollaxis(np.nan_to_num(tempdata), 0, 3)

                for rot in range(ROTATE):
                    fdata[cnt] = np.rot90(tempdata, rot)
                    flabel[cnt] = clscnt
                    cnt += 1
        cnt2 += 1
        print("\rWrite data", cnt2, "/", total, time.time() - zzz, end="")
    rng_state = np.random.get_state()
    np.random.shuffle(fdata)
    print("\nData shuffled")
    np.random.set_state(rng_state)
    np.random.shuffle(flabel)
    print("Label shuffled")
    return fdata


def get_test_data(data_list):
    logthis("Test dataset started.")
    fp = np.memmap(os.path.join(WRITE_PATH, "test_orig.dat"), dtype=np.float32, mode="w+",
                   shape=(len(data_list[hl["t"]]) // 4, SHAPE[1], SHAPE[2], SHAPE[3], SHAPE[0]))
    set_size = fp.shape[0]
    with open(os.path.join(WRITE_PATH, "intensitymodel.txt"), 'r') as f:
        trained_model = pickle.load(f)
    for i in range(set_size):
        for mod in range(MODCNT):
            curimg = get_img(ORIG_READ_PATH, data_list[hl["t"]][i, mod])
            curimg[curimg > 0] = trained_model.transform(curimg[curimg > 0], surpress_mapping_check=True)
            fp[i, :, :, :, mod] = curimg
        print("\rTest data", i + 1, "/", set_size, end="")
    return fp


def gauss_norm(fdata, datasetclass):
    logthis(datasetclass + " Gauss Normalization Started.")
    if fdata is None:
        fdata = np.memmap(os.path.join(WRITE_PATH, datasetclass + "_train.dat"), dtype=np.float32,
                          mode="r+").reshape((-1, KERNEL[0], KERNEL[1], SHAPE[0]))

    allmean, allstd = np.load(os.path.join(WRITE_PATH, datasetclass + "_zmuv.npy"))
    for i in range(4):
        fdata[..., i] -= allmean[i]
        print(str(i) + " Train mean zero")
        fdata[..., i] /= allstd[i]
        print(str(i) + "Train unit variance")


def logthis(a):
    print("\n" + str(datetime.now()) + ":", a)


def start_preproces():
    data_name_list = get_img_name_list(ORIG_READ_PATH)
    train_irs(data_name_list, "h")
    h_label = getlabel(data_name_list, "h")
    get_test_data(data_name_list)
    h_data = read_from_file(data_name_list, "h")
    h_patch = finalize_input(h_data, h_label, data_name_list, "h")
    gauss_norm(h_patch, "h")


if __name__ == '__main__':
    start_preproces()
