import os
import cv2
import numpy as np
from dataparser import split
import shutil

def get_derivatives(img):
    der = [img]
    der.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    der.append(cv2.rotate(der[-1], cv2.ROTATE_90_CLOCKWISE))
    der.append(cv2.rotate(der[-1], cv2.ROTATE_90_CLOCKWISE))
    der.append(cv2.flip(img, 0))
    der.append(cv2.flip(img, 1))
    return der


def generate(saved=False):
    if saved:
        imgs = np.load("imgs.npy")
        labels = np.load("labels.npy")
        label_names = CLASS_LABELS
        return imgs, labels, label_names
    base = "nmlo-contest-4/train/train/"
    labels = []
    label_names = []
    imgs = []
    for labelnum, folder in enumerate(os.listdir(base)):
        print(folder)
        path = os.path.join(base, folder)
        if not os.path.isdir(path):
            continue
        label_names.append(folder)
        for filename in os.listdir(path):
            filename = os.path.join(os.path.join(base, folder), filename)
            img = cv2.imread(filename)
            derivatives = get_derivatives(img)
            for img in derivatives:
                imgs.append(img)
                labels.append(labelnum)
    imgs = np.array(imgs)
    labels = np.array(labels)
    # labels = one_hot(labels)
    print(label_names)
    print(labels.shape)
    np.save("imgs", imgs)
    np.save("labels", labels)
    return imgs, labels, label_names


def split_train_val(val_split=0.1):
    base = "nmlo-contest-4/train/train/"
    newbase = "nmlo-contest-4/split/"
    if os.path.exists(newbase):
        shutil.rmtree(newbase)
    os.mkdir(newbase)
    train_path = os.path.join(newbase, "train")
    val_path = os.path.join(newbase, "val")
    os.mkdir(train_path)
    os.mkdir(val_path)
    for labelnum, folder in enumerate(os.listdir(base)):
        print(folder)
        path = os.path.join(base, folder)
        if not os.path.isdir(path):
            continue
        train_folder = os.path.join(train_path, folder)
        val_folder = os.path.join(val_path, folder)
        os.mkdir(train_folder)
        os.mkdir(val_folder)
        num_imgs = len(os.listdir(path))
        threshold = int(num_imgs * val_split)
        for i, filename in enumerate(os.listdir(path)):
            fold = train_folder
            if i < threshold:
                fold = val_folder
            img = cv2.imread(os.path.join(path, filename))
            # print(os.path.join(fold, filename))
            cv2.imwrite(os.path.join(fold, filename), img)


def resize_all():
    base = "nmlo-contest-4/train/train"
    base2 = "nmlo-contest-4/resized"
    os.mkdir(base2)
    for folder in os.listdir(base):
        currpath = os.path.join(base, folder)
        newpath = os.path.join(base2, folder)
        os.mkdir(newpath)
        for filepath in os.listdir(currpath):
            img = cv2.imread(os.path.join(currpath, filepath))
            img = cv2.resize(img, (224, 224))
            cv2.imwrite(os.path.join(newpath, filepath), img)


resize_all()
