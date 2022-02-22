import re
import numpy as np
import os


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


# AF:0 HB:1 SP:2
def get_feature(PATH):
    training_features = None
    training_lables = []
    for file in os.listdir(PATH):
        file_path = os.path.join(PATH, file)
        if "arm" in file or "others" in file or "AF" in file or "counting" in file:
            if training_features is None:
                training_features = np.load(file_path)
                labels = [0 for i in range(len(training_features))]
            else:
                features = np.load(file_path)
                labels = [0 for i in range(len(features))]
                training_features = np.concatenate((training_features, features), axis=0)
            training_lables.extend(labels)

        if "head" in file or "HB" in file or "HB" in file or "throwing" in file:
            if training_features is None:
                training_features = np.load(file_path)
                labels = [1 for i in range(len(training_features))]
            else:
                features = np.load(file_path)
                labels = [1 for i in range(len(features))]
                training_features = np.concatenate((training_features, features), axis=0)
            training_lables.extend(labels)

        if "spin" in file or "SP" in file or "wash" in file:
            if training_features is None:
                training_features = np.load(file_path)
                labels = [2 for i in range(len(training_features))]
            else:
                features = np.load(file_path)
                labels = [2 for i in range(len(features))]
                training_features = np.concatenate((training_features, features), axis=0)
            training_lables.extend(labels)
    return training_features, np.array(training_lables).squeeze()