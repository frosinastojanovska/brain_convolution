import os
import pickle
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from train_adhd import ADHD, train_brain_convolution_model, test_brain_convolution_model
from model.utils import read_adhd_data, find_statistics


def evaluation_results(y_true, y_pred, decoded=False):
    """Evaluate metric for the prediction

    :param y_true: true classes
    :type y_true: numpy.array
    :param y_pred: predicted classes
    :type y_pred: numpy.array
    :param decoded: True if the predicted classes are decoded,
                    False if they are one-hot encoded
    :type decoded: bool
    :return: accuracy, precision micro, precision macro,
             recall micro, recall macro, f1 micro, f1 macro
    :type: list(float, float, float, float)
    """
    if not decoded:
        y_true = np.argmax(y_true, axis=0)
        y_pred = np.argmax(y_pred, axis=0)

    accuracy = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    return [accuracy, precision_micro, precision_macro,
            recall_micro, recall_macro, f1_micro, f1_macro]


def create_folds(file_path):
    """Creates 10 folds for cross validation

    :param file_path: file path to save the folds
    :type file_path: str
    :return: None
    """
    data = read_adhd_data()
    data = data[data['class'] != 'ADHD-Hyperactive/Impulsive']
    stats = find_statistics('data/ADHD200_CC200/', 'data/adhd_stats.pkl')
    subject_ids = data.network_name.values.tolist()
    subject_graph_files = os.listdir('data/ADHD200_CC200/')
    subject_ids = [subject for subject in subject_ids if
                   subject + '_connectivity_matrix_file.txt' in subject_graph_files]
    dataset = ADHD('ADHD200_CC200', subject_ids, data, stats)

    X = subject_ids
    y = []
    for subject_id in subject_ids:
        encoded_class = dataset.get_brain_class(subject_id)
        class_id = np.argmax(encoded_class)
        y.append(class_id)
    X = np.array(X)
    y = np.array(y)
    skf = StratifiedKFold(n_splits=10)
    folds = [([subject_ids[i] for i in train.tolist()], [subject_ids[i] for i in test.tolist()])
             for train, test in skf.split(X, y)]
    with open(file_path, 'wb') as f:
        pickle.dump(folds, f, pickle.HIGHEST_PROTOCOL)


def get_data(subject_ids):
    """Create dataset for training and testing the classifiers

    :param subject_ids: list of subject ids from the dataset
    :type subject_ids: list(str)
    :return: features (flatten matrix), classes
    :rtype: numpy.array, numpy.array
    """
    data = read_adhd_data()
    data = data[data['class'] != 'ADHD-Hyperactive/Impulsive']
    stats = find_statistics('data/ADHD200_CC200/', 'data/adhd_stats.pkl')
    dataset = ADHD('ADHD200_CC200', subject_ids, data, stats)
    x = []
    y = []
    for subject_id in subject_ids:
        matrix, node_features, encoded_class = dataset.get_brain_graph_and_class(subject_id)
        adj = matrix.ravel()
        decoded_class = np.argmax(encoded_class)
        x.append(adj)
        y.append(decoded_class)

    return np.array(x), np.array(y)


def evaluate_ordinary_classification(folds, method):
    assert method in ['bayes', 'random_forest', 'linear_svm', 'rbf_svm', 'mlp']

    if method == 'bayes':
        classifier = MultinomialNB()
    elif method == 'random_forest':
        classifier = RandomForestClassifier(n_estimators=100, criterion='entropy',
                                            bootstrap=True, n_jobs=-1)
    elif method == 'linear_svm':
        classifier = LinearSVC(C=0.9)
    elif method == 'rbf_svm':
        classifier = SVC(C=0.8, kernel='rbf', gamma=0.005)
    else:
        classifier = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                                   alpha=0.0001, batch_size=300, learning_rate_init=0.001, power_t=0.5,
                                   max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=True,
                                   validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    metrics = []
    for train, test in folds:
        x, y = get_data(train)
        classifier.fit(x, y)
        x, y = get_data(test)
        predicted = classifier.predict(x)
        metrics.append(evaluation_results(y, predicted, decoded=True))

    metrics = np.array(metrics)
    print("Accuracy: %0.2f (+/- %0.2f)" % (metrics[:, 0].mean(), metrics[:, 0].std() * 2))
    print("Precision micro: %0.2f (+/- %0.2f)" % (metrics[:, 1].mean(), metrics[:, 1].std() * 2))
    print("Precision macro: %0.2f (+/- %0.2f)" % (metrics[:, 2].mean(), metrics[:, 2].std() * 2))
    print("Recall micro: %0.2f (+/- %0.2f)" % (metrics[:, 3].mean(), metrics[:, 3].std() * 2))
    print("Recall macro: %0.2f (+/- %0.2f)" % (metrics[:, 4].mean(), metrics[:, 4].std() * 2))
    print("F1 micro: %0.2f (+/- %0.2f)" % (metrics[:, 5].mean(), metrics[:, 5].std() * 2))
    print("F1 macro: %0.2f (+/- %0.2f)" % (metrics[:, 6].mean(), metrics[:, 6].std() * 2))


def evaluate_brain_convolution_model(folds):
    metrics = []
    for train, test in folds:
        trained_model = train_brain_convolution_model(train)
        x, y = get_data(test)
        predicted = test_brain_convolution_model(test, trained_model)
        metrics.append(evaluation_results(y, predicted, decoded=False))

    metrics = np.array(metrics)
    print("Accuracy: %0.2f (+/- %0.2f)" % (metrics[:, 0].mean(), metrics[:, 0].std() * 2))
    print("Precision micro: %0.2f (+/- %0.2f)" % (metrics[:, 1].mean(), metrics[:, 1].std() * 2))
    print("Precision macro: %0.2f (+/- %0.2f)" % (metrics[:, 2].mean(), metrics[:, 2].std() * 2))
    print("Recall micro: %0.2f (+/- %0.2f)" % (metrics[:, 3].mean(), metrics[:, 3].std() * 2))
    print("Recall macro: %0.2f (+/- %0.2f)" % (metrics[:, 4].mean(), metrics[:, 4].std() * 2))
    print("F1 micro: %0.2f (+/- %0.2f)" % (metrics[:, 5].mean(), metrics[:, 5].std() * 2))
    print("F1 macro: %0.2f (+/- %0.2f)" % (metrics[:, 6].mean(), metrics[:, 6].std() * 2))


if __name__ == '__main__':
    file_folds = 'data/adhd_folds.pkl'
    if not os.path.exists(file_folds):
        create_folds(file_folds)

    with open(file_folds, 'rb') as f:
        folds = pickle.load(f)

    # evaluate_ordinary_classification(folds, 'bayes')
    evaluate_ordinary_classification(folds, 'random_forest')
    # evaluate_ordinary_classification(folds, 'linear_svm')
    # evaluate_ordinary_classification(folds, 'rbf_svm')
    # evaluate_ordinary_classification(folds, 'mlp')
