from collections import defaultdict
import sys
import numpy as np

ATHEISM = 0
BOOKS = 1

if __name__ == "__main__":
    # Parse the input parameter to get the data folder, using current working directory as default
    args = sys.argv[1:]
    data_folder = args[0] if len(args) > 0 else '.'
    if data_folder.endswith("/") or data_folder.endswith("\\"):
        data_folder = data_folder[:-1]

    # Load data
    train_data = np.loadtxt(f"{data_folder}/trainData.txt", delimiter=" ")
    train_label = np.loadtxt(f"{data_folder}/trainLabel.txt", delimiter=" ")
    train_label = np.array([int(i - 1) for i in train_label])
    test_data = np.loadtxt(f"{data_folder}/testData.txt", delimiter=" ")
    test_label = np.loadtxt(f"{data_folder}/testLabel.txt", delimiter=" ")
    test_label = np.array([int(i - 1) for i in test_label])

    words = []              # The index of words
    words_mapping = []      # The mapping of words for printing
    with open(f"{data_folder}/words.txt") as f:
        words_mapping = [line.strip() for line in f]
    with open(f"{data_folder}/words.txt") as f:
        words = range(1, 1 + len(f.readlines()))
    
    # Parse the train data into a dict, key: doc_id, value: set of word_id for better performance
    train_dict = defaultdict(set)
    for (doc_id, word_id) in train_data:
        train_dict[int(doc_id)].add(int(word_id))
    test_dict = defaultdict(set)
    for (doc_id, word_id) in test_data:
        test_dict[int(doc_id)].add((int(word_id)))

    docs = list(train_dict.keys())
    docs.sort()
    docs = np.array(docs)
    train_label = train_label[docs - 1]
    parsed_train_data = np.array([[ 1 if word_id in train_dict[doc_id] else 0 for word_id in words] for doc_id in docs])

    atheism_data = parsed_train_data[train_label == ATHEISM]
    books_data = parsed_train_data[train_label == BOOKS]


    theta_c = sum(train_label) / len(train_label)
    theta_1s = (np.sum(atheism_data, axis=0) + 1) / ( len(atheism_data) + 2)
    theta_2s = (np.sum(books_data, axis=0) + 1) / ( len(books_data) + 2)
    
    discrimination = np.abs(np.log(theta_1s) -  np.log(theta_2s))
    top_discri_ids = (-discrimination).argsort()[:10]
    top_discri_words = [words_mapping[i] for i in top_discri_ids]
    print("The 10 most discriminative word features are:")
    print(top_discri_words)

    def predict(docs_data):
        y_pred = [0] * len(docs_data)
        for i, x in enumerate(docs_data):
            atheism_prob = np.sum(x * np.log(theta_1s) + (1-x) * np.log(1-theta_1s)) + np.log(theta_c)
            books_prob = np.sum(x * np.log(theta_2s) + (1-x) * np.log(1-theta_2s)) + np.log(1-theta_c)
            y_pred[i] = ATHEISM if atheism_prob > books_prob else BOOKS
        return np.array(y_pred)

    train_acc = np.sum(train_label == predict(parsed_train_data)) / len(train_label)
    print(f"Train accuracy: {train_acc}")

    test_docs = list(test_dict.keys())
    test_docs.sort()
    test_docs = np.array(test_docs)
    test_data = np.array([[ 1 if word_id in test_dict[doc_id] else 0 for word_id in words] for doc_id in test_docs])
    test_label = test_label[test_docs - 1]
    test_acc = np.sum(test_label == predict(test_data)) / len(test_label)
    print(f"Test accuracy: {test_acc}")