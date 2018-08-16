import numpy
import collections
from sklearn.feature_extraction import text
from sklearn.cluster import DBSCAN
from sklearn import metrics


def main():
    corpus, true_labels = _read_corpus('compact_clean_reviews.csv')
    vectorizer = text.TfidfVectorizer()
    input_data = vectorizer.fit_transform(corpus)
    db = DBSCAN(eps=0.97, min_samples=10).fit(input_data)
    labels = db.labels_

    counter = collections.Counter(true_labels)
    print('Records by cluster:', counter)
    
    # number of members by cluster
    result_set = {}
    for cluster in set(labels):
        result_set[cluster] = 0
    for lable in labels:
        result_set[lable] += 1

    hits = _count_hits(true_labels, labels)

    print('Clustering results:', result_set)
    print('Confusion Matrix:', hits)
    print('Model Accuracy:', _calculate_accuracy(hits, len(true_labels)))


def _calculate_accuracy(hits, records):
    return (hits['0'][0] + hits['1'][1]) / records


def _read_corpus(reviews_file_path):
    data = []
    labels = []
    with open(reviews_file_path) as reviews:
        for review in reviews:
            str_review = review.rstrip().split(',')
            data.append(str_review[0])
            labels.append(str_review[1])
    return data, labels


def _count_hits(true_labels, labels):
    hits = {}
    for i in range(0, len(true_labels)):
        if hits.get(int(true_labels[i])):
            if hits[int(true_labels[i])].get(labels[i]):
                hits[int(true_labels[i])][labels[i]] += 1
            else:
                hits[int(true_labels[i])][labels[i]] = 1
        else:
            hits[int(true_labels[i])] = {labels[i]:1}
    return hits

if __name__ == '__main__':
    main()