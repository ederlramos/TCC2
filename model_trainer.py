import numpy
import collections
from sklearn.feature_extraction import text
from sklearn.cluster import DBSCAN
from sklearn import metrics


def main():
    corpus, labels_true = _read_corpus()
    vectorizer = text.TfidfVectorizer()
    input_data = vectorizer.fit_transform(corpus)
    db = DBSCAN(eps=0.98, min_samples=10).fit(input_data)
    core_samples_mask = numpy.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    counter = collections.Counter(labels_true)
    print('Records by cluster:', counter)
    
    # number of members by cluster
    xpto = {}
    for cluster in set(labels):
        xpto[cluster] = 0
    for label in labels:
        xpto[label] += 1

    hits = {
        '0':{
            -1:0,
            0:0,
            1:0
        },
        '1':{
            -1:0,
            0:0,
            1:0
        },
    }
     
    for j in range(0, len(labels_true)):
        hits[labels_true[j]][labels[j]] += 1

    print('Clustering results:', xpto)
    print('Confusion Matrix:', hits)
    print('Model Accuracy:', _calculate_accuracy(hits, len(labels_true)))


def _calculate_accuracy(hits, records):
    return (hits['0'][0] + hits['1'][1]) / records


def _read_corpus():
    data = []
    labels = []
    with open('clean_reviews.csv') as reviews:
        for review in reviews:
            str_review = review.rstrip().split(',')
            data.append(str_review[0])
            labels.append('0' if str_review[1] == 'positive' else '1')
    return data, labels

if __name__ == '__main__':
    main()