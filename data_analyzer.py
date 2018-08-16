import csv
import string
import operator
import math
import random
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import ngrams


def main():
    records = _read_records()
    print('Analyzing {} records'.format(len(records)))
    # split_reviews = _split_data(records)
    split_reviews = [r[1] for r in records]
    
    #for reviews in split_reviews:
        #print('Analyzing {} reviews by rating {}'.format(len(reviews), split_reviews.index(reviews) + 1))
    _calculate_document_frequency(10, 1, split_reviews)
        # _count_top_ngrams(20, 1, reviews)
        # _count_top_ngrams(10, 2, reviews)
        # _count_top_inversion_ngrams(10, 2, reviews)
        # _count_top_ngrams(10, 3, reviews)
        # _count_top_inversion_ngrams(10, 3, reviews)


def _read_records():
    records = []
    with open('raw_reviews.csv') as raw_file:
        csv_file = csv.reader(raw_file, delimiter=',')
        for row in csv_file:
            records.append(row)
    return records


def _split_data(records):
    one_star = []
    two_star = []
    three_star = []
    four_star = []
    five_star = []
    for review in records:
        rating = review[2]
        if rating == '1':
            one_star.append(review[1])
        elif rating == '2':
            two_star.append(review[1])
        elif rating == '3':
            three_star.append(review[1])
        elif rating == '4':
            four_star.append(review[1])
        else:
            five_star.append(review[1])
    return [one_star, two_star, three_star, four_star, five_star]


def _calculate_document_frequency(n, ngram_size, records):
    ngram_df = {}
    for row in records:
        sanitized = _remove_stopwords(row.translate(str.maketrans('', '', string.punctuation)))
        record_ngrams = ngrams(sanitized.split(), ngram_size)
        for ngram in set(record_ngrams):
            if '-'.join(ngram) in ngram_df:
                ngram_df['-'.join(ngram)] += 1
            else:
                ngram_df['-'.join(ngram)] = 1
    sorted_ngrams = sorted(ngram_df.items(), key=operator.itemgetter(1))
    sorted_ngrams.reverse()
    print('-- DF about {} grams --'.format(ngram_size))
    print('Unique ngrams:', len(ngram_df))
    print('top {} N-Grams:'.format(n))
    for i, ngram in enumerate(sorted_ngrams):
        print(ngram)
        if i == n:
            break



def _count_top_ngrams(n, ngram_size, records):
    ngram_tf = {}
    for row in records:
        sanitized = _remove_stopwords(row.translate(str.maketrans('', '', string.punctuation)))
        record_ngrams = ngrams(sanitized.split(), ngram_size)
        for ngram in record_ngrams:
            if '-'.join(ngram) in ngram_tf:
                ngram_tf['-'.join(ngram)] += 1
            else:
                ngram_tf['-'.join(ngram)] = 1
    sorted_ngrams = sorted(ngram_tf.items(), key=operator.itemgetter(1))
    sorted_ngrams.reverse()
    print('-- metadata about {} grams --'.format(ngram_size))
    print('Unique ngrams:', len(ngram_tf))
    print('top {} N-Grams:'.format(n))
    for i, ngram in enumerate(sorted_ngrams):
        print(ngram)
        if i == n:
            break


def _count_top_inversion_ngrams(n, ngram_size, records):
    ngram_tf = {}
    for row in records:
        sanitized = _remove_stopwords(row.translate(str.maketrans('', '', string.punctuation)))
        record_ngrams = ngrams(sanitized.split(), ngram_size)
        for ngram in record_ngrams:
            if 'não' in ngram or 'nunca' in ngram or 'jamais' in ngram or 'nao' in ngram:
                if '-'.join(ngram) in ngram_tf:
                    ngram_tf['-'.join(ngram)] += 1
                else:
                    ngram_tf['-'.join(ngram)] = 1
    sorted_ngrams = sorted(ngram_tf.items(), key=operator.itemgetter(1))
    sorted_ngrams.reverse()
    print('-- metadata about {} grams --'.format(ngram_size))
    print('Unique ngrams:', len(ngram_tf))
    print('top {} inversion N-Grams:'.format(n))
    for i, ngram in enumerate(sorted_ngrams):
        print(ngram)
        if i == n:
            break


def _detect_anomaly_candidate(record, clean_record, rating, lexicon_dictionary):
    polarity = 0
    considered_tokens = []
    for token in clean_record.split(' '):
        try:
            if lexicon_dictionary[token] == 'pos':
                polarity += 1
                considered_tokens.append(token)
            else:
                polarity += -1
                considered_tokens.append(token)
        except Exception:
            pass

    if polarity > 0 and rating == 'negative':
        print('{} anomaly candidate detected: {} \n considered tokens: {} \n'.format(rating, record, considered_tokens))
    elif polarity <= 0 and rating == 'positive':
        print('{} anomaly candidate detected: {} \n considered tokens: {} \n'.format(rating, record, considered_tokens))


def _remove_stopwords(record):
    stop_words = set(stopwords.words('portuguese'))
    clean_record = []
    for i in record.lower().split():
        if i not in stop_words:
            clean_record.append(i)
    return ' '.join(clean_record)


def _preprocess_text(text, lexicon_dictionary):
    tokenizer = RegexpTokenizer(r'\w+')
    clean_text = ''
    for token in tokenizer.tokenize(text):
        if token in lexicon_dictionary:
            clean_text += ' ' + token
    return clean_text


def _sanitize_record(record, lexicon_dictionary):
    clean_sentence = _remove_stopwords(record)
    return _preprocess_text(clean_sentence, lexicon_dictionary)


def _read_lexicon():
    lexicon_dictionary = {}
    with open('clean_liwc.txt') as reader:
        for row in reader:
            word, polarity = row.split(',')
            lexicon_dictionary[word.rstrip()] = polarity.rstrip()
    return lexicon_dictionary


if __name__ == '__main__':
    main()