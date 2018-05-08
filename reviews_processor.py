import csv
import string
import operator
import math
import random
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import ngrams

def main():
    lexicon_dictionary = _read_lexicon()
    # _count_ngrams(1, lexicon_dictionary)
    # _count_ngrams(2, lexicon_dictionary)
    # _count_ngrams(3, lexicon_dictionary)
    reviews = _process_reviews(lexicon_dictionary)
    _write_reviews(reviews, 'clean_reviews.csv')


def _process_reviews(lexicon_dictionary):
    review_counter = [0,0,0,0,0]
    reviews = []
    with open('raw_reviews.csv') as raw_file:
        csv_file = csv.reader(raw_file, delimiter=',')
        for row in csv_file:
            rating = row[2]
            if rating == '1':
                review_counter[0] += 1
                reviews.append([row[1], _sanitize_record(row[1], lexicon_dictionary), 'negative'])
            if rating == '2':
                review_counter[1] += 1
                reviews.append([row[1], _sanitize_record(row[1], lexicon_dictionary), 'negative'])
            if rating == '3':
                review_counter[2] += 1
                reviews.append([row[1], _sanitize_record(row[1], lexicon_dictionary), 'negative'])
            if rating == '4':
                review_counter[3] += 1
                reviews.append([row[1], _sanitize_record(row[1], lexicon_dictionary), 'positive'])
            if rating == '5':
                review_counter[4] += 1
                reviews.append([row[1], _sanitize_record(row[1], lexicon_dictionary), 'positive'])
    print('Sample division by stars:', review_counter)
    print('Total records:', sum(review_counter))
    return reviews


def _count_ngrams(n, lexicon_dictionary):
    ngram_result = {}
    with open('teste.csv') as raw_file:
        csv_file = csv.reader(raw_file, delimiter=',')
        for row in csv_file:
            sanitized = _remove_stopwords(row[0].translate(str.maketrans('', '', string.punctuation)))
            records_ngrams = ngrams(sanitized.split(), n)
            for ngram in records_ngrams:
                if '-'.join(ngram) in ngram_result:
                    ngram_result['-'.join(ngram)] += 1
                else:
                    ngram_result['-'.join(ngram)] = 1
    sorted_ngrams = sorted(ngram_result.items(), key=operator.itemgetter(1))
    sorted_ngrams.reverse()
    print('-- metadata about {} grams --'.format(n))
    print('unique ngrams count', len(ngram_result))
    print('top 20 N-Grams')
    for i, record in enumerate(sorted_ngrams):
        print(record)
        if i == 20:
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


def _sanitize_record(record, lexicon_dictionary):
    clean_sentence = _remove_stopwords(record)
    return _preprocess_text(clean_sentence, lexicon_dictionary)


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


def _write_reviews(reviews, file_name):
    _write_compact_reviews(reviews, 'compact_' + file_name)
    with open(file_name, mode='w') as output:
        for review in reviews:            
            if len(review) > 0:
                output.write(review[0] + ',' + review[1] + ',' + review[2] + '\n')


def _write_compact_reviews(reviews, file_name):
    with open(file_name, mode='w') as output:
        for review in reviews:            
            if len(review) > 0:
                output.write(review[1] + ',' + review[2] + '\n')


def _read_lexicon():
    lexicon_dictionary = {}
    with open('clean_liwc.txt') as reader:
        for row in reader:
            word, polarity = row.split(',')
            lexicon_dictionary[word.rstrip()] = polarity.rstrip()
    return lexicon_dictionary


if __name__ == '__main__':
    main()