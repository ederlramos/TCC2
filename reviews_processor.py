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
    reviews = _process_reviews(lexicon_dictionary)
    # xpto = [r[0] for r in _sample_records(reviews, 0.20)]
    # _calculate_document_frequency(10, 1, xpto)
    # _calculate_document_frequency(10, 2, xpto)
    # _calculate_document_frequency(10, 3, xpto)
    _write_reviews(_sample_records(reviews, 0.20), 'clean_reviews_v4.csv')


def _process_reviews(lexicon_dictionary):
    review_counter = [0,0,0,0,0]
    reviews = []
    with open('raw_reviews.csv') as raw_file:
        csv_file = csv.reader(raw_file, delimiter=',')
        for row in csv_file:
            rating = row[2]
            if rating == '1':
                review_counter[0] += 1
                reviews.append([row[1], _sanitize_record(row[1], lexicon_dictionary), 'negative', rating])
            elif rating == '2':
                review_counter[1] += 1
                reviews.append([row[1], _sanitize_record(row[1], lexicon_dictionary), 'negative', rating])
            elif rating == '3':
                review_counter[2] += 1
                reviews.append([row[1], _sanitize_record(row[1], lexicon_dictionary), 'positive', rating])
            elif rating == '4':
                review_counter[3] += 1
                reviews.append([row[1], _sanitize_record(row[1], lexicon_dictionary), 'positive', rating])
            elif rating == '5':
                review_counter[4] += 1
                reviews.append([row[1], _sanitize_record(row[1], lexicon_dictionary), 'positive', rating])
    print('Sample division by stars:', review_counter)
    print('Total records:', sum(review_counter))
    return reviews


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


def _sample_records(reviews, percentage):
    positive_reviews = []
    negative_reviews = []

    for review in reviews:
        if review[2] == 'negative':
            negative_reviews.append(review)
        else:
            positive_reviews.append(review)
    
    positive_reviews = positive_reviews[1:math.floor(len(positive_reviews) * percentage)]
    negative_reviews = negative_reviews[1:math.floor(len(negative_reviews) * percentage)]
    positive_reviews.extend(negative_reviews)

    review_counter = [0,0,0,0,0]
    for review in positive_reviews:
        review_counter[int(review[3]) - 1] += 1    
    
    print('Sample size:', len(positive_reviews))
    return positive_reviews


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