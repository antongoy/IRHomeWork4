#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import re
import sys
import random
import heapq

from compress import *

from math import log10
from base64 import b64decode
from zlib import decompress
from pickle import load
from itertools import islice
from bisect import bisect_left
from stop_words import stop_words

__author__ = 'anton-goy'


def parse_arguments():
    compression_method = sys.argv[1]

    if compression_method == 'varbyte':
        return varbyte_uncompress, sys.argv[2]
    elif compression_method == 'simple9':
        return simple9_uncompress, sys.argv[2]
    else:
        raise AttributeError("Wrong compression method")


def read_inverted_index(offset, length):
    global inverted_index_file, uncompress_method

    inverted_index_file.seek(offset)

    encode_line = inverted_index_file.read(length)
    encode_posting_list, encode_tfs = encode_line.strip().split('\t')

    posting_list = from_gaps(uncompress_method(b64decode(encode_posting_list)))
    tfs = uncompress_method(b64decode(encode_tfs))

    return map(int, posting_list), map(int, tfs)


def read_direct_index(offset, length):
    global direct_index_file

    direct_index_file.seek(offset)

    encode_line = direct_index_file.read(length)
    _, _, text = encode_line.strip().split('\t')

    words = decompress(b64decode(text)).split(',')

    return words


def get_term_data(terms):
    global inverted_dictionary, n_documents

    terms_posting_lists = []
    terms_tfs = []

    for term in terms:
        try:
            offset, length, _, _ = inverted_dictionary[term]
        except KeyError:
            terms_posting_lists.append([])
            terms_tfs.append([])

            continue

        posting_list, tfs = read_inverted_index(offset, length)

        terms_posting_lists.append(posting_list)
        terms_tfs.append(tfs)

    terms_idfs = compute_idfs([len(posting_list) for posting_list in terms_posting_lists], len(terms))

    return terms_posting_lists, terms_tfs, terms_idfs


def index(array, element):
    found_index = bisect_left(array, element)

    if found_index != len(array) and array[found_index] == element:
        return found_index

    raise ValueError


def posting_list_intersection(terms, terms_posting_lists):
    found_docs_indexes = {}

    if not all(terms_posting_lists):
        return [], found_docs_indexes

    found_docs = list(reduce(lambda x, y: x & y, [set(posting_list)
                                                  for posting_list in terms_posting_lists if posting_list]))

    for doc_id in found_docs:
        for term, posting_list in zip(terms, terms_posting_lists):
            i = index(posting_list, doc_id)
            found_docs_indexes[(term, doc_id)] = i

    return found_docs, found_docs_indexes


def initial_ranking(found_docs, found_docs_indexes, terms, terms_tfs, terms_idfs):
    k = 2
    b = 0.75

    doc_ranks = {}

    for doc_id in found_docs:
        bm25 = 0.0
        for term_index, term in enumerate(terms):
            doc_index = found_docs_indexes[(term, doc_id)]

            tf = terms_tfs[term_index][doc_index]
            idf = terms_idfs[term_index]
            bm25 += (tf * idf) / (tf + k * (b + direct_dictionary[doc_id] * (1 - b)))

        doc_ranks[doc_id] = bm25

    return doc_ranks


def compute_inverses(passage):
    passage = [p for p in passage if p >= 0]

    n_inversions = sum([1 for i in range(len(passage))
                        for j in range(i, len(passage)) if passage[i] > passage[j]])

    return n_inversions


def compute_distinct_terms_in_passage(terms, terms_in_passage):
    return sum([1 for term in terms if term in terms_in_passage])


def compute_tfs_in_passage(terms, tfs):
    return [tfs[term] if term in tfs else 0 for term in terms]


def get_terms_in_passage(start_index, end_index, terms_in_document):
    terms_in_passage_list = [(pos, word) for pos, word in terms_in_document
                             if start_index <= pos <= end_index]
    tfs = {}
    for _, term in terms_in_passage_list:
        if term in tfs:
            tfs[term] += 1
        else:
            tfs[term] = 1

    return tfs


def compute_passage_features(passage, start_index, end_index, terms_in_document, terms, terms_idfs):
    terms_in_passage = get_terms_in_passage(start_index, end_index, terms_in_document)

    n_distinct_terms_in_passage = compute_distinct_terms_in_passage(terms, terms_in_passage)
    n_terms = len(terms)
    completeness = n_distinct_terms_in_passage / n_terms

    tfs = compute_tfs_in_passage(terms, terms_in_passage)
    tf_idf = sum([tfs[i] * terms_idfs[i] for i in range(n_terms)])

    n_terms_in_passage = sum(tfs)
    passage_text_length = end_index - start_index + 1
    density = 1 / (passage_text_length - n_terms_in_passage + 1)

    n_inverse = 1 / (compute_inverses(passage) + 1)

    return [completeness, density, tf_idf, n_inverse]


def passage_ranking(heap, terms, terms_idfs, terms_positions, parameters):
    global direct_dictionary

    def window(seq, n=2):
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    n_terms = len(terms)
    passage_ranks = []

    for doc_id, _ in heap:
        current_passage = [-1] * n_terms
        best_passage_rank = -1

        terms_in_document = [(pos, term) for term in terms for pos in terms_positions[(term, doc_id)]]
        terms_in_document.sort(key=lambda x: x[0])

        for sliding_window in window(terms_in_document, n_terms):
            is_useful = False

            window_start_index = sliding_window[0][0]

            for term_pos, term in enumerate(terms):
                for pos_in_window, word in sliding_window:
                    if word == term and current_passage[term_pos] < window_start_index:
                        current_passage[term_pos] = pos_in_window
                        is_useful = True
                        break

            if is_useful:
                end_index = max(current_passage)
                start_index = min(current_passage, key=lambda x: x if x != -1 else end_index)

                passage_features = \
                    compute_passage_features(current_passage, start_index, end_index, terms_in_document,
                                             terms,
                                             terms_idfs)

                current_passage_rank = sum([p * f for p, f in zip(parameters, passage_features)])

                if current_passage_rank > best_passage_rank:
                    best_passage_rank = current_passage_rank

        passage_ranks.append(best_passage_rank)

    return passage_ranks


def compute_idfs(posting_list_lengths, n_terms):
    global n_documents

    return [log10((n_documents + n_terms) / (length + 1)) for length in posting_list_lengths]


def get_terms_positions(terms, heap, found_docs_indexes):
    global coord_index_file, inverted_dictionary, uncompress_method

    terms_positions = {}
    memo = {}

    for term in terms:
        for doc_id, _ in heap:
            if not term in memo:
                _, _, offset, length = inverted_dictionary[term]

                coord_index_file.seek(offset)
                positions = coord_index_file.read(length).split(',')
                memo[term] = positions

            i = found_docs_indexes[(term, doc_id)]
            terms_positions[(term, doc_id)] = uncompress_method(b64decode(memo[term][i]))

    return terms_positions


def compute_quality(parameters, data):
    heap_max_size = 100

    avg_pos = 0
    n_successes = 0

    for query, relevant_url in data:
        query = query.strip().lower()
        terms = [term for term in query.split() if not unicode(term, encoding='utf-8') in stop_words]

        if not terms:
            continue

        terms_posting_lists, terms_tfs, terms_idfs = get_term_data(terms)

        found_docs, found_docs_indexes = posting_list_intersection(terms, terms_posting_lists)

        if not found_docs:
            continue

        initial_ranks = initial_ranking(found_docs, found_docs_indexes, terms, terms_tfs, terms_idfs)

        heap = [(doc_id, initial_ranks[doc_id]) for doc_id in found_docs]
        heap = heapq.nlargest(heap_max_size, heap, key=lambda x: x[1])

        terms_positions = get_terms_positions(terms, heap, found_docs_indexes)

        passage_ranks = passage_ranking(heap, terms, terms_idfs, terms_positions, parameters)

        output_ranking = [(doc_id, url_dictionary[doc_id], rank)
                          for (doc_id, _), rank in zip(heap, passage_ranks)]

        output_ranking.sort(key=lambda x: x[2])

        relevant_url = url_normalization(relevant_url)

        for i, (doc_id, url, rank) in enumerate(output_ranking):
            url = url_normalization(url)
            if url == relevant_url:
                avg_pos += i
                n_successes += 1
                break

    return avg_pos / n_successes, n_successes


def url_normalization(url):
    if re.search('\?[^/]+$', url):
        return url

    if url[-1] != '/':
        return url + '/'
    else:
        return url


def generate_trial_vector(i, population, F):
    population_size = len(population)

    n = random.randint(0, population_size - 1)
    while n == i:
        n = random.randint(0, population_size - 1)

    k = random.randint(0, population_size - 1)
    while k == i or k == n:
        k = random.randint(0, population_size - 1)

    m = random.randint(0, population_size - 1)
    while m == i or m == n or m == k:
        m = random.randint(0, population_size - 1)

    new_vector = [population[n][t] + F * (population[k][t] - population[m][t]) for t in range(4)]

    for coord in range(len(new_vector)):
        if random.randint(0, 4) >= 3:
            new_vector[coord] = population[i][coord]

    return new_vector


def main():
    print('Start the main function...')

    data = []

    population_size = 5
    F = 0.5

    with open('valid_marks.tsv') as marks_file:
        for line in marks_file:
            data.append(line.strip().split('\t'))

    #print(compute_quality([1, 1, 20, 2], data[:50]))
    population = [[random.randint(0, 100) for i in range(4)] for i in range(population_size)]
    qualities = [compute_quality(population[i], data) for i in range(population_size)]

    n_generations = 20

    for t in range(n_generations):
        print("Generation #%d" % t)
        new_population = []

        for i in range(population_size):
            trial_vector = generate_trial_vector(i, population, F)
            trial_vector_quality = compute_quality(trial_vector, data)

            if trial_vector_quality < qualities[i]:
                qualities[i] = trial_vector_quality

            new_population.append(trial_vector)

        print(qualities)

    print(new_population)
    print(qualities)


def building_dicts():
    global inverted_dictionary, direct_dictionary, n_documents

    print('Loading the dictionaries from file....')
    direct_dictionary = load(direct_index_dictionary_file)
    inverted_dictionary = load(inverted_index_dictionary_file)

    print('Make the url dictionary...')
    for line in urls_file:
        doc_id, url = line.strip().split('\t')
        doc_id = int(doc_id)

        url_dictionary[doc_id] = url

    n_documents = len(url_dictionary)


if __name__ == '__main__':
    uncompress_method, url_filename = parse_arguments()

    with open('inverted_index_dictionary', 'rb') as inverted_index_dictionary_file, \
            open('direct_index_dictionary', 'rb') as direct_index_dictionary_file, \
            open('inverted_index', 'r') as inverted_index_file, \
            open('direct_index', 'r') as direct_index_file, \
            open('coord_index', 'r') as coord_index_file, \
            open(url_filename, 'r') as urls_file:
        url_dictionary = {}
        inverted_dictionary = {}
        direct_dictionary = {}

        n_documents = 0

        building_dicts()
        main()
