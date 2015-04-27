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


class HeapNode():
    def __init__(self, x, y):
        self.docid = x
        self.rank = y

    def __cmp__(self, other):
        return cmp(self.rank, other.rank)

    def __str__(self):
        return '({0}, {1})'.format(self.docid, self.rank)

    def __repr__(self):
        return '({0}, {1})'.format(self.docid, self.rank)


def parse_arguments():
    """
    :rtype : tuple
    """
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

    return [int(doc_id) for doc_id in posting_list], \
           [int(tf) for tf in tfs],


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
    k = 1.8
    b = 0.5

    doc_ranks = {}

    for k, doc_id in enumerate(found_docs):
        bm25 = 0.0
        for term_index, term in enumerate(terms):
            doc_index = found_docs_indexes[(term, doc_id)]

            tf = terms_tfs[term_index][doc_index]
            idf = terms_idfs[term_index]
            bm25 += (tf * idf) / (tf + k * (b + direct_dictionary[doc_id][2] * (1 - b)))

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


def compute_terms_in_passage(start_index, end_index, terms_in_document):
    for i, (pos, word) in enumerate(terms_in_document):
        if pos < start_index:
            continue

        if pos == start_index:
            k = i

        if pos == end_index:
            l = i
            break

        if start_index <= pos <= end_index:
            continue

    tfs = {}
    for _, term in terms_in_document[k:l + 1]:
        if term in tfs:
            tfs[term] += 1
        else:
            tfs[term] = 1

    return tfs


def compute_passage_features(passage, start_index, end_index, terms_in_document, terms, terms_idfs):
    n_terms = len(terms)

    terms_in_passage = compute_terms_in_passage(start_index, end_index, terms_in_document)

    n_distinct_terms_in_passage = compute_distinct_terms_in_passage(terms, terms_in_passage)
    assert n_distinct_terms_in_passage <= n_terms

    passage_text_length = end_index - start_index + 1

    completeness = n_distinct_terms_in_passage / n_terms
    tfs = compute_tfs_in_passage(terms, terms_in_passage)

    n_terms_in_passage = sum(tfs)
    density = passage_text_length - n_terms_in_passage

    tf_idf = sum([tfs[i] * terms_idfs[i] for i in range(n_terms)])

    n_inverse = len(passage) - compute_inverses(passage)

    return [completeness, density, tf_idf, n_inverse, ]


def passage_ranking(heap, found_docs_indexes, terms, terms_idfs, terms_positions, parameters):
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

    for heap_node in heap:
        doc_id = heap_node.docid

        current_passage = [-1] * n_terms
        current_passage_rank = None

        best_passage_rank = -1

        terms_in_document = [(pos, term) for term in terms for pos in terms_positions[(term, doc_id)]]
        terms_in_document.sort(key=lambda x: x[0])

        for slide_window in window(terms_in_document, n_terms):
            is_window_useful = False

            for term_pos, term in enumerate(terms):
                for word_pos, word in slide_window:

                    if word == term and not current_passage[term_pos] == word_pos:
                        current_passage[term_pos] = word_pos
                        is_window_useful = True

            if is_window_useful:
                end_index = max(current_passage)

                start_index = end_index

                for i in current_passage:
                    if i == -1:
                        continue

                    if i < start_index:
                        start_index = i

                passage_features = compute_passage_features(current_passage,
                                                            start_index,
                                                            end_index,
                                                            terms_in_document,
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
        for heap_node in heap:
            doc_id = heap_node.docid
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

        heap = [HeapNode(doc_id, -initial_ranks[doc_id]) for doc_id in found_docs]
        heap = heapq.nlargest(heap_max_size, heap)

        terms_positions = get_terms_positions(terms, heap, found_docs_indexes)

        passage_ranks = passage_ranking(heap, found_docs_indexes, terms, terms_idfs, terms_positions, parameters)

        output_ranking = [(heap_node.docid, url_dictionary[heap_node.docid], rank)
                          for heap_node, rank in zip(heap, passage_ranks)]

        output_ranking.sort(key=lambda x: x[2])

        relevant_url = url_normalization(relevant_url)

        for i, (doc_id, url, rank) in enumerate(output_ranking):
            url = url_normalization(url)
            if url == relevant_url:
                avg_pos += i
                n_successes += 1
                break

    return avg_pos / n_successes


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

    new_vector = [population[n][t] + F *(population[k][t] - population[m][t]) for t in range(4)]

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

    print('Loading the dictionary from file....')
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
