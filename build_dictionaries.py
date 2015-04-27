#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import pickle


def main():
    inverted_index_dictionary = {}
    direct_index_dictionary = {}

    with open("raw_inverted_index_all", 'r') as raw_inverted_index_file, \
            open("inverted_index", 'wb') as inverted_index_file, \
            open("coord_index", 'wb') as coord_index_file:

        for line in raw_inverted_index_file:
            term, posting_list, tfs, positions = line.strip().split('\t')

            output1 = '\t'.join([posting_list, tfs, '\n'])
            offset1 = inverted_index_file.tell()

            output2 = positions
            offset2 = coord_index_file.tell()

            inverted_index_dictionary[term] = (offset1, len(output1), offset2, len(output2))

            print(output1, end='', file=inverted_index_file)
            print(output2, end='', file=coord_index_file)


    with open("direct_index_all", 'r') as direct_index:
        while True:
            line = direct_index.readline()

            if not line:
                break

            doc_id, doc_length = line.strip().split('\t')

            direct_index_dictionary[int(doc_id)] = int(doc_length)

    with open('inverted_index_dictionary', 'wb') as inverted_dictionary_file:
        pickle.dump(inverted_index_dictionary, inverted_dictionary_file, 2)

    with open('direct_index_dictionary', 'wb') as direct_dictionary_file:
        pickle.dump(direct_index_dictionary, direct_dictionary_file, 2)


if __name__ == '__main__':
    main()