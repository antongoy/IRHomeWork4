#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import sys
import math

from compress import *
from base64 import b64encode

__author__ = 'anton-goy'


def compress_posting_list(posting_list, compression_method):
    posting_list.sort()
    gaps = to_gaps(posting_list)

    return compression_method(gaps)


def compute_tfs(posting_list_dict, posting_list):
    return [posting_list_dict[doc_id] for doc_id in posting_list]


def compress_term_positions(posting_list_positions, posting_list, compression_method):
    return ','.join([b64encode(compression_method(posting_list_positions[doc_id]))
                     for doc_id in posting_list])


def main():
    if sys.argv[1] == 'varbyte':
        compression_method = varbyte_compress
    elif sys.argv[1] == 'simple9':
        compression_method = simple9_compress
    else:
        print('Wrong compression method')
        return

    current_word = None
    posting_list_tfs = {}
    posting_list_positions = {}

    for line in sys.stdin:
        word, position, doc_id = line.strip().split('\t')

        doc_id = int(doc_id)
        position = int(position)

        if not current_word:
            current_word = word

        if current_word != word:
            posting_list = posting_list_tfs.keys()

            encode_posting_list = compress_posting_list(posting_list, compression_method)
            encode_tfs = compression_method(compute_tfs(posting_list_tfs, posting_list))
            encode_positions = compress_term_positions(posting_list_positions, posting_list, compression_method)

            print(current_word, b64encode(encode_posting_list), b64encode(encode_tfs), encode_positions, sep='\t')

            current_word = word
            posting_list_tfs = {}
            posting_list_positions = {}

        if not doc_id in posting_list_tfs:
            posting_list_tfs[doc_id] = 1
            posting_list_positions[doc_id] = [position]
        else:
            posting_list_tfs[doc_id] += 1
            posting_list_positions[doc_id].append(position)

if __name__ == '__main__':
    main()
