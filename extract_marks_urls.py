#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import re

__author__ = 'anton-goy'


def url_normalization(url):
    if re.search('\?[^/]+$', url):
        return url

    if url[-1] != '/':
        return url + '/'
    else:
        return url


urls_filename = sys.argv[1]

urls = set()

for line in open(urls_filename, 'r'):
    doc_id, url = line.strip().split('\t')
    url = url_normalization(url)
    urls.add(url)

for line in open('marks.tsv', 'r'):
    _, url = line.strip().split('\t')
    url = url_normalization(url)
    if url in urls:
        print(line.strip())
