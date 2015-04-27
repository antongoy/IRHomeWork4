#!/usr/bin/env python
from __future__ import print_function

import re
import sys

from base64 import b64decode
from zlib import decompress
from stop_words import stop_words

from lxml.etree import XPath
from lxml.html import document_fromstring
from lxml.html.clean import Cleaner

__author__ = 'anton-goy'


def main():
    split_regexp = re.compile('\w+', re.U)
    cleaner = Cleaner(style=True)

    for line in sys.stdin:
        doc_id, document = line.strip().split('\t')

        document = unicode(decompress(b64decode(document)), encoding='utf-8')
        document = document_fromstring(cleaner.clean_html(document))
        document = ' '.join(XPath('.//text()')(document)).lower()

        words = [word for word in re.findall(split_regexp, document) if not word in stop_words]
        print(*[('%s\t%s\t%s' % (word, position, doc_id)).encode('utf-8')
                for position, word in enumerate(words)], sep='\n')


if __name__ == '__main__':
    main()
