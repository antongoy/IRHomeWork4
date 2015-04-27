#!/usr/bin/env python
from __future__ import print_function

import re
import sys

from base64 import b64decode, b64encode
from zlib import decompress, compress

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

        words = re.findall(split_regexp, document)

        encode_text = b64encode(compress(",".join(words).encode('utf-8'), 1))

        print(doc_id, len(words), encode_text, sep='\t')


if __name__ == '__main__':
    main()
