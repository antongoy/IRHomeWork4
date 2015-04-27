from __future__ import print_function

import struct
import ctypes
import threading

from itertools import izip_longest, chain

__author__ = 'anton-goy'


def varbyte_compress(numbers):
    return ''.join([varbyte_compress_number(n) for n in numbers])


def varbyte_compress_number(number):
    if number < 128:
        return chr(number ^ 128)

    decomposition = []

    while number >= 128:
        number, remainder = divmod(number, 128)
        decomposition.append(remainder)

    decomposition.append(number)
    encode_string = b''

    for i, n in enumerate(reversed(decomposition)):
        if i != len(decomposition) - 1:
            encode_string += chr(n)
        else:
            encode_string += chr(n ^ 128)

    return encode_string


def varbyte_uncompress(byte_string):
    decode_numbers = []
    n = 0
    for byte in byte_string:
        byte_number = ord(byte)

        if byte_number < 128:
            n = 128 * n + byte_number
        else:
            n = 128 * n + (byte_number - 128)
            decode_numbers.append(n)
            n = 0

    return decode_numbers


def to_gaps(numbers):
    return [n - numbers[i - 1] if i != 0 else n for i, n in enumerate(numbers)]


def from_gaps(gaps):
    n = len(gaps)
    for i, gap in enumerate(gaps, 1):
        if i != n:
            gaps[i] += gap

    return gaps


def simple9_compress(numbers):
    selectors = [(0, 28, 1, 2 ** 1),
                 (1, 14, 2, 2 ** 2),
                 (2, 9, 3, 2 ** 3),
                 (3, 7, 4, 2 ** 4),
                 (4, 5, 5, 2 ** 5),
                 (5, 4, 7, 2 ** 7),
                 (6, 3, 9, 2 ** 9),
                 (7, 2, 14, 2 ** 14),
                 (8, 1, 28, 2 ** 28)]

    encode_numbers = []
    current_number = 0

    n_numbers = len(numbers)
    i = 0
    rest = n_numbers - i

    while i < n_numbers:
        for code, amount, length, limit in selectors:
            if amount > rest:
                continue

            numbers_slice = numbers[i:i + amount]

            for n in numbers_slice:
                if n >= limit:
                    break
            else:
                current_number |= code << 28
                shift = 28 - length
                for x in numbers_slice:
                    current_number |= x << shift
                    shift -= length
                encode_numbers.append(ctypes.c_uint32(current_number))

                current_number = 0
                i += amount
                rest = n_numbers - i

                break

    return ''.join([struct.pack('I', n.value) for n in encode_numbers])


def simple9_uncompress(encode_string):

    masks28 = [(1 << i, i) for i in range(27, -1, -1)]
    masks14 = [(3 << i, i) for i in range(26, -1, -2)]
    masks9 = [(7 << i, i) for i in range(25, -1, -3)]
    masks7 = [(15 << i, i) for i in range(24, -1, -4)]
    masks5 = [(31 << i, i) for i in range(23, -1, -5)]
    masks4 = [(127 << i, i) for i in range(21, -1, -7)]
    masks3 = [(511 << i, i) for i in range(19, -1, -9)]
    masks2 = [(16383 << i, i) for i in range(14, -1, -14)]
    masks1 = [(268435455, 0)]

    def unpack_number(n, masks):
        return [(mask & n) >> shift for mask, shift in masks]

    switch = {0: lambda n: unpack_number(n, masks28),
              1: lambda n: unpack_number(n, masks14),
              2: lambda n: unpack_number(n, masks9),
              3: lambda n: unpack_number(n, masks7),
              4: lambda n: unpack_number(n, masks5),
              5: lambda n: unpack_number(n, masks4),
              6: lambda n: unpack_number(n, masks3),
              7: lambda n: unpack_number(n, masks2),
              8: lambda n: unpack_number(n, masks1)}

    def grouper(string):
        return [struct.unpack('I', string[i:i + 4])[0] for i in range(0, len(string), 4)]

    return list(chain(*[switch[num >> 28](num) for num in grouper(encode_string)]))