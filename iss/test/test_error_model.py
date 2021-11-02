#!/usr/bin/env python
# -*- coding: utf-8 -*-

from iss.error_models import ErrorModel, basic, kde, perfect

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from nose.tools import raises

import random
import numpy as np


def test_perfect_phred():
    err_mod = perfect.PerfectErrorModel()

    distribution = err_mod.gen_phred_scores(20, 'forward')[:10]
    assert distribution == [40, 40, 40, 40, 40, 40, 40, 40, 40, 40]


def test_basic_phred():
    np.random.seed(42)
    err_mod = basic.BasicErrorModel()

    distribution = err_mod.gen_phred_scores(20, 'forward')[:10]
    assert distribution == [23, 19, 25, 40, 19, 19, 40, 26, 18, 23]


def test_kde_phred():
    np.random.seed(42)
    err_mod = kde.KDErrorModel('data/ecoli.npz')
    distribution = err_mod.gen_phred_scores(err_mod.quality_reverse,
                                            'reverse', 'auto')[10:]
    assert distribution == [40, 40, 40, 40, 40, 40, 40, 40, 10, 10]


def test_kde_phred_low():
    np.random.seed(42)
    err_mod = kde.KDErrorModel('data/ecoli.npz')
    err_mod.quality_forward = err_mod.quality_reverse = [np.tile(([0.,
        0.02931223, 0.04742587, 0.07585818, 0.11920292,
        0.18242552, 0.26894142, 0.37754067, 0.5, 0.62245933,
        0.73105858, 0.81757448, 0.88079708, 0.92414182, 0.95257413,
        0.97068777, 0.98201379, 0.98901306, 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1.]), (20, 1)) for i in range(4)]
    distribution = err_mod.gen_phred_scores(err_mod.quality_reverse,
                                            'reverse', 'low')[10:]
    assert distribution == [1, 15, 12, 6, 5, 6, 7, 9, 8, 7]


def test_kde_phred_middle_low():
    np.random.seed(42)
    err_mod = kde.KDErrorModel('data/ecoli.npz')
    err_mod.quality_forward = err_mod.quality_reverse = [np.tile(([0., 0., 0., 0.,
            0.02188127, 0.02931223, 0.03916572, 0.05215356, 0.06913842, 0.09112296,
            0.11920292, 0.15446527, 0.19781611, 0.24973989, 0.31002552,
            0.37754067, 0.450166, 0.52497919, 0.59868766, 0.66818777,
            0.73105858, 0.78583498, 0.83201839, 0.86989153, 0.90024951,
            0.92414182, 0.94267582, 0.95689275, 0.96770454, 0.97587298,
            0.98201379, 0.98661308, 1., 1., 1., 1., 1., 1., 1., 1.,
            1.]), (20, 1)) for i in range(4)]
    distribution = err_mod.gen_phred_scores(err_mod.quality_reverse,
                                            'reverse', 'middle_low')[10:]
    assert distribution == [4, 29, 23, 13, 12, 12, 14, 17, 16, 14]


def test_kde_phred_middle_high():
    np.random.seed(42)
    err_mod = kde.KDErrorModel('data/ecoli.npz')
    err_mod.quality_forward = err_mod.quality_reverse = [np.tile(([0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.02931223,
            0.04742587, 0.07585818, 0.11920292, 0.18242552, 0.26894142,
            0.37754067, 0.5, 0.62245933, 0.73105858, 0.81757448,
            0.88079708, 0.92414182, 0.95257413, 0.97068777, 0.98201379,
            0.98901306, 1., 1., 1., 1., 1.]), (20, 1)) for i in range(4)]
    distribution = err_mod.gen_phred_scores(err_mod.quality_reverse,
                                            'reverse', 'middle_high')[10:]
    assert distribution == [19, 33, 30, 24, 23, 24, 25, 27, 26, 25]


def test_kde_phred_high():
    np.random.seed(42)
    err_mod = kde.KDErrorModel('data/ecoli.npz')
    err_mod.quality_forward = err_mod.quality_reverse = [np.tile(([0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.11920292,
            0.5, 0.88079708, 0.98201379, 1., 1., 1.]), (20, 1)) for i in range(4)]
    distribution = err_mod.gen_phred_scores(err_mod.quality_reverse,
                                            'reverse', 'high')[10:]
    assert distribution == [34, 37, 36, 35, 35, 35, 35, 36, 35, 35]


def test_introduce_errors():
    np.random.seed(42)
    err_mod = basic.BasicErrorModel()

    read = SeqRecord(
        Seq(str('AATGC' * 25)),
        id='read_1',
        description='test read'
    )
    read = err_mod.introduce_error_scores(read, 'forward', ('basic', None))
    qualities = read.letter_annotations["phred_quality"][:10]
    assert qualities == [40, 26, 40, 40, 25, 25, 40, 40, 22, 40]


def test_mut_sequence():
    random.seed(42)
    np.random.seed(42)

    err_mod = basic.BasicErrorModel()

    read = SeqRecord(
        Seq(str('AAAAA' * 25)),
        id='read_1',
        description='test read'
    )
    read.letter_annotations["phred_quality"] = [5] * 125
    read.seq = err_mod.mut_sequence(read, 'forward')
    assert str(read.seq[:10]) == 'AAAACAGAAA'


def test_introduce_indels():
    random.seed(42)
    np.random.seed(42)

    err_mod = basic.BasicErrorModel()
    err_mod.ins_for[1]['G'] = 1.0
    err_mod.del_for[0]['A'] = 1.0
    bounds = (5, 130)
    read = SeqRecord(
        Seq(str('ATATA' * 25)),
        id='read_1',
        description='test read'
    )
    ref_genome = SeqRecord(
        Seq(str('ATATA' * 100)),
        id='ref_genome',
        description='test reference'
    )
    read.seq = err_mod.introduce_indels(
        read, 'forward', ref_genome, bounds)
    assert len(read.seq) == 125
    assert read.seq[:10] == 'ATGATAATAT'


@raises(SystemExit)
def test_bad_err_mod():
    err_mod = kde.KDErrorModel('data/empty_file')
