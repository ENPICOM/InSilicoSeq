#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from iss import generator
from iss.util import cleanup
from iss.error_models import ErrorModel, basic, kde

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from nose.tools import with_setup, raises

import os
import sys
import random
import numpy as np

# due to inconsistent seeding between python 2 and 3, some of the following
# tests are disabled with python2


def setup_function():
    output_file_prefix = 'data/.test'


def teardown_function():
    cleanup(['data/.test.iss.tmp.my_genome.0_R1.fastq',
             'data/.test.iss.tmp.my_genome.0_R2.fastq'])


@raises(SystemExit)
def test_cleanup_fail():
    cleanup('data/does_not_exist')


@with_setup(setup_function, teardown_function)
def test_simulate_and_save():
    err_mod = basic.BasicErrorModel()
    ref_genome = SeqRecord(
        Seq(str('AAAAACCCCC' * 100)),
        id='my_genome',
        description='test genome'
    )
    generator.reads(ref_genome, err_mod, 1000, 0, 'data/.test', 0, ('basic', None), True)


@with_setup(setup_function, teardown_function)
def test_simulate_and_save_short():
    err_mod = basic.BasicErrorModel()
    ref_genome = SeqRecord(
        Seq(str('AACCC' * 100)),
        id='my_genome',
        description='test genome'
    )
    generator.reads(ref_genome, err_mod, 1000, 0, 'data/.test', 0, ('basic', None), True)


@raises(AssertionError)
def test_small_input():
    err_mod = kde.KDErrorModel('data/ecoli.npz')
    ref_genome = SeqRecord(
        Seq(str('AAAAACCCCC')),
        id='my_genome',
        description='test genome'
    )
    generator.simulate_read(ref_genome, err_mod, 1, 0, ('kde', 'auto'))


def test_basic():
    if sys.version_info > (3,):
        random.seed(42)
        np.random.seed(42)
        err_mod = basic.BasicErrorModel()
        ref_genome = SeqRecord(
            Seq(str('AAAAACCCCC' * 100)),
            id='my_genome',
            description='test genome'
        )
        read_tuple = generator.simulate_read(ref_genome, err_mod, 1, 0, ('basic', None))
        big_read = ''.join(str(read_tuple[0].seq) + str(read_tuple[1].seq))
        assert big_read[-15:] == 'TTTTGGGGGTTTTTG'


def test_kde():
    if sys.version_info > (3,):
        random.seed(42)
        np.random.seed(42)
        err_mod = kde.KDErrorModel('data/ecoli.npz')
        ref_genome = SeqRecord(
            Seq(str('CGTTTCAACC' * 400)),
            id='my_genome',
            description='test genome'
        )
        read_tuple = generator.simulate_read(ref_genome, err_mod, 1, 0, ('kde', 'auto'))
        big_read = ''.join(str(read_tuple[0].seq) + str(read_tuple[1].seq))
        assert big_read[:15] == 'CCGTTTCAACCCGTT'


def test_kde_short():
    if sys.version_info > (3,):
        random.seed(42)
        np.random.seed(42)
        err_mod = kde.KDErrorModel('data/ecoli.npz')
        ref_genome = SeqRecord(
            Seq(str('AAACC' * 100)),
            id='my_genome',
            description='test genome'
        )
        read_tuple = generator.simulate_read(ref_genome, err_mod, 1, 0, ('kde', 'auto'))
        big_read = ''.join(str(read_tuple[0].seq) + str(read_tuple[1].seq))
        assert big_read == 'ACCAAACCAAACCAAACCAAGGTTTGGTTTGGTTTGGTGT'

def test_kde_low_quality():
    if sys.version_info > (3,):
        random.seed(42)
        np.random.seed(42)
        err_mod = kde.KDErrorModel('data/ecoli.npz')
        err_mod.quality_forward = err_mod.quality_reverse = [np.tile(([0.,
        0.02931223, 0.04742587, 0.07585818, 0.11920292,
        0.18242552, 0.26894142, 0.37754067, 0.5, 0.62245933,
        0.73105858, 0.81757448, 0.88079708, 0.92414182, 0.95257413,
        0.97068777, 0.98201379, 0.98901306, 1., 1., 1., 1., 1., 1., 1., 
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1.]), (20, 1)) for i in range(4)]

        ref_genome = SeqRecord(
            Seq(str('AAACC' * 100)),
            id='my_genome',
            description='test genome'
        )
        read_tuple = generator.simulate_read(ref_genome, err_mod, 1, 0, ('kde', 'low'))
        big_read = ''.join(str(read_tuple[0].seq) + str(read_tuple[1].seq))
        assert big_read == 'ACCATTCTACACCAAAGCAAAGTCCGATTGGGTTTGCTGT'

def test_kde_middle_low_quality():
    if sys.version_info > (3,):
        random.seed(42)
        np.random.seed(42)
        err_mod = kde.KDErrorModel('data/ecoli.npz')
        err_mod.quality_forward = err_mod.quality_reverse = [np.tile(([0., 0., 0., 0., 
            0.02188127, 0.02931223, 0.03916572, 0.05215356, 0.06913842, 0.09112296,
            0.11920292, 0.15446527, 0.19781611, 0.24973989, 0.31002552,
            0.37754067, 0.450166  , 0.52497919, 0.59868766, 0.66818777,
            0.73105858, 0.78583498, 0.83201839, 0.86989153, 0.90024951,
            0.92414182, 0.94267582, 0.95689275, 0.96770454, 0.97587298,
            0.98201379, 0.98661308, 1., 1., 1., 1., 1., 1., 1., 1.,
            1.]), (20, 1)) for i in range(4)]
        ref_genome = SeqRecord(
            Seq(str('AAACC' * 100)),
            id='my_genome',
            description='test genome'
        )
        read_tuple = generator.simulate_read(ref_genome, err_mod, 1, 0, ('kde', 'middle_low'))
        big_read = ''.join(str(read_tuple[0].seq) + str(read_tuple[1].seq))
        assert big_read == 'ACCATACCATACCAAACCAAGGTGAGGTTCGGTTTGGTTT'

def test_kde_middle_high_quality():
    if sys.version_info > (3,):
        random.seed(42)
        np.random.seed(42)
        err_mod = kde.KDErrorModel('data/ecoli.npz')
        err_mod.quality_forward = err_mod.quality_reverse = [np.tile(([0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.02931223,
        0.04742587, 0.07585818, 0.11920292, 0.18242552, 0.26894142,
        0.37754067, 0.5, 0.62245933, 0.73105858, 0.81757448,
        0.88079708, 0.92414182, 0.95257413, 0.97068777, 0.98201379,
        0.98901306, 1., 1., 1., 1., 1.]), (20, 1)) for i in range(4)]
        ref_genome = SeqRecord(
            Seq(str('AAACC' * 100)),
            id='my_genome',
            description='test genome'
        )
        read_tuple = generator.simulate_read(ref_genome, err_mod, 1, 0, ('kde', 'middle_high'))
        big_read = ''.join(str(read_tuple[0].seq) + str(read_tuple[1].seq))
        assert big_read == 'ACCAAACCAAACCAAACCAAGGTTTGGTTTGGTTTGGTTT'

def test_kde_high_quality():
    if sys.version_info > (3,):
        random.seed(42)
        np.random.seed(42)
        err_mod = kde.KDErrorModel('data/ecoli.npz')
        err_mod.quality_forward = err_mod.quality_reverse = [np.tile(([0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.11920292,
        0.5, 0.88079708, 0.98201379, 1., 1., 1.]), (20, 1)) for i in range(4)]

        ref_genome = SeqRecord(
            Seq(str('AAACC' * 100)),
            id='my_genome',
            description='test genome'
        )
        read_tuple = generator.simulate_read(ref_genome, err_mod, 1, 0, ('kde', 'high'))
        big_read = ''.join(str(read_tuple[0].seq) + str(read_tuple[1].seq))
        assert big_read == 'ACCAAACCAAACCAAACCAAGGTTTGGTTTGGTTTGGTTT'