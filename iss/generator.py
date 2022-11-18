#!/usr/bin/env python
# -*- coding: utf-8 -*-

from iss.util import load, rev_comp

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import GC
from Bio.SeqRecord import SeqRecord

import sys
import random
import logging
from typing import Tuple
import numpy as np


LOGGER = logging.getLogger(__name__)


def reads(record, ErrorModel, n_pairs, cpu_number, output, seed, sequence_type,
          gc_bias=False, record_mmep=None, store_mutations=False) -> str:
    """Simulate reads from one genome (or sequence) according to an ErrorModel

    This function makes use of the `simulate_read` function to simulate reads
    and save them in a fastq file

    Args:
        record (SeqRecord): sequence or genome of reference
        ErrorModel (ErrorModel): an ErrorModel
        n_pairs (int): the number of reads to generate
        cpu_number (int): an int indentifying the cpu that is used by the
            function. Is used for naming the output file
        output (str): the output file prefix
        seed (int): random seed to use
        sequencing_type (str): metagenomics or amplicon sequencing used
        gc_bias (bool): if set, the function may skip a read due to abnormal
            GC content

    Returns:
        str: the name of the output file
    """
    # load the record from disk if memmap_record is given
    if record_mmep is not None:
        record: SeqRecord = load(record_mmep)

    if seed is not None:
        random.seed(seed + cpu_number)
        np.random.seed(seed + cpu_number)
    LOGGER.debug(
        'Cpu #%s: Generating %s read pairs'
        % (cpu_number, n_pairs))
    read_tuple_list = []
    i = 0

    mutation_tuple_list = []

    while i < n_pairs:
        # try:
        #     forward, reverse = simulate_read(record, ErrorModel, i)
        # except ValueError as e:
        #     LOGGER.error('Skipping this record: %s' % record.id)
        #     return
        try:
            forward, reverse = simulate_read(record, ErrorModel, i, cpu_number, sequence_type)
            if store_mutations:
                mutation_tuple_list.extend(forward.annotations["mutations"])
                mutation_tuple_list.extend(reverse.annotations["mutations"])
        except AssertionError as e:
            LOGGER.warning(
                '%s shorter than read length for this ErrorModel' % record.id)
            LOGGER.warning(
                'Skipping %s. You will have less reads than specified'
                % record.id)
            break
        else:
            if gc_bias:
                stiched_seq = forward.seq + reverse.seq
                gc_content = GC(stiched_seq)
                if 40 < gc_content < 60:
                    read_tuple_list.append((forward, reverse))
                    i += 1
                elif np.random.rand() < 0.90:
                    read_tuple_list.append((forward, reverse))
                    i += 1
                else:
                    continue
            else:
                read_tuple_list.append((forward, reverse))
                i += 1

    temp_file_name = output + '.iss.tmp.%s.%s' % (record.id, cpu_number)
    to_fastq(read_tuple_list, temp_file_name)
    if store_mutations:
        to_vcf(mutation_tuple_list, temp_file_name)

    return temp_file_name


def simulate_read(record: SeqRecord, ErrorModel, i, cpu_number, sequence_type) -> Tuple[SeqRecord, SeqRecord]:
    """From a read pair from one genome (or sequence) according to an
    ErrorModel

    Each read is a SeqRecord object
    returns a tuple containing the forward and reverse read.

    Args:
        record (SeqRecord): sequence or genome of reference
        ErrorModel (ErrorModel): an ErrorModel class
        i (int): a number identifying the read
        cpu_number (int): cpu number. Is added to the read id.
        sequence_type (str): metagenomics or amplicon sequencing used

    Returns:
        tuple: tuple containg a forward read and a reverse read
    """
    sequence = record.seq
    header = record.id

    read_length = ErrorModel.read_length
    insert_size = ErrorModel.random_insert_size()

    # generate the forward read
    try:  # a ref sequence has to be longer than 2 * read_length + i_size
        assert read_length < len(record.seq)
        # assign the start position of the forward read
        # if sequence_type == metagenomics, get a random start position
        # if sequence_type == amplicon, start position is the start of the read
        if sequence_type == 'metagenomics':
            forward_start = random.randrange(
                0, len(record.seq) - (2 * read_length + insert_size))
        elif sequence_type == 'amplicon':
            forward_start = 0
        else:
            raise RuntimeError(f"sequence type '{sequence_type}' is not supported")
    except AssertionError as e:
        raise
    except ValueError as e:
        LOGGER.debug(
            '%s shorter than template length for this ErrorModel:%s'
            % (record.id, e))
        forward_start = max(0, random.randrange(
            0, len(record.seq) - read_length))

    forward_end = forward_start + read_length
    bounds = (forward_start, forward_end)
    # create a perfect read
    forward = SeqRecord(
        Seq(str(sequence[forward_start:forward_end])),
        id='%s_%s_%s/1' % (header, i, cpu_number),
        description=''
    )

    forward.annotations["mutations"] = []
    forward.annotations["original"] = str(forward.seq)
    # add the indels, the qual scores and modify the record accordingly
    forward = ErrorModel.introduce_indels(forward, 'forward', sequence, bounds)
    forward = ErrorModel.introduce_error_scores(forward, 'forward')
    forward = ErrorModel.mut_sequence(forward, 'forward')

    # generate the reverseread
    # assign start position reverse read
    # if sequence_type == metagenomics, get a start position based on insert_size
    # if sequence_type == amplicon, start position is the end of the read
    if sequence_type == "metagenomics":
        reverse_start = forward_end + insert_size
        reverse_end = reverse_start + read_length
    elif sequence_type == "amplicon":
        reverse_start = len(record.seq) - read_length
        reverse_end = reverse_start + read_length
    else:
        raise ValueError(f"Sequence type {sequence_type} not known")
    if reverse_end > len(record.seq):
        # we use random insert when the modelled template length distribution
        # is too large
        reverse_end = random.randrange(read_length, len(record.seq))
        reverse_start = reverse_end - read_length
    bounds = (reverse_start, reverse_end)
    # create a perfect read
    reverse = SeqRecord(
        Seq(rev_comp(str(sequence[reverse_start:reverse_end]))),
        id='%s_%s_%s/2' % (header, i, cpu_number),
        description=''
    )

    reverse.annotations["mutations"] = []
    reverse.annotations["original"] = str(reverse.seq)
    # add the indels, the qual scores and modify the record accordingly
    reverse = ErrorModel.introduce_indels(reverse, 'reverse', sequence, bounds)
    reverse = ErrorModel.introduce_error_scores(reverse, 'reverse')
    reverse = ErrorModel.mut_sequence(reverse, 'reverse')

    return (forward, reverse)


def to_fastq(generator, output):
    """Write reads to a fastq file

    Take a generator or a list containing read pairs (tuples) and write them
        in two fastq files: output_R1.fastq and output_R2.fastq

    Args:
        generator (generator): a read generator (or list)
        output (string): the output files prefix
    """
    # define name of output files
    output_forward = output + '_R1.fastq'
    output_reverse = output + '_R2.fastq'

    try:
        f = open(output_forward, 'a')
        r = open(output_reverse, 'a')
    except PermissionError as e:
        LOGGER.error('Failed to open output file(s): %s' % e)
        sys.exit(1)
    else:
        with f, r:
            for read_tuple in generator:
                SeqIO.write(read_tuple[0], f, 'fastq-sanger')
                SeqIO.write(read_tuple[1], r, 'fastq-sanger')


def to_vcf(generator, output):
    output_vcf = output + ".vcf"
    with open(output_vcf, 'w') as f:
        for vcf_dict in generator:
            # pass
            line = "\t".join([
                str(vcf_dict["id"]),
                str(vcf_dict["position"] + 1), # vcf files have 1-based index
                ".",
                vcf_dict["ref"],
                str(vcf_dict["alt"]),
                str(vcf_dict["quality"]),
                "",
                ""
            ])
            f.write(line + "\n")
