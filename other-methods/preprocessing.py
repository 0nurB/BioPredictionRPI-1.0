#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import re
import argparse
warnings.filterwarnings("ignore")
from Bio import SeqIO


def preprocessing(finput, foutput):
    alphabet = ("B|J|O|X|Z")
    replacement_char = 'A'  # Choose a replacement character outside of the specified alphabet
    file = open(foutput, 'a')
    
    for seq_record in SeqIO.parse(finput, "fasta"):
        name_seq = seq_record.name
        seq = seq_record.seq

        # Replace characters in the sequence that match the specified alphabet
        corrected_seq = re.sub(alphabet, replacement_char, str(seq))

        #if corrected_seq != str(seq):
        #    print(name_seq)
        #    print("Corrected Sequence")
        #else:
        file.write(">%s" % (str(name_seq)))
        file.write("\n")
        file.write(corrected_seq)
        file.write("\n")




#############################################################################    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Fasta format file, E.g., dataset.fasta')
    parser.add_argument('-o', '--output', help='Fasta format file, E.g., preprocessing.fasta')
    args = parser.parse_args()
    finput = str(args.input)
    foutput = str(args.output)
    preprocessing(finput,foutput)
#############################################################################]
