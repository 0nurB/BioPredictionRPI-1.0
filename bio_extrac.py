import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
import networkx as nx

import torch
import statistics
from Bio import SeqIO
import argparse
import subprocess
import sys
import os.path
from subprocess import Popen
from multiprocessing import Manager
from sklearn.exceptions import FitFailedWarning

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path + '/repDNA/repDNA/')
from nac import *
from psenac import *
from ac import *


# Define the reduced alphabet mapping
reduced_alphabet = {
    'A': 0, 'G': 0, 'V': 0,
    'I': 1, 'L': 1, 'F': 1, 'P': 1,
    'Y': 2, 'M': 2, 'T': 2, 'S': 2,
    'H': 3, 'N': 3, 'Q': 3, 'W': 3,
    'R': 4, 'K': 4,
    'D': 5, 'E': 5,
    'C': 6
}

# Function to encode a protein sequence
def encode_protein_sequence(sequence):
    encoded_sequence = np.zeros((7, 7, 7))
    for i in range(len(sequence) - 2):
        triad = sequence[i:i+3]
        if all(aa in reduced_alphabet for aa in triad):
            idx1, idx2, idx3 = [reduced_alphabet[aa] for aa in triad]
            encoded_sequence[idx1, idx2, idx3] += 1

    # Normalize the frequencies
    total_triads = len(sequence) - 2
    encoded_sequence /= total_triads

    # Flatten the 3D matrix into a 1D vector
    return encoded_sequence.flatten()

# Read FASTA file

def group_feats(fasta_file_path, output):
    sequences = []
    for record in SeqIO.parse(fasta_file_path, 'fasta'):
        sequences.append(str(record.seq))

    # Encode each protein sequence
    encoded_sequences = [encode_protein_sequence(seq) for seq in sequences]

    # Create a DataFrame from the encoded sequences
    columns = [f'feature_{i}' for i in range(343)]
    df = pd.DataFrame(encoded_sequences, columns=columns)

    # Add a column for sequence IDs (assuming the FASTA headers contain sequence IDs)
    df['nameseq'] = [record.id for record in SeqIO.parse(fasta_file_path, 'fasta')]

    # Reorder columns with 'nameseq' as the first column
    df = df[['nameseq'] + columns]

    # Save the DataFrame to a CSV file
    csv_file_path = output + '/fq_grups.csv'
    df.to_csv(csv_file_path, index=False)

    #print(f'Data saved to {csv_file_path}')

amino_mappings = {
    "H1": {"A": 0.62, "C": 0.29, "D": -0.9, "E": -0.74, "F": 1.19, "G": 0.48, "H": -0.4, "I": 1.38,
           "K": -1.5, "L": 1.06, "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.53, "S": -0.18,
           "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26},
    "H2": {"A": -0.5, "C": -1, "D": 3, "E": 3, "F": -2.5, "G": 0, "H": -0.5, "I": -1.8,
           "K": 3, "L": -1.8, "M": -1.3, "N": 2, "P": 0, "Q": 0.2, "R": 3, "S": 0.3,
           "T": -0.4, "V": -1.5, "W": -3.4, "Y": -2.3},
    "V": {"A": 27.5, "C": 44.6, "D": 40, "E": 62, "F": 115.5, "G": 0, "H": 79, "I": 93.5,
          "K": 100, "L": 93.5, "M": 94.1, "N": 58.7, "P": 41.9, "Q": 80.7, "R": 105, "S": 29.3,
          "T": 51.3, "V": 71.5, "W": 145.5, "Y": 117.3},
    "P1": {"A": 8.1, "C": 5.5, "D": 13, "E": 12.3, "F": 5.2, "G": 9, "H": 10.4, "I": 5.2,
           "K": 11.3, "L": 4.9, "M": 5.7, "N": 11.6, "P": 8, "Q": 10.5, "R": 10.5, "S": 9.2,
           "T": 8.6, "V": 5.9, "W": 5.4, "Y": 6.2},
    "P2": {"A": 0.046, "C": 0.128, "D": 0.105, "E": 0.151, "F": 0.29, "G": 0, "H": 0.23, "I": 0.186,
           "K": 0.219, "L": 0.186, "M": 0.221, "N": 0.134, "P": 0.131, "Q": 0.18, "R": 0.291, "S": 0.062,
           "T": 0.108, "V": 0.14, "W": 0.409, "Y": 0.298},
    "SASA": {"A": 1.181, "C": 1.461, "D": 1.587, "E": 1.862, "F": 2.228, "G": 0.881, "H": 2.025, "I": 1.81,
             "K": 2.258, "L": 1.931, "M": 2.034, "N": 1.655, "P": 1.468, "Q": 1.932, "R": 2.56, "S": 1.298,
             "T": 1.525, "V": 1.645, "W": 2.663, "Y": 2.368},
    "NCI": {"A": 0.007187, "C": -0.03661, "D": -0.02382, "E": 0.006802, "F": 0.037552, "G": 0.179052,
            "H": -0.01069, "I": 0.021631, "K": 0.017708, "L": 0.051672, "M": 0.002683, "N": 0.005392,
            "P": 0.239531, "Q": 0.049211, "R": 0.043587, "S": 0.004627, "T": 0.003352, "V": 0.057004,
            "W": 0.037977, "Y": 0.023599}
}

def fasta_to_dataframe(fasta_file):
    records = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    data = {'Nome da Sequência': list(records.keys()), 'Sequência': [str(record.seq) for record in records.values()]}
    df = pd.DataFrame(data) 
    return df

def prop_qf(data, amino_mappings, path):

    # Criar uma lista para armazenar os dados
    data_list = []

    # Iterar sobre as linhas do DataFrame
    for mapping_name, mapping_values in amino_mappings.items():
        result_df = pd.DataFrame()

        for index, row in data.iterrows():
            name = row[data.columns[0]]
            sequence = row[data.columns[1]]

            sequence_features = []

            numeric_sequence = np.array([mapping_values.get(amino, 0) for amino in sequence])
            fft_result = np.fft.fft(numeric_sequence)

            complex_array = fft_result[:200]

            modulos = np.abs(complex_array)

            num_colunas = len(modulos)  # Obtém o número real de colunas
            df = pd.DataFrame([modulos], columns=[f'{mapping_name}_{i+1}' for i in range(num_colunas)], index=[name])

            # Adicionar o DataFrame ao DataFrame resultante
            result_df = pd.concat([result_df, df])
        result_df = result_df.dropna(axis=1)
        result_df.index.name = 'nameseq'  # Define o nome do índice
        result_df.reset_index(inplace=True)  # Move o índice para uma coluna
        output = os.path.join(path, 'feat_'+mapping_name+'.csv')
        result_df.to_csv(output, index=False)

def prop_qf_mean(data, amino_mappings, path):
    # Criar uma lista para armazenar os dados
    data_list = []

    # Iterar sobre as linhas do DataFrame
    for index, row in data.iterrows():
        name = row[data.columns[0]]
        sequence = row[data.columns[1]]

        sequence_features = []

        for mapping_name, mapping_values in amino_mappings.items():
            numeric_sequence = np.array([mapping_values.get(amino, 0) for amino in sequence])
            fft_result = np.fft.fft(numeric_sequence)
            
            spectrum = np.abs(fft_result) ** 2
            spectrumTwo = np.abs(fft_result)

            # Calcular as características do espectro
            average = np.mean(spectrum)
            median = np.median(spectrum)
            maximum = np.max(spectrum)
            minimum = np.min(spectrum)

            peak = (len(spectrum) / 3) / average
            peak_two = (len(spectrumTwo) / 3) / np.mean(spectrumTwo)
            standard_deviation = np.std(spectrum)
            standard_deviation_pop = statistics.stdev(spectrum)
            percentile15 = np.percentile(spectrum, 15)
            percentile25 = np.percentile(spectrum, 25)
            percentile50 = np.percentile(spectrum, 50)
            percentile75 = np.percentile(spectrum, 75)
            amplitude = maximum - minimum
            variance = statistics.variance(spectrum)
            interquartile_range = np.percentile(spectrum, 75) - np.percentile(spectrum, 25)
            semi_interquartile_range = (np.percentile(spectrum, 75) - np.percentile(spectrum, 25)) / 2
            coefficient_of_variation = standard_deviation / average
            skewness = (3 * (average - median)) / standard_deviation
            kurtosis = (np.percentile(spectrum, 75) - np.percentile(spectrum, 25)) / (2 * (np.percentile(spectrum, 90) - np.percentile(spectrum, 10)))

            # Adicionar as características à lista sequence_features
            sequence_features.extend([
                average, median, maximum, minimum, peak, peak_two, standard_deviation, 
                standard_deviation_pop, percentile15, percentile25, percentile50, 
                percentile75, amplitude, variance, interquartile_range, semi_interquartile_range, 
                coefficient_of_variation, skewness, kurtosis
            ])

        # Adicionar os dados à lista
        data_list.append([name] + sequence_features)

    # Criar os nomes das colunas
    columns = ["nameseq"]
    for mapping_name in amino_mappings.keys():
        features_list = [
            "average", "median", "maximum", "minimum", "peak", "peak_two", "standard_deviation", 
            "standard_deviation_pop", "percentile15", "percentile25", "percentile50", 
            "percentile75", "amplitude", "variance", "interquartile_range", "semi_interquartile_range", 
            "coefficient_of_variation", "skewness", "kurtosis"
        ]
        columns.extend([f"{mapping_name}_{feature}" for feature in features_list])

    df = pd.DataFrame(data_list, columns=columns)
    output = os.path.join(path, 'Mean_feat.csv')
    df.to_csv(output, index=False)

def extrac_math_features(features_amino, sequences, stype, path):
    """
    Extract features from amino acid-based sequences.

    Parameters:
    - features_amino (list): List of feature extraction options.
    - sequences (str): Input sequences in FASTA format.
    - path (str): Output directory path.

    Returns:
    - datasets_extr (list): List of paths to extracted datasets.
    - names_math (list): List of feature names.
    """
    features_nucleot = features_amino
    fasta = [sequences]   
    datasets_extr = []
    names_math = []
    commands = []

    print(f'Extracting features with MathFeature...')

    if stype == 0:
        #features_nucleot = [1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18]
        """Feature extraction for nucleotide-based sequences """    
        for i in range(len(fasta)):
            file = fasta[i].split('/')[-1]
            if i == 0:  # Train
                preprocessed_fasta = os.path.join(path + '/pre_' + file)
                subprocess.run(['python', 'other-methods/preprocessing.py',
                                '-i', fasta[i], '-o', preprocessed_fasta],
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                #sequencias_unicas = {record.seq: record for record in SeqIO.parse(preprocessed_fasta, "fasta")}
                #SeqIO.write(sequencias_unicas.values(), preprocessed_fasta, "fasta")

                if 1 in features_nucleot:
                    
                    dataset = os.path.join(path, 'NAC_dna' + '.csv')
                    datasets_extr.append(dataset)
                    names_math.append('NAC_dna')
                    commands.append(['python', 'MathFeature/methods/ExtractionTechniques.py',
                                    '-i', preprocessed_fasta, '-o', dataset, '-l', 'DNA',
                                    '-t', 'NAC', '-seq', '1'])
                
                if 2 in features_nucleot:
                    dataset = os.path.join(path, 'DNC_dna' + '.csv')
                    datasets_extr.append(dataset)
                    names_math.append('DNC_dna')
                    commands.append(['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
                                    preprocessed_fasta, '-o', dataset, '-l', 'DNA',
                                    '-t', 'DNC', '-seq', '1'])

                if 3 in features_nucleot:
                    dataset = os.path.join(path, 'TNC_dna' + '.csv')
                    datasets_extr.append(dataset)
                    names_math.append('TNC_dna')
                    commands.append(['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
                                    preprocessed_fasta, '-o', dataset, '-l', 'DNA',
                                    '-t', 'TNC', '-seq', '1'])
                    
                if 25 in features_nucleot:
                    dataset = os.path.join(path, 'QNC_dna' + '.csv')
                    datasets_extr.append(dataset)
                    names_math.append('QNC_dna')
                    commands.append(['python', 'repDNA/ExtractionTechniques.py', '-i',
                                    preprocessed_fasta, '-o', dataset, '-l', 'DNA',
                                    '-t', 'QNC', '-seq', '1'])

                if '4' in features_nucleot:
                    dataset_di = os.path.join(path, 'kGap_di' + '.csv')
                    dataset_tri = os.path.join(path, 'kGap_tri' +'.csv')
                    
                    datasets_extr.append(dataset_di)
                    names_math.append('kGap_di')
                    datasets_extr.append(dataset_tri)
                    names_math.append('kGap_tri')

                    commands.append(['python', 'MathFeature/methods/Kgap.py', '-i',
                                    preprocessed_fasta, '-o', dataset_di, '-l',
                                    'DNA', '-k', '1', '-bef', '1',
                                    '-aft', '2', '-seq', '1'])

                    commands.append(['python', 'MathFeature/methods/Kgap.py', '-i',
                                    preprocessed_fasta, '-o', dataset_tri, '-l',
                                    'DNA', '-k', '1', '-bef', '1',
                                    '-aft', '3', '-seq', '1'])

                if '5' in features_nucleot:
                    dataset = os.path.join(path, 'ORF_dna' + '.csv')
                    datasets_extr.append(dataset)
                    names_math.append('ORF_dna')
                    commands.append(['python', 'MathFeature/methods/CodingClass.py', '-i',
                                    preprocessed_fasta, '-o', dataset, '-l', 'DNA'])

                if '6' in features_nucleot:
                    dataset = os.path.join(path, 'Fickett_dna' + '.csv')
                    datasets_extr.append(dataset)
                    names_math.append('Fickett_dna')
                    commands.append(['python', 'MathFeature/methods/FickettScore.py', '-i',
                                    preprocessed_fasta, '-o', dataset, '-l', 'DNA',
                                    '-seq', '1'])
        
                if 7 in features_nucleot:
                    dataset = os.path.join(path, 'Shannon_DNA' + '.csv')
                    datasets_extr.append(dataset)
                    names_math.append('Shannon_dna')
                    commands.append(['python', 'MathFeature/methods/EntropyClass.py', '-i',
                                    preprocessed_fasta, '-o', dataset, '-l', 'DNA',
                                    '-k', '5', '-e', 'Shannon'])

                if '8' in features_nucleot:
                    dataset =os.path.join(self.path, 'FourierBinary_' + self.ftype + '.csv')
                    subprocess.run(['python', 'MathFeature/methods/FourierClass.py', '-i',
                                    preprocessed_fasta, '-o', dataset, '-l', 'DNA',
                                    '-r', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

                if 9 in features_nucleot:
                    dataset = os.path.join(path, 'FourierComplex_dna' + '.csv')
                    datasets_extr.append(dataset)
                    names_math.append('FourierComplex_dna')
                    commands.append(['python', 'other-methods/FourierClass.py', '-i',
                                    preprocessed_fasta, '-o', dataset, '-l', 'DNA',
                                    '-r', '6'])

                if '10' in features_nucleot:
                    dataset = os.path.join(path, 'Tsallis_dna' + '.csv')
                    datasets_extr.append(dataset)
                    names_math.append('Tsallis_dna')
                    subprocess.run(['python', 'other-methods/TsallisEntropy.py', '-i',
                                    preprocessed_fasta, '-o', dataset, '-l', 'DNA',
                                    '-k', '5', '-q', '2.3'])

                if '12' in features_nucleot:
                    #dataset = path + '/BinaryMapping.csv'
                    dataset = os.path.join(self.path, 'BinaryMapping_' + self.ftype + '.csv')
                    #labels_list = ftrain_labels + ftest_labels
                    labels_list = 'DNA'
                    text_input = ''
                    
                    text_input += preprocessed_fasta + '\n' + labels_list + '\n'

                    subprocess.run(['python', 'MathFeature/methods/MappingClass.py',
                                '-n', str(len(preprocessed_fasta)), '-o',
                                dataset, '-r', '1'], text=True, input=text_input,
                               stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

                    with open(dataset, 'r') as temp_f:
                        col_count = [len(l.split(",")) for l in temp_f.readlines()]

                    colnames = ['BinaryMapping_' + str(i) for i in range(0, max(col_count))]

                    df = pd.read_csv(dataset, names=colnames, header=None)
                    df.rename(columns={df.columns[0]: 'nameseq', df.columns[-1]: 'label'}, inplace=True)
                    df.to_csv(dataset, index=False)

                    datasets.append(dataset)

            if 13 in features_nucleot:   
                dataset = os.path.join(path, 'Rev_kmer_dna' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('Rev_kmer_dna')
                                    
                rev_kmer = RevcKmer(k=3, normalize=True, upto=True)
                data_kmer = rev_kmer.make_revckmer_vec(open(preprocessed_fasta))
                data_kmer = pd.DataFrame(data_kmer)
                data_kmer = data_kmer.add_prefix('Rev_kmer_')
                data_kmer.to_csv(dataset, index=False)

            if 14 in features_nucleot: 
                dataset = os.path.join(path, 'Pse_dnc_dna' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('Pse_dnc_dna')
                                    
                psednc = PseDNC()
                data_psednc = psednc.make_psednc_vec(open(preprocessed_fasta))
                data_psednc = pd.DataFrame(data_psednc)
                data_psednc = data_psednc.add_prefix('Pse_dnc_')
                data_psednc.to_csv(dataset, index=False)

            if 15 in features_nucleot:
                dataset = os.path.join(path, 'Pse_knc_dna' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('Pse_knc_dna')
                                    
                pseknc = PseKNC()
                data_pseknc = pseknc.make_pseknc_vec(open(preprocessed_fasta))
                data_pseknc = pd.DataFrame(data_pseknc)
                data_pseknc = data_pseknc.add_prefix('Pse_knc_')
                data_pseknc.to_csv(dataset, index=False)

            if 16 in features_nucleot:
                dataset = os.path.join(path, 'SCPseDNC_dna' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('SCPseDNC_dna')                   
                                    
                sc_psednc = SCPseDNC()
                data_sc_psednc = sc_psednc.make_scpsednc_vec(open(preprocessed_fasta), all_property=True)
                data_sc_psednc = pd.DataFrame(data_sc_psednc)
                data_sc_psednc = data_sc_psednc.add_prefix('SCPseDNC_')
                data_sc_psednc.to_csv(dataset, index=False)

            if 17 in features_nucleot:
                dataset = os.path.join(path, 'SCPseTNC_dna' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('SCPseTNC_dna')                                         
                                   
                sc_psetnc = SCPseTNC(lamada=2, w=0.05)
                data_sc_psetnc = sc_psetnc.make_scpsetnc_vec(open(preprocessed_fasta), all_property=True)
                data_sc_psetnc = pd.DataFrame(data_sc_psetnc)
                data_sc_psetnc = data_sc_psetnc.add_prefix('SCPseTNC_')
                data_sc_psetnc.to_csv(dataset, index=False)

            if 18 in features_nucleot:
                dataset = os.path.join(path, 'DAC_dna' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('DAC_dna') 
                
                dac = DAC(2)
                data_dac = dac.make_dac_vec(open(preprocessed_fasta), all_property=True)
                data_dac = pd.DataFrame(data_dac)
                data_dac = data_dac.add_prefix('DAC_')
                data_dac.to_csv(dataset, index=False)
    
    if stype == 1:
        """Feature extraction for aminoacids-based sequences"""   
        for i in range(len(fasta)):
            file = fasta[i].split('/')[-1]
            if i == 0:  # Train
                preprocessed_fasta = os.path.join(path + '/pre_' + file)
                subprocess.run(['python', 'other-methods/preprocessing.py',
                                '-i', fasta[i], '-o', preprocessed_fasta],
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                #sequencias_unicas = {record.seq: record for record in SeqIO.parse(preprocessed_fasta, "fasta")}
                #SeqIO.write(sequencias_unicas.values(), preprocessed_fasta, "fasta")

            if 1 in features_amino:
                dataset = os.path.join(path, 'Shannon_protein' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('Shannon_protein')
                commands.append(['python', 'MathFeature/methods/EntropyClass.py',
                                '-i', preprocessed_fasta, '-o', dataset, '-l', 'protein',
                                '-k', '5', '-e', 'Shannon'])

            if 2 in features_amino:
                dataset = os.path.join(path, 'Tsallis_23_protein'  + '.csv')
                datasets_extr.append(dataset)
                names_math.append('Tsallis_23_protein')
                commands.append(['python', 'other-methods/TsallisEntropy.py',
                                '-i', preprocessed_fasta, '-o', dataset, '-l', 'protein',
                                '-k', '5', '-q', '2.3'])

            if 3 in features_amino:
                dataset = os.path.join(path, 'Tsallis_30_protein'  + '.csv')
                datasets_extr.append(dataset)
                names_math.append('Tsallis_30_protein')
                commands.append(['python', 'other-methods/TsallisEntropy.py',
                                '-i', preprocessed_fasta, '-o', dataset, '-l', 'protein',
                                '-k', '5', '-q', '3.0'])

            if 4 in features_amino:
                dataset = os.path.join(path, 'Tsallis_40_protein'  + '.csv')
                datasets_extr.append(dataset)
                names_math.append('Tsallis_40_protein')
                commands.append(['python', 'other-methods/TsallisEntropy.py',
                                '-i', preprocessed_fasta, '-o', dataset, '-l', 'protein',
                                '-k', '5', '-q', '4.0'])

            if 5 in features_amino:
                dataset = os.path.join(path, 'ComplexNetworks_protein' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('ComplexNetworks_protein')
                commands.append(['python', 'MathFeature/methods/ComplexNetworksClass-v2.py', '-i',
                                preprocessed_fasta, '-o', dataset, '-l', 'protein',
                                '-k', '3'])

            if 6 in features_amino:
                dataset_di = os.path.join(path, 'kGap_di_protein' + '.csv')
                datasets_extr.append(dataset_di)
                names_math.append('kGap_di_protein')
                commands.append(['python', 'MathFeature/methods/Kgap.py', '-i',
                                preprocessed_fasta, '-o', dataset_di, '-l',
                                'protein', '-k', '1', '-bef', '1',
                                '-aft', '1', '-seq', '3'])

            if 7 in features_amino:
                dataset = os.path.join(path, 'AAC_protein' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('AAC')
                commands.append(['python', 'MathFeature/methods/ExtractionTechniques-Protein.py', '-i',
                                preprocessed_fasta, '-o', dataset, '-l', 'protein','-t', 'AAC'])

            if 8 in features_amino:
                dataset = os.path.join(path, 'DPC_protein' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('DPC')
                commands.append(['python', 'MathFeature/methods/ExtractionTechniques-Protein.py', '-i',
                                preprocessed_fasta, '-o', dataset, '-l', 'protein', '-t', 'DPC'])

            if '9' in features_amino:
                dataset = os.path.join(path, 'TPC' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('TPC')
                commands.append(['python', 'MathFeature/methods/ExtractionTechniques-Protein.py', '-i',
                                preprocessed_fasta, '-o', dataset, '-l', 'protein', '-t', 'TPC'])

            if '10' in features_amino:
                dataset = os.path.join(path, 'iFeature' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('iFeature')
                commands.append(['python', 'other-methods/iFeature-modified/iFeature.py', '--file', 
                                 preprocessed_fasta, '--type', 'All', '--label', 'protein', 
                                 '--out', dataset])

            if '11' in features_amino:
                dataset = os.path.join(path, 'Global' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('Global')
                commands.append(['python', 'other-methods/modlAMP-modified/descriptors.py', '-option',
                                 'peptide', '-label', 'protein', '-input', preprocessed_fasta, 
                                 '-output', dataset])

            if '12' in features_amino:
                dataset = os.path.join(path, 'Peptide' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('Peptide')
                commands.append(['python', 'other-methods/modlAMP-modified/descriptors.py',
                                 '-option','peptide', '-label', 'protein', '-input', 
                                 preprocessed_fasta, '-output', dataset])   
                
            if 13 in features_amino:
                dataset = os.path.join(path, 'Mean_feat' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('Mean_feat')
                df = fasta_to_dataframe(preprocessed_fasta)
                prop_qf(df,amino_mappings, path)
                
            if 14 in features_amino:
                dataset = os.path.join(path, 'feat_H1' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('feat_H1')
                dataset = os.path.join(path, 'feat_H2' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('feat_H2')
                dataset = os.path.join(path, 'feat_NCI' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('feat_NCI')
                dataset = os.path.join(path, 'feat_P1' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('feat_P1')
                dataset = os.path.join(path, 'feat_P2' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('feat_P2')
                dataset = os.path.join(path, 'feat_SASA' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('feat_SASA')
                dataset = os.path.join(path, 'feat_V' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('feat_V')
                df = fasta_to_dataframe(preprocessed_fasta)
                prop_qf_mean(df, amino_mappings, path)
             
            if 25 in features_amino:            
                dataset = os.path.join(path, 'fq_grups' + '.csv')
                datasets_extr.append(dataset)
                names_math.append('fq_grups')
                group_feats(preprocessed_fasta, path)

                


        """Concatenating all the extracted features"""
    processes = [Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) for cmd in commands]
    for p in processes: p.wait()   
    return datasets_extr, names_math

def graph_ext(G):
    """
    Extract various centrality and graph measures from a given graph.

    Parameters:
        G (networkx.Graph): Input graph.

    Returns:
        data_nodes (dict): Dictionary containing extracted centrality and graph measures.
    """
    hits_scores = nx.hits(G)
    hubs = hits_scores[0]
    authorities = hits_scores[1]
    adj_matrix = nx.adjacency_matrix(G).toarray()
    spectrum = np.linalg.eigvals(adj_matrix)
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    #eigenvector_centrality = nx.eigenvector_centrality(G)
    pagerank = nx.pagerank(G)
    harmonic_centrality = nx.harmonic_centrality(G)
    personalized_pagerank = nx.pagerank(G, alpha=0.5)
    #directed_eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight', max_iter=10)
    load_centrality = nx.load_centrality(G)
    closeness_in_closeness_centrality = nx.closeness_centrality(G, wf_improved=False)
    adjusted_closeness_centrality = nx.closeness_centrality(G, wf_improved=False, distance="adjusted")
    weighted_closeness_centrality = nx.closeness_centrality(G, wf_improved=False, distance="weighted")
    edge_betweenness = nx.edge_betweenness_centrality(G)
    load_centrality = nx.load_centrality(G)
    #subgraph_centrality = nx.subgraph_centrality(G)
    #information_centrality = nx.information_centrality(G)
    #directed_eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight', max_iter=10)
    #current_flow_betweenness = nx.current_flow_betweenness_centrality(G)
    #current_flow_closeness = nx.current_flow_closeness_centrality(G)
    closeness_in_closeness_centrality = nx.closeness_centrality(G, wf_improved=False)
    adjusted_closeness_centrality = nx.closeness_centrality(G, wf_improved=False, distance="adjusted")
    weighted_closeness_centrality = nx.closeness_centrality(G, wf_improved=False, distance="weighted")
    #eccentricity_centrality = nx.eccentricity(G)
    harmonic_centrality = nx.harmonic_centrality(G)
    average_neighbor_degree_centrality = nx.average_neighbor_degree(G)
    degree_assortativity_centrality = nx.degree_assortativity_coefficient(G)
    clustering_centrality = nx.clustering(G)
    core_number_centrality = nx.core_number(G)

    data_nodes = {
        'Node': list(G.nodes()),
        'Hub Score': [hubs[node] for node in G.nodes()],
        'Authority Score': [authorities[node] for node in G.nodes()],
        'Spectrum': spectrum,
        'Degree Centrality': [degree_centrality[node] for node in G.nodes()],
        'Betweenness Centrality': [betweenness_centrality[node] for node in G.nodes()],
        'Closeness Centrality': [closeness_centrality[node] for node in G.nodes()],
        #'Eigenvector Centrality': [eigenvector_centrality[node] for node in G.nodes()],
        'PageRank': [pagerank[node] for node in G.nodes()],
        'Harmonic Centrality': [harmonic_centrality[node] for node in G.nodes()],
        'Personalized PageRank (Alpha=0.5)': [personalized_pagerank[node] for node in G.nodes()],
        'Closeness in Closeness Centrality': [closeness_in_closeness_centrality[node] for node in G.nodes()],
        'Adjusted Closeness Centrality': [adjusted_closeness_centrality[node] for node in G.nodes()],
        'Weighted Closeness Centrality': [weighted_closeness_centrality[node] for node in G.nodes()],
        #'Eccentricity Centrality': [eccentricity_centrality[node] for node in G.nodes()],
        'Average Neighbor Degree Centrality': [average_neighbor_degree_centrality[node] for node in G.nodes()],
        'Clustering Centrality': [clustering_centrality[node] for node in G.nodes()],
        'Core Number Centrality': [core_number_centrality[node] for node in G.nodes()],
        #'Information Centrality': [information_centrality[node] for node in G.nodes()],
        #'Directed Eigenvector Centrality': [directed_eigenvector_centrality[node] for node in G.nodes()],
        #'Current Flow Betweenness': [current_flow_betweenness[node] for node in G.nodes()],
        #'Closeness in Closeness Centrality': [closeness_in_closeness_centrality[node] for node in G.nodes()],
        'Adjusted Closeness Centrality': [adjusted_closeness_centrality[node] for node in G.nodes()],
        'Weighted Closeness Centrality': [weighted_closeness_centrality[node] for node in G.nodes()],
        #'Eccentricity Centrality': [eccentricity_centrality[node] for node in G.nodes()],
        'Harmonic Centrality': [harmonic_centrality[node] for node in G.nodes()],
        'Clustering Centrality': [clustering_centrality[node] for node in G.nodes()],
        'Core Number Centrality': [core_number_centrality[node] for node in G.nodes()],
        #'Subgraph Centrality': [subgraph_centrality[node] for node in G.nodes()]
    }
    return data_nodes





def make_graph(edges, node):
    """
    Create a graph from edges data and nodes data.

    Parameters:
        edges (pd.DataFrame): DataFrame containing edge information.
        node (pd.DataFrame): DataFrame containing node information.

    Returns:
        G (nx.Graph): NetworkX Graph representing the connections between nodes.
    """
    # Extract unique protein names from node data
    columns_graph = edges.columns
    PA = node[columns_graph[0]].unique()
    PB = node[columns_graph[1]].unique()
    
    # Create an empty undirected graph
    G = nx.Graph()
    G.add_nodes_from(PA)
    G.add_nodes_from(PB)
    
    # Add edges to the graph based on edge data
    for index, row in edges.iterrows():
        if row[columns_graph[2]] == 1:
            protein_a = row[columns_graph[0]]
            protein_b = row[columns_graph[1]]
            G.add_edge(protein_a, protein_b)
    
    # Remove self-loops from the graph
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    
    return G

def make_graph_N(edges, node):
    """
    Create a graph from edges data and nodes data.

    Parameters:
        edges (pd.DataFrame): DataFrame containing edge information.
        node (pd.DataFrame): DataFrame containing node information.

    Returns:
        G (nx.Graph): NetworkX Graph representing the connections between nodes.
    """
    # Extract unique protein names from node data
    columns_graph = edges.columns
    PA = node[columns_graph[0]].unique()
    PB = node[columns_graph[1]].unique()
    
    # Create an empty undirected graph
    G = nx.Graph()
    G.add_nodes_from(PA)
    G.add_nodes_from(PB)
    
    # Add edges to the graph based on edge data
    for index, row in edges.iterrows():
        if row[columns_graph[2]] == 0:
            protein_a = row[columns_graph[0]]
            protein_b = row[columns_graph[1]]
            G.add_edge(protein_a, protein_b)
    
    # Remove self-loops from the graph
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    
    return G



def make_graph_all(edges, node):
    """
    Create a graph from edges data with both positive and negative edges and nodes data.

    Parameters:
        edges (pd.DataFrame): DataFrame containing edge information.
        node (pd.DataFrame): DataFrame containing node information.

    Returns:
        G (nx.Graph): NetworkX Graph representing the connections between nodes.
    """
    # Extract unique protein names from node data
    columns_graph = edges.columns
    PA = node[columns_graph[0]].unique()
    PB = node[columns_graph[1]].unique()
    
    # Create an empty undirected graph
    G = nx.Graph()
    G.add_nodes_from(PA)
    G.add_nodes_from(PB)
    
    # Add edges to the graph based on edge data (including positive and negative edges)
    for index, row in edges.iterrows():
        if row[columns_graph[2]] == 2 or row[columns_graph[2]] == 0:
            protein_a = row[columns_graph[0]]
            protein_b = row[columns_graph[1]]
            G.add_edge(protein_a, protein_b)
    
    # Remove self-loops from the graph
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    
    return G


def make_graph_bi(edges, node, label=3):

    columns_graph = edges.columns
    PA = node[columns_graph[0]].unique().tolist()
    PB = node[columns_graph[1]].unique().tolist()

    # Create an empty undirected graph
    G = nx.Graph()

    G.add_nodes_from(PA, bipartite=0)
    G.add_nodes_from(PB, bipartite=1)



    # Adicionando nós e arestas ao grafo
    for _, row in edges.iterrows():
        if row['Label'] == label or label == 3:        
            G.add_node(row['ProteinA'], bipartite=0)
            G.add_node(row['ProteinB'], bipartite=1)
            G.add_edge(row['ProteinA'], row['ProteinB'])

    # Dividindo os conjuntos de nós
    proteina_nodes = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
    rna_nodes = set(G) - proteina_nodes

    # Criando uma layout bipartida
    pos = {node: (1, i) for i, node in enumerate(proteina_nodes)}
    pos.update({node: (2, i) for i, node in enumerate(rna_nodes)})

    return G







