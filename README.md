![Python](https://img.shields.io/badge/python-v3.7-blue)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Status](https://img.shields.io/badge/status-up-brightgreen)

<h1 align="center">
  <img src="https://github.com/0nurB/BioPredictionRPI-1.0/blob/main/assets/logoBio.png" alt="BioPrediction" width="500">
</h1>

<h4 align="center">BioPrediction: Democratizing Machine Learning in the Study of Molecular Interactions</h4>

<h4 align="center">Democratizing Machine Learning in Life Sciences</h4>

<p align="center">
  <a href="https://github.com/Bonidia/BioPrediction">Home</a> •
  <a href="http://autoaipandemics.icmc.usp.br">AutoAI-Pandemics</a> •
  <a href="#installing-dependencies-and-package">Installing</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citation">Citation</a> 
</p>

<h1 align="center"></h1>


## Main

**Published paper:** Florentino, B. R., Parmezan Bonidia, R., Sanches, N. H., Da Rocha, U. N., & De Carvalho, A. C. (2024). BioPrediction-RPI: Democratizing the prediction of interaction between non-coding RNA and protein with end-to-end machine learning. Computational and Structural Biotechnology Journal, 23, 2267-2276. https://doi.org/10.1016/j.csbj.2024.05.031.

**See our example on Colab:** BioPrediction_RPI_Example.ipynb File!

BioPrediction is part of a bigger project that proposes to democratize Machine Learning for the analysis, study, and control of epidemics and pandemics. [Take a look!!!](http://autoaipandemics.icmc.usp.br)

## Awards

BioPrediction - Project selected to participate in Prototypes for Humanity 2023, during COP28-Dubai, chosen from 3000 entries, from more than 100 countries, standing out among the 100 best, Prototypes for Humanity - COP28-Dubai. [Take a look!!!](https://www.prototypesforhumanity.com/project/bioprediction-framework/)


## Abstract

Machine Learning (ML) algorithms have been important tools for the extraction of useful knowledge from biological sequences particularly in healthcare, agriculture, and the environment. However, the categorical and unstructured nature of these sequences requiring usually additional feature engineering
steps, before an ML algorithm can be efficiently applied. The addition of these steps to the ML algorithm creates a processing pipeline, known as end-to-end ML. Despite the excellent results obtained by applying end-to-end ML to biotechnology problems, the performance obtained depends on the expertise of the user in the components of the pipeline. In this work, we propose an end-to-end ML-based framework called BioPrediction-RPI, which can identify implicit interactions between sequences, such as pairs of non-coding RNA and proteins, without the need for specialized expertise in end-to-end ML. This framework applies feature engineering to represent each sequence by structural and topological features. These features are divided into feature groups and used to train partial models, whose partial decisions are combined into a final decision, which, provides insights to the user by giving an interpretability report. In our experiments, the developed framework was competitive when compared with various expert-created models. We assessed BioPrediction-RPI with 12 datasets when it presented equal or
better performance than all tools in 40% to 100% of cases, depending on the experiment. Finally, BioPrediction-RPI can fine-tune models based on new data and perform at the same level as ML experts, thus democratizing end-to-end ML and increasing its access to those working in biological sciences.

* To the best of our knowledge, this is the first study to propose an automated pipeline for feature engineering and model training to classify interactions between biological sequences, competitive with models developed by experts.
  
* The pipeline was mainly tested on datasets regarding RNA-Protein interactions.

* BioPrediction-RPI does not require specialist human assistance.

* BioPrediction-RPI can accelerate new studies, democratizing the use of ML techniques by non-experts in ML.

<h1 align="center">
  <img src="https://github.com/0nurB/BioPredictionRPI-1.0/blob/main/img/overall.png" alt="BioPrediction-Flowchart" width="600"> 
</h1>

## Maintainers

* Robson Parmezan Bonidia, Bruno Rafael Florentino and Natan Henrique Sanches.

* **Correspondence:** rpbonidia@gmail.com or bonidia@usp.br, brunorf1204@usp.br, natan.sanches@usp.br


## Installing dependencies and package

#### Via miniconda (Terminal)

Installing BioPrediction using Miniconda to manage its dependencies, e.g.:

```sh
$ git clone https://github.com/0nurB/BioPredictionRPI-1.0.git BioPredictionRPI

$ cd BioPredictionRPI

$ git submodule init

$ git submodule update
```

**1 - Install Miniconda:** 

```sh

See documentation: https://docs.conda.io/en/latest/miniconda.html

$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

$ chmod +x Miniconda3-latest-Linux-x86_64.sh

$ ./Miniconda3-latest-Linux-x86_64.sh

$ export PATH=~/miniconda3/bin:$PATH

```

**2 - Create environment:**

```sh

conda create --name bioprediction-rpi python=3.11.5

```

**3 - Activate environment:**

```sh

conda activate bioprediction

```

**4 - Install packages:**

```sh

pip install -r requeriments.txt

```
## How to use

Execute the BioPrediction pipeline with the following command:

```sh
...
To run the code (Example): $ python BioPrediction.py -h

where:

 -input_interactions_train: CSV format file with the interaction matrix, e.g., datasets/exp_1/RPI369/RPI369_pairs.csv
 -input_interactions_test: CSV format file with the interaction matrix, e.g., datasets/exp_1/RPI369/RPI369_test_pairs.csv
 -input_interactions_candidates: CSV format file with the interaction candidates to the prediction, e.g., datasets/exp_1/RPI369/RPI369_candidates_pairs.csv

 -sequences_dictionary_protein: txt or fasta format file with the sequences, e.g., datasets/exp_1/RPI369/RPI369_protein_seq.fa
 -sequences_dictionary_rna: txt or fasta format file with the sequences, e.g., datasets/exp_1/RPI369/RPI369_dna_seq.fa

Those dictionaries must contain all sequences in train, test, and candidates.

 -topology_features: uses topology features to characterization of the sequences, e.g., yes or no, default=no)
 -output: output path, e.g., experiment_1

execution example:
python BioPrediction.py -input_interactions_train datasets/exp_1/RPI369/RPI369_pairs.csv -sequences_dictionary_protein datasets/exp_1/RPI369/RPI369_protein_seq.fa -sequences_dictionary_rna datasets/exp_1/RPI369/RPI369_dna_seq.fa -output exp_369

```

## Citation

If you use this code in a scientific publication, we would appreciate citations to the following paper:

Florentino, B. R., Parmezan Bonidia, R., Sanches, N. H., Da Rocha, U. N., & De Carvalho, A. C. (2024). BioPrediction-RPI: Democratizing the prediction of interaction between non-coding RNA and protein with end-to-end machine learning. Computational and Structural Biotechnology Journal, 23, 2267-2276. https://doi.org/10.1016/j.csbj.2024.05.031
