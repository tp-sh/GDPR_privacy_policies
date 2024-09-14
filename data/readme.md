# Description

This directory contains the data used and generated in our study. Below is a brief description of each file:

- `GoPPC-150`: This directory contains 150 hierarchically labeled privacy policies. It represents our proposed corpus and serves as the foundation for our study.

- `GDPR_taxonomy.ppx`: This file presents our proposed GDPR-oriented privacy policy taxonomy. The red boxes represent newly added concepts, underlined text represents concepts covered by the ICO template, and the percentages represent coverage rates of different GDPR concepts in our new GoPPC-150 corpus.

- `Torre_classifier_results.csv`: This file contains the results of our reproduction of Torre et al.'s classifier on our corpus.

- `Web_list.csv`: This file lists the 150 websites included in our corpus along with their update times.

- `comparison_experiments.csv`: This file contains the results(f1 values) of comparing different mainstream machine learning models, using 300-D TF-IDF features of the current node as input. The experiments were conducted at the segment level on the GoPPC-150 corpus.

- `complete_classifier_results.csv`: This file presents the results(f1 values) of 12 types of classifiers (as defined in our paper) on our corpus for each label at both the segment and document level.

- `keyword_list.csv`: This file contains a list of keywords for each of the 96 concepts in the GDPR taxonomy.

- `opp_result.csv`: This file presents our experiment results on the OPP-115 datasets at both the segment and document level.