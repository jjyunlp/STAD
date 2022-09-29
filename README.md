# STAD: Self-Training with Ambiguous Data for Low-Resource Relation Extraction
This is the implementation of our [work](https://arxiv.org/pdf/2209.01431.pdf) for [COLING-2022](https://coling2022.org/).

## Overview
Self-training has proven effective for improving NLP tasks. The common practice is to construct confident synthetic data via a probability threshold. In this work, we propose a novel self-training framework on low-resource relation extraction with modeling ambiguous data. In detail, we first propose a method to identify ambiguous but useful instances from the uncertain instances and then divide the relations into candidate-label set and negative-label set for each ambiguous instance. Next, we propose a set-negative training method on the negative-label sets for the ambiguous instances and a positive training method for the confident instances. Finally, a joint-training method is proposed to build the final relation extraction system on all data. Experimental results on SemEval-2010 Task8 and Re-TACRED demonstrate the effectiveness of the proposed method. Extensive anayses on top-N evaluation provide a deeper understanding of how the proposed methods learn from the ambiguous data.
<p align='center'>
<img width="510" alt="image" src="https://user-images.githubusercontent.com/29971305/190667801-db7ab5f1-20b8-427d-b47f-1654148045e6.png">
</p>

## Preparation
Download SemEval-2010 Task8 dataset and Re-TACRED datasets.

Split data into our low-resource setting using scripts in `src/preprocess/building_low_resource.py`

## Dependencies
- python >= 3.6
- pytorch >= 1.0.1

## Running
Modify the exp_data_dir for output and model_name_or_path for load BERT in `running_for_re_tacred_exclude_NA_cross_data.py`

- train_small_base part is for training the SUPERVISED model on small labeled dataset.

- train_self_training part is for training the SELF-TRAINING model by tagging the unlabeled data with above SUPERVISED model, and then merge confident data with small labeled data.
- train_self_training_partial_negative_and_ablation part is for training our STAD model and its ablation models. 

## Research Citation
If the code is useful for your research project, we appreciate if you cite the following [paper](https://arxiv.org/pdf/2209.01431.pdf):
```
@article{yu2022stad,
  title={STAD: Self-Training with Ambiguous Data for Low-Resource Relation Extraction},
  author={Yu, Junjie and Wang, Xing and Zhao, Jiangjiang and Yang, Chunjie and Chen, Wenliang},
  journal={arXiv preprint arXiv:2209.01431},
  year={2022}
}
```
