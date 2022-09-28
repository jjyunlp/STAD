# STAD: Self-Training with Ambiguous Data for Low-Resource Relation Extraction
This is the implementation of our submissions for [COLING-2022](https://coling2022.org/).

## Overview
<img width="510" alt="image" src="https://user-images.githubusercontent.com/29971305/190667801-db7ab5f1-20b8-427d-b47f-1654148045e6.png">


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
