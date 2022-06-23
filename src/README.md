# Self-Training for Relation Extraction

## About The Project
A self-training based framework for semi-supervised relation extraction.

## Code Framework
- baseline.py
- self_training_merge.py
- self_training_staged_fine_tune.py


### models
- bert_model.py: base relation extraction model

- model_utils.py
	- EntityPooler: output hidden states of assigned location(batch based).
	- CLSPooler: built-in cls output has been done by dense and activation
	- some loss functions
	- tensor2onehot

### data_loader_and_dumper
- data_loader_and_dumper.py: base module
- tacred_loader.py: child class
- semeval_loader.py

### data_processor
- data_processor.py: base data processor for relation extraction.
	- json -> inst
	- inst -> bert_example
	- bert_example -> bert_feature
	- bert_feature -> bert_tensor
- self_training_data_processor
	- unlabel data

### data_prepare
- small data: randomly split data
- unlabel data

## Requirements
PythonXX
PyTorch(==1.0.1)

## Dataset
- SemEval
- Re-TACRED

## Installation
TODO



