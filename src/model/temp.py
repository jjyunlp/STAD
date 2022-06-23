import torch

def negative_training(logits, targets, mode=None):
	"""
	计算每一个partial example中0位置的loss，最后得到一个avg loss
	而不是之前将所有0位置的概率加起来求Loss
	"""
	row_max = logits.max(dim=-1)[0].view(-1, 1)
	probs = torch.exp(logits - row_max)
	softmax_probs = probs / probs.sum(dim=-1).unsqueeze(-1).expand(-1, logits.size()[-1])
	softmax_probs = 1 - softmax_probs
	log_prob = torch.log(softmax_probs)
	targets = 1 - targets	# Depend on your target, If you use 1 to represent a negative, delete this code
	class_num = targets.sum(dim=-1)
	log_prob = log_prob * targets
	avg_log_prob = log_prob.sum(dim=-1) / class_num
	loss = -torch.mean(avg_log_prob)
	return loss

def positive_and_negative_training(logits, targets, flags, mode=None):
	"""
	混合了clean data and parital labeled data
	flags: if clean data: 1, ambiguous data: 0.如果batch size = 4，则flags.size = [4]
	"""
	# 将flags扩展成logits tensor的大小，并反转下1/0
	flags = 1 - flags.unsqueeze(-1).expand(-1, logits.size()[-1])
	row_max = logits.max(dim=-1)[0].view(-1, 1)
	probs = torch.exp(logits - row_max)
	softmax_probs = probs / probs.sum(dim=-1).unsqueeze(-1).expand(-1, logits.size()[-1])
	# 转换成positive and negative targets,即partial example中，negative class为1
	targets = torch.abs(targets - flags)    # (1-flags)是将flag转换下
	# softmax probs也要处理下，partial example的prob全变成1-prob
	softmax_probs = torch.abs(flags - softmax_probs)
	log_prob = torch.log(softmax_probs)
	class_num = targets.sum(dim=-1)
	log_prob = log_prob * targets
	avg_log_prob = log_prob.sum(dim=-1) / class_num
	loss = -torch.mean(avg_log_prob)
	return loss