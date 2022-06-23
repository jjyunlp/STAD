import torch
from torch import tensor
import torch.nn.functional as F
from torch import logsumexp, mul, nn

from torch import autograd
input = autograd.Variable(torch.randn(3,3), requires_grad=True)
input = torch.Tensor([[1, 4.9, 5], [10, 12.9, 0.1], [5, 10, 1], [6, 2, 3]])
partial_input = torch.Tensor([[0.1, 4.9, 5], [13, 12.9, 0.1], [5, 10, 1]])
print(input)
m = nn.Sigmoid()
print(m(input))

target = torch.FloatTensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
target_long = torch.LongTensor([2, 0, 1])
input = torch.Tensor([[1, 4.9, 5, 3.0],
                      [10, 12.9, 0.1, 5.0],
                      [5, 10, 1, 2.0]])
partial_target = torch.FloatTensor([[1, 0, 1, 0],
                                    [1, 0, 0, 0],
                                    [0, 1, 1, 0]])
flags = torch.Tensor([0, 1, 0])
print(target)
"""
loss_func = nn.BCELoss()
loss = loss_func(m(input), target)
print(loss)
loss_func = nn.BCEWithLogitsLoss()
loss = loss_func(input, target)
print(loss)
"""

def semiloss(outputs_x, targets_x):
	Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
	return Lx


def partial_annotation_cross_entropy(predictions, targets, epsilon=1e-12):
	"""
	Computes cross entropy between targets (encoded as one-hot vectors)
	and predictions. 
	Input: predictions (N, k) ndarray
		targets (N, k) ndarray        
	Returns: scalar
	"""
	#predictions = np.clip(predictions, epsilon, 1. - epsilon)
	# if size of data % batch_size = 1, not [[]], but []
	# So, softmax(predictions, 0), not 1
	predictions = torch.softmax(predictions, 1)
	print("probs")
	print(predictions)
	N = predictions.shape[0]
	# 如果我们希望在多个标签上的概率是相同的，那我们加一个loss，比如
	# 或者设计一个CE，能够支持[0.5, 0.5, 0]这种标签，同时也支持[1, 0, 0]的标签
	# 前者是在多个上面loss的相加，后者是一个上面的相加
	# 比如，现在预测的概率是[0.7, 0.2, 0.1]
	# 若是前者标签，那loss = 0.7-
	multi_prob = torch.sum(targets * predictions, dim=1)
	print("Multi-Prob: ")
	print(multi_prob)
	# multi_prob = torch.max(targets * predictions, dim=1)[0]  # 只要所有1位置上，prob最大的那个
	# ce = -torch.sum(torch.log(multi_prob) + 1e-9)/N
	ce = -torch.sum(torch.log(multi_prob + epsilon))/N
	return ce

def own_cross_entropy(logits, targets):
	# copy from HaiTao Wang
	# 想办法改成支持ambiguity annotation的，不过在取得max的大小后，需要调整onehot的值
	#  跟我们的设想不一样，CE是只计算1位置上的信息，而忽略所有0位置上的情况。也就是只有当1位置上的值不是最大的时候，才会有loss产生，否则就是0.
	row_max = logits.max(dim=-1)[0].view(-1, 1)
	log_sum_exp = torch.log(torch.sum(torch.exp(logits - row_max), dim=-1)).view(-1, 1)
    # torch.logsumexp()
	log_prob = logits - row_max - log_sum_exp
	entropy = - log_prob * targets
	classifier_loss = torch.mean(entropy.sum(dim=-1), dim=-1)
	#print(entropy)
	return classifier_loss

def negative_training_for_partial(logits, targets):
    row_max = logits.max(dim=-1)[0].view(-1, 1)
    probs = torch.exp(logits - row_max)
    softmax_probs = probs / probs.sum(dim=-1).unsqueeze(-1).expand(-1, logits.size()[-1])
    print(probs)
    print(probs.sum(dim=-1).unsqueeze(-1).expand(-1, logits.size()[-1]))
    print(softmax_probs)
    softmax_probs = 1 - softmax_probs
    print(softmax_probs)
    log_prob = torch.log(softmax_probs)
    print(log_prob)
    targets = 1- targets
    class_num = targets.sum(dim=-1)
    log_prob = log_prob * targets
    print(log_prob)
    avg_log_prob = log_prob.sum(dim=-1) / class_num
    print(avg_log_prob)
    loss = -torch.mean(avg_log_prob)
    print(loss)


def positive_and_negative_training(logits, targets, flags, mode=None):
    """
    计算每一个partial example中0位置的loss，最后得到一个avg loss
    而不是之前将所有0位置的概率加起来求Loss
    2021/09/07 下一步，添加一个flag标记，表明当前的是partial 还是clean，从而选用positive training还是negative training
    融合在一起
    flags: if clean data: 1, ambiguous data: 0
    """
    flags = 1 - flags.unsqueeze(-1).expand(-1, logits.size()[-1])
    row_max = logits.max(dim=-1)[0].view(-1, 1)
    probs = torch.exp(logits - row_max)
    softmax_probs = probs / probs.sum(dim=-1).unsqueeze(-1).expand(-1, logits.size()[-1])
    # 转换成positive and negative targets,即partial example中，negative class为1
    print(targets)
    targets = torch.abs(targets - flags)    # (1-flags)是将flag转换下
    print(targets)
    # softmax probs也要处理下，partial example的prob全变成1-prob
    print(softmax_probs)
    softmax_probs = torch.abs(flags - softmax_probs)
    print(softmax_probs)
    log_prob = torch.log(softmax_probs)
    class_num = targets.sum(dim=-1)
    log_prob = log_prob * targets
    print("log_prob", log_prob)
    print(class_num)
    avg_log_prob = log_prob.sum(dim=-1) / class_num
    loss = -torch.mean(avg_log_prob)
    print(loss)
    return loss
    

def positive_training_support_partial(logits, targets, mode=None):
    """
    针对有一个或多个class标为1的数据
    对于多个class标为1，则将其概率相加，得到loss，并除以标为1的总数。不除一下的话，loss这边会更新多一点，不公平。
    比如，一个1预测为0.9，假设更新alpha
    若两个1的预测概率分别为0.5，0.4，那合起来也是0.9，更新值也是alpha，但这个alpha会加在两个class对应的参数上，显然多了。
    """
    row_max = logits.max(dim=-1)[0].view(-1, 1)
    probs = torch.exp(logits - row_max)
    print("probs", probs)
    softmax_probs = probs / probs.sum(dim=-1).unsqueeze(-1).expand(-1, logits.size()[-1])
    # 将位置为1的概率加起来
    print("softmax_probs", softmax_probs)
    multi_softmax_probs = torch.sum(targets * softmax_probs, dim=1)
    print("multi_softmax_probs", multi_softmax_probs)
    log_probs = torch.log(multi_softmax_probs)
    print("log_probs", log_probs)
    class_num = targets.sum(dim=-1)
    print("class_num", class_num)
    avg_log_prob = log_probs / class_num     # class num可以尝试针对多个1的情况
    print("avg_log_prob", avg_log_prob)
    loss = -torch.mean(avg_log_prob)
    print(loss)
    return loss


def own_cross_entropy_for_negative_training(logits, targets, mode=None):
    # base copy from HaiTao Wang
    # for negative training, calculate all 0's loss, then average
    row_max = logits.max(dim=-1)[0].view(-1, 1)
    print(row_max, lmax)
    
    sum_exp = torch.sum(torch.exp(logits - row_max), dim=-1)
    print('sum_exp', sum_exp)
    # sum_exp = sum_exp.expand(logits.size())
    sum_exp = sum_exp.unsqueeze(-1).expand(-1, 3)
    print('sum_exp', sum_exp)
    log_sum_exp = torch.log(torch.sum(torch.exp(logits - row_max), dim=-1)).view(-1, 1)
    print(log_sum_exp)
    log_prob = logits - row_max - log_sum_exp
    print("exp(logits)", torch.exp(logits))
    print("sum-exp", sum_exp - torch.exp(logits))
    
    log_prob = torch.log(sum_exp - torch.exp(logits) - row_max) - log_sum_exp
    print("log_prob", log_prob)
    entropy = - log_prob * targets
    print("entropy", entropy)
    # 需要先算一下每个的平均值，即partial label的平均值
    # entropy = torch.mean
    # 这个sum，应该就把ambiguity考虑了吧，虽然跟我想象的不太一样
    print("sum", entropy.sum(dim=-1))
    print("mean", torch.mean(entropy, dim=-1))
    class_num = targets.sum(dim=-1)     # 每个例子分别有多少个0
    print("class_num", class_num)
    print("0's mean", entropy.sum(dim=-1)/class_num)
    print('my loss', torch.mean(entropy.sum(dim=-1)/class_num))
    print("target sum", targets.sum(dim=-1))
    classifier_loss = torch.mean(entropy.sum(dim=-1), dim=-1)
    print("loss", classifier_loss)
    #print(entropy)
    return classifier_loss


def negative_training(predictions, targets, epsilon=1e-12, mode=None):
    # 将所有1位置的概率加起来，log(1-p)进行训练
    targets = - targets + 1     # 0/1转换
    predictions = torch.softmax(predictions, 1)
    predictions = 1 - predictions
    N = predictions.shape[0]
    # 先求loss，各个位置上的loss
    loss = torch.log(predictions + epsilon)
    print(loss)
    

    #multi_prob = torch.sum(targets * predictions, dim=1)
    #ce = -torch.sum(torch.log(1 - multi_prob + epsilon))/N
    return loss


def negative_training_support_partial_sum_prob(logits, targets, flags=None, mode=None):
    """
    针对有一个或多个class标为1的数据
    对于多个class标为1，则将其概率相加，得到loss，并除以标为1的总数。不除一下的话，loss这边会更新多一点，不公平。
    比如，一个1预测为0.9，假设更新alpha
    若两个1的预测概率分别为0.5，0.4，那合起来也是0.9，更新值也是alpha，但这个alpha会加在两个class对应的参数上，显然多了。
    对于partial annotated data，用0位置的概率和求loss，最终1-prob
    clean data还是正常的positive
    需要引入flags向量
    """
    # 将flags扩展成logits tensor的大小，并反转下1/0
    # flags = 1 - flags.unsqueeze(-1).expand(-1, logits.size()[-1])
    flags = 1 - flags     # 因为是先把各个位置的概率加起来了，因此，不用扩展了，就[batch]就行
    row_max = logits.max(dim=-1)[0].view(-1, 1)
    probs = torch.exp(logits - row_max)
    softmax_probs = probs / probs.sum(dim=-1).unsqueeze(-1).expand(-1, logits.size()[-1])
    print("softmax_probs", softmax_probs)
    # 将所有1位置的概率加起来，clean data就还是一个位置的概率，而partial data则会累加多个
    multi_softmax_probs = torch.sum(targets * softmax_probs, dim=1)
    print("multi softmax_probs", multi_softmax_probs)
    # 转换成positive and negative targets,即partial example中，negative class为1
    # targets = torch.abs(targets - flags)    # (1-flags)是将flag转换下
    # softmax probs也要处理下，partial example的prob全变成1-prob
    multi_softmax_probs = torch.abs(flags - multi_softmax_probs)
    print("multi softmax_probs", multi_softmax_probs)
    log_prob = torch.log(multi_softmax_probs)
    print("log_prob", log_prob)
    class_num = targets.sum(dim=-1)
    avg_log_prob = log_prob / class_num
    print("avg_log_prob", avg_log_prob)
    loss = -torch.mean(avg_log_prob)
    print(loss)
    return loss

tt = target

import random
for t in target:
	print(len(t))
	index = random.randint(0, len(t)-1)
	while t[index] == 1:
		index = random.randint(0, len(t)-1)
	t *= 0
	t[index] = 1
#"""

print(input)
print(partial_target)
# own_cross_entropy_for_negative_training(input, partial_target)
# positive_and_negative_training(input, partial_target, flags)
# positive_training_support_partial(input, partial_target)
negative_training_support_partial_sum_prob(input, partial_target, flags)
exit()


probs = torch.randn(5, 3)
max_idx = torch.argmax(probs, -1, keepdim=True)
print(probs)
print(max_idx)

one_hot = torch.FloatTensor(probs.shape)
one_hot.zero_()
one_hot.scatter_(-1, max_idx, 1)
print(one_hot)

probs = torch.softmax(torch.randn(5, 3), dim=-1)
max_idx = torch.argmax(probs, -1, keepdim=True)
print(probs)
print(max_idx)

one_hot2 = torch.FloatTensor(probs.shape)
one_hot2.zero_()
one_hot2.scatter_(-1, max_idx, 1)

print(one_hot)

print(one_hot2)
final_onehot = torch.mul(one_hot, one_hot2)
print(final_onehot)

N = probs.shape[0]
multi_prob = torch.sum(final_onehot * probs, dim=1)
print(multi_prob)
print(torch.log(1- multi_prob))
ce = -torch.sum(torch.log(1 - multi_prob) + 1e-9)/N
print(ce)
a = torch.Tensor([1.0, 1.0])
print(torch.log(a))

print("input, target")
print(input)
print(target)
loss = partial_annotation_cross_entropy(input, target)
print("loss:")
print(loss)
exit()
print("Partial input and target")
print(partial_input, partial_target)
loss = partial_annotation_cross_entropy(partial_input, partial_target)
print("partial 0.5 0.5")
print(loss)
standard_loss = F.cross_entropy(input, target_long)
print(standard_loss)
exit()

loss = semiloss(input, target)
print(loss)

loss = semiloss(partial_input, partial_target)
print(loss)
loss = semiloss(input, partial_target)
print(loss)
exit()

loss = own_cross_entropy(input, target)
print(loss)
partial_target = torch.FloatTensor([[0, 1, 1], [1, 1, 0], [0, 1, 0]])
loss = partial_annotation_cross_entropy(input, partial_target)
print("partial 1 1 ")
print(loss)

ce_target = torch.LongTensor([2, 0, 1])
loss = own_cross_entropy(input, target)
print(loss)
loss = own_cross_entropy(input, partial_target)
print(loss)


from torch.nn import CrossEntropyLoss

loss_func = CrossEntropyLoss()
loss = loss_func(input, ce_target)
print(loss)