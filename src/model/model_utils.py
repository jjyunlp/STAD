"""
放一些通用的、模型中需要用到的操作
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class ConsistencyLoss(object):
    """
    two input
    """
    def __call__(self, func_type, primary_logits, aux_logits, current_epoch=None, total_epoch=None):
        """
        输入的是classifier后的表示，需要softmax下，即变成distribution.
        目前用不到rampup
        """
        loss = None
        if func_type == 'KL':
            loss_func = nn.KLDivLoss(reduction='sum')   # KL
            primary_logits = F.softmax(primary_logits)
        elif func_type == 'BiKL':
            #loss_func = nn.KLDivLoss(reduction='sum')   # KL
            # primary_distribution = F.log_softmax(primary_logits)
            #primary_distribution = F.softmax(primary_logits, dim=-1)
            #aux_distribution = F.softmax(aux_logits, dim=-1)
            primary_loss = F.kl_div(F.log_softmax(primary_logits, dim=-1), F.softmax(aux_logits, dim=-1), reduction='sum')
            aux_loss = F.kl_div(F.log_softmax(aux_logits, dim=-1), F.softmax(primary_logits, dim=-1), reduction='sum')
            primary_loss = primary_loss.mean()
            aux_loss = aux_loss.mean()
            loss = (primary_loss + aux_loss)/2
            # loss = loss_func(primary_distribution, aux_distribution) + loss_func(aux_distribution, primary_distribution)
        elif func_type == 'myBiKL':
            loss_func = nn.KLDivLoss(reduction='sum')   # KL
            primary_distribution = F.softmax(primary_logits, dim=-1)
            aux_distribution = F.softmax(aux_logits, dim=-1)
            primary_loss = loss_func(primary_distribution, aux_distribution)
            aux_loss = loss_func(aux_distribution, primary_distribution)
            loss = (primary_loss + aux_loss)
            # loss = loss_func(primary_distribution, aux_distribution) + loss_func(aux_distribution, primary_distribution)
        elif func_type == "MSE":
            loss_func = nn.MSELoss(reduction='sum')   # L2
        elif func_type == 'CE':
            loss_func = own_cross_entropy
        else:
            print(f"LOSS FUNC ERROR: {func_type}")
            exit()
        weight = 1.0
        if current_epoch is not None and total_epoch is not None:
            weight = linear_rampup(current_epoch, total_epoch)
        return (loss, weight)
        # return (sum_loss, weight)


class MyLossFunction(object):
    def linear_rampup(current, rampup_length):
        """根据当前训练轮数，返回loss的权重。

        Args:
            current ([type]): 当前轮数
            rampup_length ([type]): 总的训练轮数，我们目前设置为20

        Returns:
            [type]: [description]
        """
        # 使得lambda随着epoch越来越大，直至最后为1.0，即两个loss平起平坐
        if rampup_length == 0:
            return 1.0
        else:
            # -> 如果是整个unlabel全部拿上来，那必须用。即，一开始unlabel基本不参与loss
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)

    def own_cross_entropy(logits, targets, mode=None):
        # mode不用
        # copy from HaiTao Wang
        row_max = logits.max(dim=-1)[0].view(-1, 1)
        log_sum_exp = torch.log(torch.sum(torch.exp(logits - row_max), dim=-1)).view(-1, 1)
        log_prob = logits - row_max - log_sum_exp
        entropy = - log_prob * targets
        # 这个sum，应该就把ambiguity考虑了吧，虽然跟我想象的不太一样
        classifier_loss = torch.mean(entropy.sum(dim=-1), dim=-1)
        #print(entropy)
        return classifier_loss

    def own_cross_entropy_for_negative_training(logits, targets, mode=None):
        # base copy from HaiTao Wang
        # for negative training, calculate all 0's loss, then average
        row_max = logits.max(dim=-1)[0].view(-1, 1)
        log_sum_exp = torch.log(torch.sum(torch.exp(logits - row_max), dim=-1)).view(-1, 1)
        log_prob = logits - row_max - log_sum_exp
        entropy = - log_prob * targets
        # 需要先算一下每个的平均值，即partial label的平均值
        # entropy = torch.mean
        # 这个sum，应该就把ambiguity考虑了吧，虽然跟我想象的不太一样
        classifier_loss = torch.mean(entropy.sum(dim=-1), dim=-1)
        #print(entropy)
        return classifier_loss
    
    def own_cross_entropy_for_partial_annotation(logits, targets):
        # 先得到最大的logit
        row_max = logits.max(dim=-1)[0].view(-1, 1)
        log_sum_exp = torch.log(torch.sum(torch.exp(logits - row_max), dim=-1)).view(-1, 1)
        log_prob = logits - row_max - log_sum_exp
        print("TODO")
        exit()


    
    def BCE_loss_for_multiple_classification(predictions, targets):
        bce_mean = F.binary_cross_entropy_with_logits(predictions, targets)
        return bce_mean



    def partial_annotation_cross_entropy(predictions, targets, epsilon=1e-12, mode='sum'):
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
        N = predictions.shape[0]
        #loss = prob - onehot
        if mode == 'sum':
            multi_prob = torch.sum(targets * predictions, dim=1)
        elif mode == 'max':
            multi_prob = torch.max(targets * predictions, dim=1)[0]  # 只要所有1位置上，prob最大的那个
        elif mode == 'min': # 这个就是不可能可以的东西，浪费时间
            multi_prob = torch.min(targets * predictions, dim=1)[0]  # 只要所有1位置上，prob最大的那个
        else:
            print(f"Error mode: {mode}")
        # ce = -torch.sum(torch.log(multi_prob) + 1e-9)/N
        # 不知道这个是否有影响，将epsilon加在log之前。感觉这个才能避免inf的问题
        ce = -torch.sum(torch.log(multi_prob + epsilon))/N
        return ce

    def positive_training(logits, targets, mode=None):
        """
        针对只有一个class标为1的数据
        """
        row_max = logits.max(dim=-1)[0].view(-1, 1)
        probs = torch.exp(logits - row_max)
        softmax_probs = probs / probs.sum(dim=-1).unsqueeze(-1).expand(-1, logits.size()[-1])
        log_prob = torch.log(softmax_probs)
        class_num = targets.sum(dim=-1)
        log_prob = log_prob * targets
        avg_log_prob = log_prob.sum(dim=-1) / class_num     # class num可以尝试针对多个1的情况
        loss = -torch.mean(avg_log_prob)
        return loss

    def positive_training_support_partial_sum_prob(logits, targets, flags=None, mode=None):
        """
        针对有一个或多个class标为1的数据
        对于多个class标为1，则将其概率相加，得到loss，并除以标为1的总数。不除一下的话，loss这边会更新多一点，不公平。
        比如，一个1预测为0.9，假设更新alpha
        若两个1的预测概率分别为0.5，0.4，那合起来也是0.9，更新值也是alpha，但这个alpha会加在两个class对应的参数上，显然多了。
        """
        row_max = logits.max(dim=-1)[0].view(-1, 1)
        probs = torch.exp(logits - row_max)
        softmax_probs = probs / probs.sum(dim=-1).unsqueeze(-1).expand(-1, logits.size()[-1])
        # 将位置为1的概率加起来
        multi_softmax_probs = torch.sum(targets * softmax_probs, dim=1)
        log_probs = torch.log(multi_softmax_probs)
        class_num = targets.sum(dim=-1)
        # 求平均这一步最重要，也不是求平均，就是概率和的loss需要再除以总数，否则会多更新
        avg_log_prob = log_probs / class_num     # class num可以尝试针对多个1的情况，这一步最重要
        loss = -torch.mean(avg_log_prob)
        return loss

    def positive_training_support_partial_sum_loss(logits, targets, flags=None, mode=None):
        """
        针对有一个或多个class标为1的数据
        对于多个class标为1，得到各个loss，求和后取平均。
        """
        row_max = logits.max(dim=-1)[0].view(-1, 1)
        probs = torch.exp(logits - row_max)
        softmax_probs = probs / probs.sum(dim=-1).unsqueeze(-1).expand(-1, logits.size()[-1])
        # 计算所有1位置的loss
        log_probs = torch.log(softmax_probs)
        multi_log_probs = torch.sum(targets * log_probs, dim=1)
        class_num = targets.sum(dim=-1)
        # 求平均这一步最重要，也不是求平均，就是概率和的loss需要再除以总数，否则会多更新
        avg_log_prob = multi_log_probs / class_num     # class num可以尝试针对多个1的情况，这一步最重要
        loss = -torch.mean(avg_log_prob)
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
        # 将所有1位置的概率加起来，clean data就还是一个位置的概率，而partial data则会累加多个
        multi_softmax_probs = torch.sum(targets * softmax_probs, dim=1)
        # softmax probs也要处理下，partial example的prob全变成1-prob
        multi_softmax_probs = torch.abs(flags - multi_softmax_probs)
        log_prob = torch.log(multi_softmax_probs)
        class_num = targets.sum(dim=-1)
        avg_log_prob = log_prob / class_num
        loss = -torch.mean(avg_log_prob)
        return loss

    def negative_training_support_partial_sum_loss(logits, targets, flags=None, mode=None):
        """
        针对有一个或多个class标为1的数据
        对于多个class标为1，则将其概率相加，得到loss，并除以标为1的总数。不除一下的话，loss这边会更新多一点，不公平。
        比如，一个1预测为0.9，假设更新alpha
        若两个1的预测概率分别为0.5，0.4，那合起来也是0.9，更新值也是alpha，但这个alpha会加在两个class对应的参数上，显然多了。
        对于partial annotated data，用0位置的概率和求loss，最终1-prob
        clean data还是正常的positive
        需要引入flag向量
        这个和下下的positive_and_negative_training是一样的，不同的是，他的输入已经random选了一个negative label了
        """
        # 将flags扩展成logits tensor的大小，并反转下1/0
        flags = 1 - flags.unsqueeze(-1).expand(-1, logits.size()[-1])
        row_max = logits.max(dim=-1)[0].view(-1, 1)
        probs = torch.exp(logits - row_max)
        softmax_probs = probs / probs.sum(dim=-1).unsqueeze(-1).expand(-1, logits.size()[-1])
        # 转换成positive and negative targets,即partial example中，negative class为1
        targets = torch.abs(targets - flags)
        # softmax probs也要处理下，partial example的prob全变成1-prob
        softmax_probs = torch.abs(flags - softmax_probs)
        log_prob = torch.log(softmax_probs)
        class_num = targets.sum(dim=-1)
        log_prob = log_prob * targets
        avg_log_prob = log_prob.sum(dim=-1) / class_num
        loss = -torch.mean(avg_log_prob)
        return loss

    def negative_training(logits, targets, mode=None):
        """
        计算每一个partial example中0位置的loss，最后得到一个avg loss
        而不是之前将所有0位置的概率加起来求Loss
        2021/09/07 下一步，添加一个flag标记，表明当前的是partial 还是clean，从而选用positive training还是negative training
        """
        row_max = logits.max(dim=-1)[0].view(-1, 1)
        probs = torch.exp(logits - row_max)
        softmax_probs = probs / probs.sum(dim=-1).unsqueeze(-1).expand(-1, logits.size()[-1])
        softmax_probs = 1 - softmax_probs
        log_prob = torch.log(softmax_probs)
        targets = 1 - targets
        class_num = targets.sum(dim=-1)
        log_prob = log_prob * targets
        avg_log_prob = log_prob.sum(dim=-1) / class_num
        loss = -torch.mean(avg_log_prob)
        return loss

    def positive_and_negative_training(logits, targets, flags, mode=None):
        """
        混合了clean data and parital labeled data
        flags: if clean data: 1, ambiguous data: 0.如果batch size = 4，则flags.size = [4]
        这个看起来是最完善的版本
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

    def negative_training_old(predictions, targets, epsilon=1e-12, mode=None):
        for target in targets:
            index = random.randint(0, len(target)-1)
            while target[index] == 1:
                index = random.randint(0, len(target)-1)
            target *= 0
            target[index] = 1
        predictions = torch.softmax(predictions, 1)
        N = predictions.shape[0]
        multi_prob = torch.sum(targets * predictions, dim=1)
        ce = -torch.sum(torch.log(1 - multi_prob) + 1e-9)/N
        return ce

    def negative_training_for_ambiguous(predictions, targets, epsilon=1e-12, mode=None):
        # 每次根据概率，将最高的概率变成1，其余为0，得到一个onehot，再将这个onehot与标注的反onehot进行融合，取交集。
        # 这样就只有一个位置为1. 并进行log(1-p)进行训练
        # 意思是最高的概率在negative labels中，我们因此进行更新，否则的话loss为0。
        # 因此，仅更新最高概率不在已有的模糊标注中的情况
        # 也可以改成在所有0位置上的最高概率，进行negative training
        # 基于我们的假设：如果一个模型预测的类别在部分标注的候选集中，那当前模型对于这个实例是成功的。
        targets = - targets + 1     # 0/1转换
        probs = torch.softmax(predictions, 1)
        max_idx = torch.argmax(probs, -1, keepdim=True)
        # one_hot = torch.FloatTensor(probs.shape)
        # one_hot.zero_()
        one_hot = targets * 0   # 避免的cpu cuda的转换，
        one_hot.scatter_(-1, max_idx, 1)

        targets = torch.mul(one_hot, targets)   # negative labels AND highest prob index

        N = probs.shape[0]
        multi_prob = torch.sum(targets * probs, dim=1)
        ce = -torch.sum(torch.log(1 - multi_prob) + epsilon)/N
        return ce

    def negative_training_for_ambiguous_in_negative_class(predictions, targets, epsilon=1e-12, mode=None):
        # 每次根据概率，将negative中【与上述函数的不同之处】最高的概率变成1，其余为0
        # 这样就只有一个位置为1. 并进行log(1-p)进行训练
        # 意思是在negative labels中取一个概率最高的，将它进行打压。 进行negative training
        # 【我们想到的问题是，这个打压下去了，会不会导致其他negative上来。而如果只有一个positive label，今天提高，则没有这个问题。
        # 即1位置的提升了，其他位置自然只能下降。但某个0位置的降低了，并不表示会抬高ambiguous中的1位置】
        # 还没写。。。。
        targets = - targets + 1     # 0/1转换
        probs = torch.softmax(predictions, 1)
        probs *= targets    # positive labels的概率为0
        max_idx = torch.argmax(probs, -1, keepdim=True)     # max_idx只能是所有negative labels中最高概率的一个
        # one_hot = torch.FloatTensor(probs.shape)
        # one_hot.zero_()
        one_hot = targets * 0   # 避免的cpu cuda的转换，
        one_hot.scatter_(-1, max_idx, 1)

        targets = torch.mul(one_hot, targets)   # negative labels AND highest prob index

        N = probs.shape[0]
        multi_prob = torch.sum(targets * probs, dim=1)
        ce = -torch.sum(torch.log(1 - multi_prob) + 1e-9)/N
        return ce
            
    
    def partial_annotation_cross_entropy_for_negative(logits, targets, epsilon=1e-12):
        """
        以negative的方式进行训练，比如，clean data就只有一个为0 ，其余都是1
        而，parital annotation的data，则有多个0。
        每次随机给一个label ？进行negative training，参照SENT的文章。
        """
        row_max = logits.max(dim=-1)[0].view(-1, 1)
        log_sum_exp = torch.log(torch.sum(torch.exp(logits - row_max), dim=-1)).view(-1, 1)
        log_prob = logits - row_max - log_sum_exp
        entropy = - log_prob * targets
        # 这个sum，应该就把ambiguity考虑了吧，虽然跟我想象的不太一样
        classifier_loss = torch.mean(entropy.sum(dim=-1), dim=-1)
        #print(entropy)
        return classifier_loss
    
        targets = - targets + 1
        #predictions = np.clip(predictions, epsilon, 1. - epsilon)
        # if size of data % batch_size = 1, not [[]], but []
        # So, softmax(predictions, 0), not 1
        predictions = F.softmin(predictions, 1)     # use softmin
        N = predictions.shape[0]
        multi_prob = torch.sum(targets * predictions, dim=1)    # 所有负例的概率和要降低
        ce = -torch.sum(torch.log(multi_prob) + 1e-9)/N
        return ce

    def partial_annotation_cross_entropy_new(logits, targets, epsilon=1e-12):
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
        row_max = logits.max(dim=-1)[0].view(-1, 1)
        log_sum_exp = torch.log(torch.sum(torch.exp(logits - row_max), dim=-1)).view(-1, 1)
        log_prob = logits - row_max - log_sum_exp
        entropy = - torch.sum(log_prob * targets)
        classifier_loss = torch.mean(entropy.sum(dim=-1), dim=-1)


        sum_exp = torch.sum(torch.exp(logits - row_max), dim=-1).view(-1, 1)
        prob = logits - row_max - log_sum_exp

        predictions = torch.softmax(predictions, 1)

        N = predictions.shape[0]
        multi_prob = torch.sum(targets * predictions, dim=1)
        ce = -torch.sum(torch.log(multi_prob) + 1e-9)/N
        return ce


class TensorOperation(object):
    """Put some base tensor operation here.
    """
    def tensor2onehot(self, index, size):
        """
        index: [batch_size]
        size: the size of one hot (num of classification types)
        """
        batch_size = index.size()[0]
        one_hot = torch.FloatTensor(batch_size, size).zero_()
        #scatter_(dim, index, src)
        # dim (python:int) – the axis along which to index, to put src in a tensor
        # src: the value to write, hence, src=1 will output onehot
        # Hence, the index and one_hot(all initialized 0) should have the same size in dimesion dim
        index = index.unsqueeze(-1)
        one_hot.scatter_(1, index, 1)
        return one_hot


class EntityPooler(nn.Module):
    r"""
    根据句子中实体的位置，输出对应的表示
    (输入的都是一个batch的tensor)
    """
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, input_id):
        """将BERT输出每个词的序列的表示中，提取出给定位置的表示

        Args:
            hidden_states ([type]): [batch, 128, 768]
            input_id ([type]): [??] 反正最后view(-1, 1)

        Returns:
            [type]: [description]
        """
        # We "pool" the relation model by simply taking the hidden state corresponding
        # to the head start tag and tail start tag ([e1] and [e2], respecitivelly).
        input_token_tensor = torch.gather(hidden_states, 1, input_id.view(-1, 1).unsqueeze(2).repeat(1, 1, hidden_states.size()[2])).squeeze()
        input_token_tensor = torch.gather(hidden_states, 1, input_id.view(-1, 1).unsqueeze(2).repeat(1, 1, hidden_states.size()[2])).squeeze()
        if input_token_tensor.dim() == 1:   # [1, 768] suqeeze to [768], fix it
            input_token_tensor = input_token_tensor.unsqueeze(0)
    
        return input_token_tensor


class CLSPooler(nn.Module):
    r"""
    BERT内部的CLS pooler直接经历了dense和activation。我们这边不用。都放在外面。
    (输入的都是一个batch的tensor)
    """
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        """将BERT输出每个词的序列的表示中，提取出给定位置的表示

        Args:
            hidden_states ([type]): [batch, 128, 768]
        Returns:
            [type]: [description]
        """
        first_token_tensor = hidden_states[:, 0]
        return first_token_tensor


class ModelInit():
    def __init__(self, args) -> None:
        self.args = args

    def init_config(self, config_class, num_labels, pseudo_hidden_dropout_prob=None):
        """
        Init/Load config/model/tokenizer
        """
        config = config_class.from_pretrained(
            self.args.config_name if self.args.config_name else self.args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=self.args.dataset_name,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
            pseudo_hidden_dropout_prob=pseudo_hidden_dropout_prob,
        )
        return config
    
    def init_tokenizer(self, tokenizer_class, never_split=None):
        tokenizer = tokenizer_class.from_pretrained(
            self.args.tokenizer_name if self.args.tokenizer_name else self.args.model_name_or_path,
            do_lower_case=self.args.do_lower_case,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
            never_split=never_split
        )
        return tokenizer

    def init_model(self, model_class, config):
        model = model_class.from_pretrained(
            self.args.model_name_or_path,
            from_tf=bool(".ckpt" in self.args.model_name_or_path),
            config=config,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )
        if self.args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        # put the model to GPU/CPU
        model.to(self.args.device)
        return model



if __name__ == "__main__":
    
    pass

    