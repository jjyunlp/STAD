from collections import Counter
import numpy as np


# 写一个完整的实验结果统计，分析，甚至比较的类
# 包括，每个关系类别的情况，而且能够调用语料得到错误的例子
# 两组输入，相互的比较


class EvaluationAndAnalysis():
    r"""
    基本功能：
    （1）根据结果输出P,R,F1
    （2）调出错误的例子
    """
    def __init__(self):
        print("Evaluation and Analysis")


    def compute_metrics(self, task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "tacred" or task_name == "ds":
            return {"tacred": self.micro_f1_for_tacred(labels, preds)}
        elif task_name == "re-tacred":
            return {"re-tacred": self.micro_f1_for_tacred(labels, preds)}
        elif task_name == "semeval":
            return {"semeval": self.macro_f1_for_semeval(labels, preds)}
        else:
            raise KeyError(task_name)

    def micro_f1_exclude_NA(self, key, prediction, verbose=True):
        """
        用于SemEval-2010 and TACRED
        通用的micro f1，第一个位置为NA，即不放入评测的一个
        has_NA: 预定义的关系中是否有NA，有的话评测时需要忽略NA相同的情况，没有的话所有关系都要参与评测
        输入输出分别是什么，都写下，时间一久什么都忘了
        evaluate metric for TACRED
        micro-averaged F1 score or micro-F1
        先计算micro-precision and micro-recall，然后在计算micro-F1 = 2*p*r/(p+r)
        而：micro-p and micro-r都是计算所有预测正确与所有标注的，以及所有预测的。
        但是，排除掉原本是no_relation，预测也是no_relation的情况
        """
        # NO_RELATION = "no_relation"
        NO_RELATION = 0
        correct_by_relation = Counter()
        guessed_by_relation = Counter()
        gold_by_relation = Counter()

        # Loop over the data to compute a score
        for row in range(len(key)):
            gold = key[row]
            guess = prediction[row]

            if gold == NO_RELATION and guess == NO_RELATION:
                pass
            elif gold == NO_RELATION and guess != NO_RELATION:
                guessed_by_relation[guess] += 1
            elif gold != NO_RELATION and guess == NO_RELATION:
                gold_by_relation[gold] += 1
            elif gold != NO_RELATION and guess != NO_RELATION:
                guessed_by_relation[guess] += 1
                gold_by_relation[gold] += 1
                if gold == guess:
                    correct_by_relation[guess] += 1

        # Print verbose information
        rel2results = None
        if verbose:
            # 这边relation都是整数
            # print("Per-relation statistics:")
            relations = gold_by_relation.keys()
            longest_relation = 0
            for relation in sorted(relations):
                longest_relation = max(len(str(relation)), longest_relation)
            rel2results = {}
            for relation in sorted(relations):
                # (compute the score)
                correct = correct_by_relation[relation]
                guessed = guessed_by_relation[relation]
                gold = gold_by_relation[relation]
                prec = 1.0
                if guessed > 0:
                    prec = float(correct) / float(guessed)
                recall = 0.0
                if gold > 0:
                    recall = float(correct) / float(gold)
                f1 = 0.0
                if prec + recall > 0:
                    f1 = 2.0 * prec * recall / (prec + recall)
                rel2results[str(relation)] = {
                    'p': round(prec, 5),
                    'r': round(recall, 5),
                    'f1': round(f1, 5), 
                }

                # (print the score)
                """
                sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
                sys.stdout.write("  P: ")
                if prec < 0.1: sys.stdout.write(' ')
                if prec < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(prec))
                sys.stdout.write("  R: ")
                if recall < 0.1: sys.stdout.write(' ')
                if recall < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(recall))
                sys.stdout.write("  F1: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(f1))
                sys.stdout.write("  #: %d" % gold)
                sys.stdout.write("\n")
                """
            print("")

        micro_p = 1.0
        if sum(guessed_by_relation.values()) > 0:
            micro_p = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
        micro_r = 0.0
        if sum(gold_by_relation.values()) > 0:
            micro_r = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
        micro_f1 = 0.0
        if micro_p + micro_r > 0.0:
            micro_f1 = 2.0 * micro_p * micro_r / (micro_p + micro_r)
        """
        print("Precision (micro): {:.3%}".format(prec_micro))
        print("   Recall (micro): {:.3%}".format(recall_micro))
        print("       F1 (micro): {:.3%}".format(f1_micro))
        """
        return {
            "micro_p": round(micro_p, 5),
            "micro_r": round(micro_r, 5),
            "micro_f1": round(micro_f1, 5),
            "verbose": rel2results,     # verbose=True, else None
        }

    def micro_f1(self, key, prediction, verbose=True):
        """
        所有relation都要评测(用于没有NA的数据，比如SemEval-2018 Task7)
        """
        # NO_RELATION = "no_relation"
        correct_by_relation = Counter()
        guessed_by_relation = Counter()
        gold_by_relation = Counter()

        # Loop over the data to compute a score
        for row in range(len(key)):
            gold = key[row]
            guess = prediction[row]

            gold_by_relation[gold] += 1
            if gold != guess:
                guessed_by_relation[guess] += 1
            else:
                correct_by_relation[guess] += 1
                guessed_by_relation[guess] += 1

        # Print verbose information
        rel2results = None
        if verbose:
            # 这边relation都是整数
            # print("Per-relation statistics:")
            relations = gold_by_relation.keys()
            longest_relation = 0
            for relation in sorted(relations):
                longest_relation = max(len(str(relation)), longest_relation)
            rel2results = {}
            for relation in sorted(relations):
                # (compute the score)
                correct = correct_by_relation[relation]
                guessed = guessed_by_relation[relation]
                gold = gold_by_relation[relation]
                prec = 1.0
                if guessed > 0:
                    prec = float(correct) / float(guessed)
                recall = 0.0
                if gold > 0:
                    recall = float(correct) / float(gold)
                f1 = 0.0
                if prec + recall > 0:
                    f1 = 2.0 * prec * recall / (prec + recall)
                rel2results[str(relation)] = {
                    'p': round(prec, 5),
                    'r': round(recall, 5),
                    'f1': round(f1, 5), 
                }

                # (print the score)
                """
                sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
                sys.stdout.write("  P: ")
                if prec < 0.1: sys.stdout.write(' ')
                if prec < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(prec))
                sys.stdout.write("  R: ")
                if recall < 0.1: sys.stdout.write(' ')
                if recall < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(recall))
                sys.stdout.write("  F1: ")
                if f1 < 0.1: sys.stdout.write(' ')
                if f1 < 1.0: sys.stdout.write(' ')
                sys.stdout.write("{:.2%}".format(f1))
                sys.stdout.write("  #: %d" % gold)
                sys.stdout.write("\n")
                """
            print("")

        micro_p = 1.0
        if sum(guessed_by_relation.values()) > 0:
            micro_p = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
        micro_r = 0.0
        if sum(gold_by_relation.values()) > 0:
            micro_r = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
        micro_f1 = 0.0
        if micro_p + micro_r > 0.0:
            micro_f1 = 2.0 * micro_p * micro_r / (micro_p + micro_r)
        """
        print("Precision (micro): {:.3%}".format(prec_micro))
        print("   Recall (micro): {:.3%}".format(recall_micro))
        print("       F1 (micro): {:.3%}".format(f1_micro))
        """
        return {
            "micro_p": round(micro_p, 5),
            "micro_r": round(micro_r, 5),
            "micro_f1": round(micro_f1, 5),
            "verbose": rel2results,     # verbose=True, else None
        }

    def macro_f1_for_semeval2010(self, labels, preds):
        """Evaluate metric for SemEval-2010.
        The official metric is macro-averaged F1-Score for (9+1)-way classification,
        taking directionality into account.
        macro: 算出各个category的F1后，取平均；
        对于precision和recall同理。
        也就是说：macro_precision, macro_recall和macro_f1之间没有联系。

        Args:
            labels ([list]): 预测的
            preds ([list]): 标注答案
        """
        pred_total = [0] * 19
        pred_correct = [0] * 19
        label_total = [0] * 19
        assert len(labels) == len(preds)
        for i, label in enumerate(labels):
            pred = preds[i]
            label_total[label] += 1
            if pred == label:
                pred_correct[label] += 1
            pred_total[preds[i]] += 1
        # 0=Others, i和i+1 是成对的关系，只不过反了下方向
        p_list = []
        r_list = []
        macro_f1 = []
        for index in range(1, len(label_total), 2):
            # 一个关系的两个方向合并，不过taking directionality into account.
            correct = pred_correct[index] + pred_correct[index + 1]
            total = pred_total[index] + pred_total[index + 1]
            if total != 0:
                p_list.append(correct/total)
            else:
                p_list.append(0.0)
            total = label_total[index] + label_total[index + 1]
            if total != 0:
                r_list.append(correct/total)
            else:
                r_list.append(0.0)
        for i in range(len(p_list)):
            if p_list[i] + r_list[i] == 0:
                macro_f1.append(0.0)
            else:
                p = p_list[i]
                r = r_list[i]
                f1 = 2 * p * r / (p + r)
                macro_f1.append(f1)
        p_list = np.asarray(p_list)
        r_list = np.asarray(r_list)
        macro_f1 = np.asarray(macro_f1)
        return{
            "macro_p": p_list.mean(),
            "macro_r": r_list.mean(),
            "macro_f1": macro_f1.mean(),
            "categoried_macro_f1": macro_f1.tolist()
        }

    def macro_f1_for_semeval2018(self, labels, preds):
        """Evaluate metric for SemEval-2018-Task7.
        The official metric is macro-averaged F1-Score for 6-way classification,
        macro: 算出各个category的F1后，取平均；
        对于precision和recall同理。
        也就是说：macro_precision, macro_recall和macro_f1之间没有联系。

        Args:
            labels ([list]): 预测的
            preds ([list]): 标注答案
        """
        pred_total = [0] * 6
        pred_correct = [0] * 6
        label_total = [0] * 6
        assert len(labels) == len(preds)
        for i, label in enumerate(labels):
            pred = preds[i]
            label_total[label] += 1
            if pred == label:
                pred_correct[label] += 1
            pred_total[preds[i]] += 1
        # 0=Others, i和i+1 是成对的关系，只不过反了下方向
        p_list = []
        r_list = []
        macro_f1 = []
        for index in range(len(label_total)):
            correct = pred_correct[index]
            total = pred_total[index]
            if total != 0:
                p_list.append(correct/total)
            else:
                p_list.append(0.0)
            total = label_total[index]
            if total != 0:
                r_list.append(correct/total)
            else:
                r_list.append(0.0)
        for i in range(len(p_list)):
            if p_list[i] + r_list[i] == 0:
                macro_f1.append(0.0)
            else:
                p = p_list[i]
                r = r_list[i]
                f1 = 2 * p * r / (p + r)
                macro_f1.append(f1)
        p_list = np.asarray(p_list)
        r_list = np.asarray(r_list)
        macro_f1 = np.asarray(macro_f1)
        return{
            "macro_p": p_list.mean(),
            "macro_r": r_list.mean(),
            "macro_f1": macro_f1.mean(),
            "categoried_macro_f1": macro_f1.tolist()
        }

    def macro_f1_for_top10_re_tacred(self, labels, preds):
        """Evaluate metric for SemEval-2018-Task7.
        The official metric is macro-averaged F1-Score for 6-way classification,
        macro: 算出各个category的F1后，取平均；
        对于precision和recall同理。
        也就是说：macro_precision, macro_recall和macro_f1之间没有联系。

        Args:
            labels ([list]): 预测的
            preds ([list]): 标注答案
        """
        pred_total = [0] * 10
        pred_correct = [0] * 10
        label_total = [0] * 10
        assert len(labels) == len(preds)
        for i, label in enumerate(labels):
            pred = preds[i]
            label_total[label] += 1
            if pred == label:
                pred_correct[label] += 1
            pred_total[preds[i]] += 1
        # 0=Others, i和i+1 是成对的关系，只不过反了下方向
        p_list = []
        r_list = []
        macro_f1 = []
        for index in range(len(label_total)):
            correct = pred_correct[index]
            total = pred_total[index]
            if total != 0:
                p_list.append(correct/total)
            else:
                p_list.append(0.0)
            total = label_total[index]
            if total != 0:
                r_list.append(correct/total)
            else:
                r_list.append(0.0)
        for i in range(len(p_list)):
            if p_list[i] + r_list[i] == 0:
                macro_f1.append(0.0)
            else:
                p = p_list[i]
                r = r_list[i]
                f1 = 2 * p * r / (p + r)
                macro_f1.append(f1)
        p_list = np.asarray(p_list)
        r_list = np.asarray(r_list)
        macro_f1 = np.asarray(macro_f1)
        return{
            "macro_p": p_list.mean(),
            "macro_r": r_list.mean(),
            "macro_f1": macro_f1.mean(),
            "categoried_macro_f1": macro_f1.tolist()
        }
