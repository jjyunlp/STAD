"""
分析超参数max_label_num对性能的影响
在SemEval的开发集上。
"""
 
import json
import matplotlib.pyplot as plt
import numpy as np


colors = ['black', 'red', 'green', 'blue', 'yellow']
label_data_size = ['100', '200', '300', '400']
merge_dev_f1 = [54.79, 70.90, 74.19, 75.06]
merge_test_f1 = [56.80, 72.76, 75.38, 76.75]
multi_task_test_f1 = [57.17, 68.43, 72.01, 72.62]
two_stage_dev_f1 = [58.85, 71.87, 74.78, 75.42]
two_stage_test_f1 = [60.01, 73.55, 76.25, 76.75]

pseudo_acc = [67.5, 86.2, 90.2, 91.6]
pseudo_acc_str = ['67.5', '86.2', '90.2', '91.6']

label_num = [1, 3, 5, 7, 9]
label_num = [1, 2, 3, 4, 5, 6, 7, 8, 9]

base_f1 = [73.5, 73.5, 73.5, 73.5, 73.5]
easy_f1 = [81.5, 81.5, 81.5, 81.5, 81.5]
ambig1 = [81.5, 80.4, 78.9, 78.5, 78.1]
ambig2 = [81.5, 82.0, 82.5, 82.4, 82.6]
two_stage =[82.7, 83.2, 83.9, 82.7, 83.2]

ambig1 = [81.5, 81.3, 80.4, 80.1, 78.9, 78.7, 78.5, 78.2, 78.1]
ambig2 = [81.5, 81.0, 82.0, 82.2, 82.5, 82.1, 82.4, 82.5, 82.4]
two_stage = [82.7, 83.1, 83.2, 82.8, 83.9, 83.8, 82.7, 82.9, 83.2]



# 创建画布
plt.figure(figsize=(8, 6))
plt.xlabel('Number of Ambiguous Labels', fontsize=14)
plt.ylabel('Micro F1(%)', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.plot(label_num, merge_test_f1, marker='^', label="Merge", color='green')
#plt.plot(label_num, multi_task_test_f1, marker='o', label="Multi-Task", color='blue')
#plt.plot(label_num, two_stage_test_f1, marker='v', label="Two-Stage", color='red')

#plt.plot(label_num, base_f1, )


plt.plot(label_num, two_stage, marker='o', label="Easy+Ambig $\Rightarrow$ Gold")
plt.plot(label_num, ambig2, marker='v', label="Gold+Easy+Ambig")
plt.plot(label_num, ambig1, marker='^', label="Gold+Easy+Ambig*")
plt.hlines(81.5, 1, 9, linestyles="dashdot", label="Gold+Easy")
plt.hlines(73.5, 1, 9, linestyles="dashed", colors='red', label="Gold")
plt.ylim(72.5, 92.5)
plt.legend(fontsize=12)
# 保存图片到本地
plt.savefig('./max_ambiguous_label_num_analysis.png')
plt.show()
'''绘制第一条数据线
1、节点为圆圈
2、线颜色为红色
3、标签名字为y1-data
'''

'''绘制第二条数据线
1、节点为五角星
2、线颜色为蓝色
3、标签名字为y2-data
'''
'''
plt.plot(x, y_2, marker='*', color='b', label='y2-data')

# 显示图例（使绘制生效）
plt.legend()

# 横坐标名称
plt.xlabel('x_label')

# 纵坐标名称
plt.ylabel('y_label')

# 保存图片到本地
plt.savefig('pci.png')

# 显示图片
plt.show()
'''