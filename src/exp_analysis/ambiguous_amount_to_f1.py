"""
生成teacher model noise问题的折线图
分析各个关系的ambiguous的数量与性能之间的关系
数量用柱状图，性能提升用折线图
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

ambiguous_amount = [192, 109, 110, 219, 133, 135, 94, 191, 170, 129]
improve_ratio = [7.6, 1.22, 1.26, 12.48, 4.58, -1.78, 0.06, 1.9, 2.96, -0.54]
relations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# new kim negative
improve_ratio = [6.72, 1.06, 0.26, 11.16, 3.06, 1.62, 0.24, 4.0, 3.02, -0.92]
improve_ratio = [5.28, 0.94, 0.22, 7.8, 2.5, 1.18, 0.18, 2.56, 2.24, -0.82]

relations = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']
relation_names = ['Entity-Destination(e1,e2)', 'Cause-Effect(e2,e1)', "Member-Collection(e2,e1)",
    "Entity-Origin(e1,e2)",
    "Message-Topic(e1,e2)",
    "Component-Whole(e2,e1)",
    "Component-Whole(e1,e2)",
    "Instrument-Agency(e2,e1)",
    "Product-Producer(e2,e1)",
    "Content-Container(e1,e2)"]

plt.figure(figsize=(8, 6))
plt.bar(relations, ambiguous_amount, label="Number of Ambiguous Data")
#plt.xticks(rotation=70)
plt.xlabel('Relation', fontsize=14)
plt.ylabel("Number", fontsize=14)
plt.ylim(0, 500)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc=1, fontsize=14)

plt.twinx()		# 	调用后可绘制次坐标轴
plt.plot(relations, improve_ratio, marker='o', label="Improvement")
plt.yticks(fontsize=40)
plt.ylim(-20, 20)
plt.ylabel("Rate(%)", fontsize=14)

#fig, ax = plt.subplots()
#fig.subplots_adjust(right=0.75)

#twin1 = ax.twinx()
# ax.axis(0, 9, 50, 100)
#p1, = ax.plot(relations, ambiguous_amount, label="number")
#p1 = ax.bar(relations, ambiguous_amount, label="number")
#p2, = twin1.plot(relations, improve_ratio, label="Improve")
#plt.ylim(-10, 15)

"""
# 创建画布
plt.figure(figsize=(8, 6))
plt.xlabel('Accuracy of 1000 Pseudo Examples(%)')
plt.ylabel('Test F1(%)' )
plt.plot(pseudo_acc_str, merge_test_f1, marker='^', label="Merge", color='green')
plt.plot(pseudo_acc_str, multi_task_test_f1, marker='o', label="Multi-Task", color='blue')
plt.plot(pseudo_acc_str, two_stage_test_f1, marker='v', label="Two-Stage", color='red')
"""
plt.yticks(fontsize=14)
plt.legend(loc=2, fontsize=14)
# 保存图片到本地
plt.savefig('./ambiguous_amount_and_improve_new.png')
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