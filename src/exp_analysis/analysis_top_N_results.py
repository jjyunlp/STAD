"""
Draw the figure of topN results for each
1) Baseline
2) Self-Training with Confident
3) Ours (partial labeling + negative training) 
"""

import matplotlib.pyplot as plt

# 折线图

topN = ['1', '2', '3', '4', '5']
base_f1 = [77.5, 89.1, 92.5, 94.7, 96.1]
st_f1 = [80.5, 89.3, 92.2, 94.0, 95.3]
ours_f1 = [81.6, 91.8, 94.2, 95.8, 96.8]

# exclude top1
#topN = [2, 3, 4, 5]
#base_f1 = [89.1, 92.5, 94.7, 96.1]
#st_f1 = [89.3, 92.2, 94.0, 95.3]
#ours_f1 = [91.8, 94.2, 95.8, 96.8]

plt.plot(topN, base_f1, 's-', color = 'r', label = 'SUPERVISED')
plt.plot(topN, st_f1, 'v-', color = 'b', label = 'SELF-TRAINING')
plt.plot(topN, ours_f1, '^-', color = 'g', label = 'Ours')

plt.xlabel('Top-N')
plt.ylabel('Micro-F1')
plt.legend(loc="best")

plt.show()
plt.savefig('topN_results_comparison.png')

