# 大数据前沿技术课程作业

```
TransR:
initialize entity and relation embeddings with results of TransE,
and initialize relation matrices as identity matrices.

结果
Model: TransE | WN18
mean_rank_raw         239 | mean_rank_filter    228
hit10_raw          61.43% | hit10_filter     69.65%
classification_acc 95.51%
Model: TransE | FB15K
mean_rank_raw         353 | mean_rank_filter    276
hit10_raw          28.22% | hit10_filter     33.84%
classification_acc 68.38%
Model: TransH | WN18
mean_rank_raw         189 | mean_rank_filter    178
hit10_raw          63.90% | hit10_filter     72.19%
classification_acc 95.54%
Model: TransH | FB15K
mean_rank_raw         292 | mean_rank_filter    214
hit10_raw          31.24% | hit10_filter     37.23%
classification_acc 70.25%
Model: TransR | WN18
mean_rank_raw         230 | mean_rank_filter    219
hit10_raw          61.60% | hit10_filter     69.74%
classification_acc 95.97%
Model: TransR | FB15K
mean_rank_raw         247 | mean_rank_filter    152
hit10_raw          31.57% | hit10_filter     38.29%
classification_acc 73.58%
```
