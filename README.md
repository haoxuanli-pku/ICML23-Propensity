## Propensity Matters: Measuring and Enhancing Balancing for Recommendation

This is the official PyTorch implementation of "Propensity Matters: Measuring and Enhancing Balancing for Recommendation" (2023 ICML)


## Overview
In this paper, we discuss the potential problems of the previously widely adopted metrics and propose a novel balanced-mean-squared-error (BMSE) metric for learning propensity. Based on the BMSE, we propose two estimators named IPS-V2 and DR-V2 to estimate the ideal loss and theoretically show that IPS-V2 and DR-V2 have greater propensity balancing ability and smaller variance without sacrificing additional bias. We further propose a co-training method to co-train the propensity model and the prediction model to achieve unbiased prediction.

## Run the code

- For Coat dataset


```python
python IPS_V2.py --dataset coat
```


```python
python DR_V2.py --dataset coat
```


- For Yahoo! R3 dataset

```python
python IPS_V2.py --dataset yahooR3
```


```python
python DR_V2.py --dataset yahooR3
```


- For Product dataset


```python
python IPS_V2.py --dataset product
```


```python
python DR_V2.py --dataset product
```


The code runs well at python 3.8.18. The required packages are as follows:
-   pytorch == 1.9.0
-   numpy == 1.24.4 
-   scipy == 1.10.1
-   pandas == 2.0.3
-   scikit-learn == 1.3.2

## Reference
If you find this code useful for your work, please kindly consider to cite our work as
```
@inproceedings{li2023propensity,
  title={Propensity matters: Measuring and enhancing balancing for recommendation},
  author={Li, Haoxuan and Xiao, Yanghao and Zheng, Chunyuan and Wu, Peng and Cui, Peng},
  booktitle={International Conference on Machine Learning},
  pages={20182--20194},
  year={2023},
  organization={PMLR}
}
```

