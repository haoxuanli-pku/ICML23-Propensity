## Propensity Matters: Measuring and Enhancing Balancing for Recommendation

This is the official pytorch implementation of "Propensity Matters: Measuring and Enhancing Balancing for Recommendation" (2023 ICML)


## Overview
In this paper, we discuss the potential problems of the previously widely adopted metrics for learned propensities, and propose balanced-mean-squared-error (BMSE) metric for debiased recommendations. Based on BMSE, we propose IPS-V2 and DR-V2 as the estimators of unbiased loss, and theoretically show that IPS-V2 and DR-V2 have greater propensity balancing and smaller variance without sacrificing additional bias. We further propose a co-training method for learning balanced representation and unbiased prediction.

## Run the code

- For coat


```python
python IPS_V2.py --dataset coat
```


```python
python DR_V2.py --dataset coat
```


- For yahoo

```python
python IPS_V2.py --dataset yahooR3
```


```python
python DR_V2.py --dataset yahooR3
```


- For product


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
If you find this code useful for your work, please consider to cite our work as
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

