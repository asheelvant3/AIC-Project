AIC PROJECT Spring 2023

### Overview

daisyRec is a Python toolkit developed for benchmarking top-N recommendation task. The name DAISY stands for multi-**D**imension f**A**ir compar**I**son for recommender **SY**stem. 

<!---
***[DaisyRec v2.0](https://github.com/recsys-benchmark/DaisyRec-v2.0) is aviable now with more functions, and you are highly recommended to use DaisyRec v2.0***
--->

### How to Run

Make sure you have a **CUDA** enviroment to accelarate since the deep-learning models could be based on it. 
You can also run the notebook you like on Google Collab.
Clone this repository and zip the folder to upload on Collab

#### 1. Install from pip
```
!unzip AIC-Project.zip
```
```
pip install daisyRec
```

```
pip install optuna
```

#### 2. Run the tune and test
Command to tune for EASE:
```
%%shell
python /content/daisyRec/run_examples/tune.py --optimization_metric=ndcg --hyperopt_trail=20 --algo_name=ease --dataset=ml-100k --prepro=origin --topk=2 --epochs=50 --test_size=0.2 --val_size=0.1 --cand_num=1000 --test_method=tsbr --val_method=tsbr --tune_pack='{"reg": {"min": 10, "max": 1000, "step": null}}'
```

Command to test for EASE:
Remember to vary the regularisation term according to your tuning
```
%%shell
python /content/daisyRec/run_examples/test.py --algo_name=ease --dataset=ml-100k --prepro=origin --topk=5 --epochs=50 --test_size=0.2 --val_size=0.1 --cand_num=100 --test_method=tsbr --val_method=tsbr --reg=50
```
Command to tune for Multi-Vae:
```
%%shell
python /content/daisyRec/run_examples/tune.py --optimization_metric=ndcg --hyperopt_trail=20 --algo_name=multi-vae --dataset=ml-100k --prepro=origin --topk=50 --epochs=50 --test_size=0.2 --val_size=0.1 --cand_num=1000 --gpu=0 --init_method=default --optimizer=default --test_method=tsbr --val_method=tsbr --tune_pack='{"batch_size": {"min": 128, "max": 512, "step": null}, "latent_dim": {"min": 64, "max": 256, "step": null}, "dropout": {"min": 0.1, "max": 0.9, "step": null}, "lr": {"min": 0.001, "max": 0.01, "step": null}, "anneal_cap": {"min": 0.1, "max": 1, "step": null}}'
```

Command to test for Multi-Vae:
```
%%shell
python /content/daisyRec/run_examples/test.py --algo_name=multi-vae --dataset=ml-100k --prepro=origin --topk=50 --epochs=50 --test_size=0.2 --val_size=0.1 --cand_num=1000 --gpu=0 --init_method=default --optimizer=default --test_method=tsbr --val_method=tsbr --batch_size=128 --latent_dim=128 --dropout=0.5 --lr=0.01 --anneal_cap=0.2
```

### Documentation 

The documentation of DaisyRec is available [here](https://daisyrec.readthedocs.io/en/latest/), which provides detailed explanations for all arguments.

### Datasets

You can download experiment data, and put them into the `data` folder.
All data are available in links below: 

  - MovieLens-[100K](https://grouplens.org/datasets/movielens/100k/) / [1M](https://grouplens.org/datasets/movielens/1m/) / [10M](https://grouplens.org/datasets/movielens/10m/) / [20M](https://grouplens.org/datasets/movielens/20m/)
  - [Netflix Prize Data](https://archive.org/download/nf_prize_dataset.tar)
  - [Amazon-Book/Electronic/Clothing/Music ](http://jmcauley.ucsd.edu/data/amazon/links.html)(ratings only)
