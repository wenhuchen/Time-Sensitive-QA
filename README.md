# Time-Sensitive-QA
The repo for Time-Sensitive Question Answering


## BigBird

Extractive QA baseline model

### Initialize from TriviaQA checkpoint
1. Running Training
```
    python -m BigBird.main model_id=triviaqa cuda=[DEVICE] mode=train per_gpu_train_batch_size=2
```

2. Running Evaluation
```
    python -m BigBird.main model_id=triviaqa datasets=hard mode=eval cuda=[DEVICE] per_gpu_train_batch_size=8 model_path=[YOUR_MODEL]
```

### Initialize from NQ checkpoint
1. Running Training
```
    python -m BigBird.main model_id=nq cuda=[DEVICE] mode=train per_gpu_train_batch_size=8
```

1. Running Evaluation
```
    python -m BigBird.main model_id=nq cuda=[DEVICE] mode=eval per_gpu_train_batch_size=8 n_gpu=1 model_path=[YOUR_MODEL]
```
