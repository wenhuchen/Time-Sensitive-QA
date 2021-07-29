# Time-Sensitive-QA
The repo for Time-Sensitive Question Answering


## BigBird
Extractive QA baseline model
### Running Training
```
    python -m BigBird.main model_id=triviaqa cuda=[DEVICE] mode=train per_gpu_train_batch_size=2 n_gpu=4
```

### Running Evaluation
```
    python -m BigBird.main model_id=nq datasets=hard mode=eval cuda=[DEVICE] per_gpu_train_batch_size=8 model_path=[YOUR_MODEL]
```
