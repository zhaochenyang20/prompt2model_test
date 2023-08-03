The training script is in `./train/train_t5_experiment.py`.

You can just run `python control.py` in the same director to launch the training.

The generated datasets are in `./generation/generated_dataset`. They are `jp2python`, `normalization,` and `SQuAD`. Note that these three are generated, and each contains roughly 5000 examples. You can check this. 

The only thing you need to change is these three lines in the `train_t5_experiment.py`.

```python
TRAINED_MODEL_ROOT = Path("/home/chenyan3/result/trained_model")
TRAINED_TOKENIZER_ROOT = Path("/home/chenyan3/result/trained_tokenizer")
RESULT_PATH = Path(f"/home/chenyan3/result/{model_store_name}_{task_name}")
```

Change it to your own root.

`    if evaluate:` means we are doing an evaluation on the generated test set.

`    if realistic:` means we are using the real dataset to test our trained models.

For the `Trainer`, the parameters are in `hyperparameter_choices_keys`. You could see all the parameters in the definition of `train_model` function.