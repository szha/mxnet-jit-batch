```
usage: main.py [-h] [--data DATA] [--word_embed WORD_EMBED]
               [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--lr LR] [--wd WD]
               [--optimizer OPTIMIZER] [--seed SEED] [--use-gpu] [--fold]
               [--inference-only]

TreeLSTM for Sentence Similarity on Dependency Trees

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           path to raw dataset. optional
  --word_embed WORD_EMBED
                        directory with word embeddings. optional
  --batch-size BATCH_SIZE
                        training batch size per device (CPU/GPU). (default:
                        256)
  --epochs EPOCHS       number of total epochs to run. (default: 20)
  --lr LR               initial learning rate. (default: 0.02)
  --wd WD               weight decay factor. (default: 0.0001)
  --optimizer OPTIMIZER
                        optimizer (default: adagrad)
  --seed SEED           random seed (default: 123)
  --use-gpu             enable the use of GPU.
  --fold                enable the use of fold for dynamic batching.
  --inference-only      run in inference-only mode.
```
