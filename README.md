# embed-distill

Knowledge distillation of a large multilingual embedding model into a lightweight Persian BERT, so you get fast, high-quality Persian sentence embeddings without the cost of running a billion-parameter teacher at inference time.

## What it does

The project follows the classic teacher–student distillation recipe applied to sentence embeddings:

1. **Data** — Persian Wikipedia sentences are downloaded from Kaggle and split 95 / 5 into train and validation sets.

2. **Teacher embeddings** — [`jinaai/jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3) is used to pre-compute normalized embeddings for every training sentence. The process saves chunked `.npy` files and supports resuming mid-run, so it can survive Colab session resets. Embeddings can optionally be uploaded back to Kaggle for reuse.

3. **Student training** — [`HooshvareLab/bert-fa-base-uncased`](https://huggingface.co/HooshvareLab/bert-fa-base-uncased) (Persian BERT) is fine-tuned as the student. A linear projection layer maps the student's hidden dimension up to the teacher's embedding dimension. The combined loss is:
   - **Cosine loss** — pushes each student embedding toward its teacher counterpart.
   - **KL divergence** — aligns the in-batch pairwise similarity distributions between student and teacher (contrastive structure).

   Training uses AdamW with differential learning rates (`2e-5` for BERT, `5e-3` for the projection), a cosine annealing scheduler, mixed-precision (AMP), gradient clipping, and early stopping with patience.

4. **Evaluation** — The distilled model is benchmarked on [FarsTail](https://github.com/dml-qom/FarsTail), a Persian natural language inference dataset. Premise–hypothesis pairs are embedded and scored by cosine similarity; AUC-ROC is reported for three models:
   - Jina-v3 teacher (upper bound)
   - Raw `bert-fa-base-uncased` baseline (before distillation)
   - Distilled student (our result)

   The notebook prints the improvement over the baseline and the percentage of the teacher–baseline gap that was closed.

## Key config

| Parameter | Default |
|---|---|
| Teacher | `jinaai/jina-embeddings-v3` |
| Student | `HooshvareLab/bert-fa-base-uncased` |
| Batch size | 256 |
| Epochs | 30 (early stopping, patience 3) |
| Temperature | 0.05 |
| Student LR | 2e-5 |
| Projection LR | 5e-3 |

## Requirements

```
pip install -r requirements.txt
```

The notebook is designed to run on Google Colab with a GPU. A Kaggle account is needed to download the training data and, optionally, to cache the teacher embeddings.

