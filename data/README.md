# Data

The polarity dataset v2.0 (Pang and Lee, 2004) is **not redistributed here**; its licence terms
have not yet been confirmed by the authors. Obtain it from NLTK:

```bash
python3 -c "import nltk; nltk.download('movie_reviews', download_dir='nltk_data')"
mkdir -p data/mr && unzip -q nltk_data/corpora/movie_reviews.zip -d data/mr
sha256sum nltk_data/corpora/movie_reviews.zip
# expected: a41211ae685019137410268134db6a1a14428c89b671eb83056151a878539008
```

`exp/ml.py` reads the reviews from `data/mr/movie_reviews/{neg,pos}/`.

SST-2 (GLUE) and the models `distilbert-base-uncased` and `Qwen/Qwen2.5-0.5B` are downloaded from
the Hugging Face Hub by `gpu_kit/p3gpu.py` at run time.
