## Files and the uses:

Model training pipelines:
    - model_pipeline_agnews.ipynb  : model training pipeline using agnews dataset 
    - model_pipeline_imdb.ipynb  : model training pipeline using imdb dataset

Trained Models:
     final model (after phase 3 of training):
        - ag_news_distilbert_hard/
        - imdb_distilbert_hard/

csv files generated using test dataset  containing  :  initial sentence, remaining sentence in final layer after pruning(after removal of stopwords) , true label,  predicted labels , pos_tags of the remaining sentence
    - cleaned_agnews_pruned.csv
    - cleaned_imdb_pruned.csv

Results and graphs:
    - results_agnews/
    - results_imdb/