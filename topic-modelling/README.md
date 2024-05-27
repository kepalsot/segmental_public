# overview
The topic modeling phase aims to transform participant annotations of movie clips into a low-dimensional topic model, where probability distributions over model-generated topics are assigned to each timestamp in the original movie clip. The topic model is implemented with latent-dirichlet allocation (LDA) (1).  

# data sources 
Participant annotations: 
Each workbook is a collection of a single participant’s annotations, where each excel sheet corresponds to annotations of a single movie clip. Besides the annotation text, each sheet contains an annotation timestamp column. Each second has a unique annotation. 

# scripts 
## segmental_preprocessing.py
Takes in the raw annotation workbooks and sorts data into movie clip-specific aggregates of each participant’s annotations, saving the second-by-second annotation resolution while allowing a topic model to be generated over the corpus of all movie clip annotations. Text preprocessing is also performed:
Remove stopwords & non-alphanumeric tokens 
Option to keep mixed tokens (those with 1> alphanumeric character)
Lemmatize remaining tokens 


## segmental_modelling.py
Takes in aggregated annotations and applies a topic model (latent dirichlet allocation). The n_components hyperparameter (n-topics) was ultimately set to 100 after testing a range of topic counts. Inter-clip correlation was preserved at 100 topics while also capturing greater intra-clip correlation. 

## segmental_analyses.py 
Exploratory data analysis for topic modeling output. Takes segmental_modeling.py output & produces a series of covariance matrices to compare model with varying n topics chosen. 

#### reference 
1. www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
