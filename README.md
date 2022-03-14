# SentenceBERT-for-topic-modeling
Embedding-based topic modeling system for news articles




This repository stores the code and documentation for the Master Thesis project - _"Design and evaluation of embedding-based topic modeling system for news articles".
The goal of the project was to design as fast and scalable topic tracking and detection system.  The system should be able to dynamcally create a new analytica dimensions for news articles - topics. 


Presentation explaining the project: \
[<img src="./img/video.png" data-canonical-src="./img/video.png" width="600" />](https://www.youtube.com/watch?v=StTqXEQ2l-Y "Talk")

## Colab Examples

## Data
The data was collected using RSS Feed and simple scraping scripts inspired by [BildMining](https://github.com/Frank86ger/BildMining)
## Notebooks



## Analitical dashboard
In the `dash` folder you can see an inmplementation of a dashboard presetning the results of topic modeling combined with performance data.

  ```bash
  $ pip install -r requirements.txt
  $ pip install dash --upgrade
  $ python ./dash/app.py  
  ```

## Resources:
#### Articles:
- Papers with code: Sentence Emebddings - https://paperswithcode.com/task/sentence-embeddings

- SentenceBERT TLDR - https://medium.com/dair-ai/tl-dr-sentencebert-8dec326daf4e 
- BERTopics - https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
- Comprehesive overview of documents embeding techniques with lots of references and comparisons:  https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d
- https://supernlp.github.io/2018/11/26/sentreps/
- http://mlexplained.com/2019/01/07/paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained/
- Explanation of the SBERT: https://medium.com/genei-technology/richer-sentence-embeddings-using-sentence-bert-part-i-ce1d9e0b1343
- MLFLOW Setup :https://github.com/ymym3412/mlflow-docker-compose#3-Set-up-NGINX-Basic-Authentication


- Boosting the quality of embedding by simple prepossessing : https://github.com/vyraun/Half-Size

- Plotly and dash for NLP viz: https://medium.com/plotly/nlp-visualisations-for-clear-immediate-insights-into-text-data-and-outputs-9ebfab168d5b

- FAISS from Facebook AI - A library for efficient similarity search and clustering of dense vectors. - https://github.com/facebookresearch/faiss

#### Papers:
- PV for Vox Media + Wikipedia: 
- https://www.catalyzex.com/paper/arxiv:1208.4411
- https://www.sciencedirect.com/science/article/pii/S1532046416300442
- https://reader.elsevier.com/reader/sd/pii/S1532046416300442
- Effective Dimensionality reduction for word embeddings - https://www.aclweb.org/anthology/W19-4328/
- Text Summarization with pretrained encoders (can be used fror cluster description) - https://arxiv.org/abs/1908.08345 
