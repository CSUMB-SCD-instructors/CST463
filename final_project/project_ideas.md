# Final project ideas

Pick one of the problems below for your final project.  If you want, you can find another problem, but it will need to be identified and approved by the instructor on the first day of the project.

Things to consider:
- Some problems involve lots of training.  Does anyone on your team have a machine with a powerful GPU?  Be sure to think about this before starting.
- The Chollet text gives lots of examples of deep learning problems.
- The datasets listed for each problem are just an example.  If you find a better data set you can find it.
- The instructor must approve your problem and data set.


- Ideas
### Image problems:
- Classify x-ray images to see whether pneumonia is present
  - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  - If you choose this project, I expect you to use Grad-CAM (described in the Chollet text) or a similar technique to explain a diagnosis of pneumonia.
- Breast cancer detection from image
  - https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
- Image colorization
  - https://www.kaggle.com/datasets/shravankumar9892/image-colorization
  - creating a data set is easy, although RGB may not be the best way to represent color
- Image upscaling
  - creating a data set is easy
- Image restoration (noise removal)
  - https://paperswithcode.com/task/image-restoration#datasets
- Captcha solving
  - https://www.kaggle.com/datasets/fournierp/captcha-version-2-images
  - https://github.com/topics/captcha-dataset
- Predict full face image from face image with mask
  - easy to create a data set


### Text problems:
- Question-answering system
  - https://rajpurkar.github.io/SQuAD-explorer/
- Text summarization (research articles, news, such as trashy Daily Mail data)
  - https://paperswithcode.com/datasets?task=text-summarization&page=1
- Sentiment analysis (datasets available; 1 million tweets)
- Predict genre and year from movie plot text 
  - https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots
- Predict gender and other attributes from tweets
  - https://data.world/datasets/twitter
  - https://www.kaggle.com/datasets/kazanova/sentiment140
- Generate title from paper abstract
  - https://www.kaggle.com/datasets/Cornell-University/arxiv
- Train a word embedding model.
- Predict whether a news article is fake.  Data: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- Language translation
  - https://paperswithcode.com/datasets?task=machine-translation
  - I don't recommend this because it is by now a well-known problem.

### Hybrid problems:
- Image captioning (Flickr 8k data set)
- Generate text from bird image (Cub-200 data set)

### Other problems:
- Predict ELO rating from chess games (lichess.org data)
- Deep learning with audio
