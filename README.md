# TOPICAL 

Topical introduces a deep learning model that can automatically tag source code repositories with semantic labels. Topical incorporates an embedding mechanism that projects the textual information, the full dependencies graph, and the source code structure into a common representational embedding that exploits the synergies between these domains. 

## GitHub Crawler
We develop a GitHub Crawler combining the official API with page scraping for additional metadata (git commits, repository tree, etc.). GitHub repositories are often classified by its owner using hand-picked topics, which can contain abbreviation, typos, and repetitions. Because of the large variations in topic names, GitHub also defines 480 featured topics, a limited number of predefined topics to be associated with the repository by its owner. In order to have a unique label for each category, the crawler maps the non-featured topics (hand-picked) to the GitHub featured topics using partial tokens matching methods relying on Levenshtein distance.

To scrap the dataset, execute this command line from the `crawler` directory : 
<html>
<code>
python api\github_crawler.py --topics topics.json</code>
</html>

after ensuring that the `crawler` directory contains a JSON file with the topics to scrap (either as a simple list or as a nested directory). The above topics.json is just an example of how to add topics for the crawler. 

Please see examples/featured_topics.json for our the tags used for our specific use case and also examples/dataset.zip for the zip of the dataset used also. As stated above, you can re-create the 20 topics and dataset using featured_topics.json

<html>
<code>
python api\github_crawler.py --topics examples/featured_topics.json</code>
</html>


## Maintenance Level
This repository is maintained to fix bugs and ensure the stability of the existing codebase. However, please note that the team does not plan to introduce new features or enhancements in the future.


## Script Hybrid Embedding Generation

Once your dataset is ready, we want to generate the `DataFrame` containing the repositories informations and script embeddings
To do so, execute this command from the `Topical` folder, making sure your dataset is in a `dataset` folder in the `Topical` folder:
<html><code>python compute_embeddings.py
</code>
</html>

## Base with Classification Head

Once the `DataFrame` containing all the informations has been generated, you can simply execute the Topical classification head, from the same `Topical` directory:
<html>
<code>
python run.py
</code>
</html>

To change one of the training paramater, display all the available arguments from the `Topical` directory:
<html>
<code>
python run.py --help
</code>
</html>
