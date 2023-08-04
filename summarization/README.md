# Repository-level Summarization

This task generates a repository-level description of the source-code using GraphCodeBERT embeddings at script level,
RNNs combined with Attention mechanism to compute a repository-level embedding and Beam Search to generate a sentence 
using the GraphCodeBERT available vocabulary.
This model can be enriched by using the Hybrid script-level embedding. 

## Generate data (in common with Topical model)

### Crawl dataset
See Crawler documentation 

### Generate embeddings 
See Topical documentation

## Run the summarization task
Necessary files:
- `attention_model.py` : original summarization model adapted to our Attention mechanism to act a repository level instead of a script level
- `attention_module.py` : Topical Attention mechanism to compute a single embedding from a sequence of embeddings
- `attention_run.py` : script to run the training and testing phase of the Summarization task at a repository level, using our Attention Mechanism in the Summarization model
### 1. Verify all the default values of the arguments in the `attention_run.py` in the argument parser
All the model's hyperparameters are left to be set in the Argument Parser

### 2. Run the model
Run the main folder `senatus-code-s4` run:

`python summarization/attention_run.py --do_train --do_test --do_eval`, and additionnal arguments if needed (see 1.)

### Visualize the results
The evaluation and test parts of the training use a Beam Search algorithm to generate descriptions
The results can be found in the following folder:

`senatus-code-s4/summarization-results`

in `.txt` files, the `gold` suffix corresponding to the groundtruth and `output` suffix corresponding to the Beam Search result. 

### 3. Interesting tests to run

### About v. README
A project-level description can be found for each repository in its `README.md` file but also in the About column scrapped by our Crawler from the GitHub page of the project.
The flag `--readme` will set the groundtruth as the README file, else it is set by default to the about section.

### Load a checkpoint
Use the `--load_model_path` flag followed by the path to the folder containing the `.bin` of the checkpoint you want to load.

### BASELINE : Script summarization + Extractive summarization
Microsoft original model acts at a snippet/script level, we adapted their model to our pipeline including our docstring preprocessing
Necessary files:
- `model.py`: Original architecture for script-level summarization using GraphCodeBERT base embedding
- `run.py`: Original script to run the training and testing phase of the Summarization task at a script-level including dosctring cleaning (with clustering technique to remove undesired Credential and other noisy strings.)
- `clustering_tools.pkl`/`Clustering on docstrings.ipynb` : notebook to fit the clustering parameter and save it in a pickle containing the fitted model to be used in the DataLoader. 