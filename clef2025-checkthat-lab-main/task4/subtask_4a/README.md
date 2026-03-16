# Subtask 4a (Scientific Web Discourse Detection)

Given a social media post (tweet), detect if it contains (1) a scientific claim, (2) a reference to a scientific study / publication, or (3) mentions of scientific entities, e.g. a university or scientist.

The task is a multilabel classification task.

__Table of contents:__

- [List of Versions](#list-of-versions)
- [Contents of the Task 4a Directory](#contents-of-the-subtask4a-directory)
- [Datasets statistics](#datasets-statistics)
- [Input Data Format](#input-data-format)
- [Output Data Format](#output-data-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Baselines & Scorers](#baselines--scorers)
- [Submission](#submission)
- [Related Work](#related-work)
- [Contact](#contact)
- [Credits](#credits)


## List of Versions
- [20/01/2025] Training data released.
- [24/03/2025] Initial training data was splitted into ct_train.tsv and ct_dev.tsv (90/10).

## Contents of the Subtask4a Directory

This repository contains the following files:

1. **baselines.ipynb** Jupyter notebook that enables participants of subtask 4a to quickly get started (includes training a microsoft/deberta-v3-base model and evaluation on the dev set)
2. **ct_train.tsv** The training dataset (tweet texts + ground-truth labels)
3. **ct_dev.tsv** The development dataset (tweet texts + ground-truth labels)
4. **README.md** this file

## Datasets statistics
The training dataset **ct_train.tsv** contains 1,229 tweets + ground-truth labels. 
The development dataset **ct_dev.tsv** contains 137 tweets + ground-truth labels. 

## Input Data Format

The tweets are provided as a .tsv file with three columns 
> index <TAB> text <TAB> labels

Where: <br>
* index: index for the data samples (used for internal purposes)
* text: the preprocessed tweet text
* labels: a list of three labels, one for each category (e.g., [0.0, 0.0, 1.0])

**Example:**
See examples at: https://checkthat.gitlab.io/clef2025/task4/#subtask-4a-dataset-examples

(user mentions are anonymized with @user, tweets that are quotes contain the word "quote" at the end of the text, tweets that contain image(s) contain the word "image" at the end of the text) 
## Output Data Format

The output must be a .csv file named predictions.csv with four columns: index, cat1_pred, cat2_pred, and cat3_pred.<br>
When making a submission, make sure to zip your file to predictions.zip first (see Submission Guidelines)!

## Evaluation Metrics

This task is a multilabel classification task and will be evaluated using the macro-averaged F1-score.

## Baselines & Scorers

The **baselines.ipynb** jupyter notebook can be found in this repository. Participants can download it and use it to get started quickly. The notebook contains:

1. Code to load the data from ct_train.tsv and ct_dev.tsv
2. Code to train a microsoft/deberta-v3-base model on ct_train.tsv
3. Code to evaluate the model on ct_dev.tsv
4. Code to create a correctly formatted predictions.csv submission file


## Submission

Submissions need to be made through the CodaLab [competition](https://codalab.lisn.upsaclay.fr/competitions/22355).

The Development Phase is running and submissions containing predictions for the ct_dev.tsv file can be submitted.<br>
The final Evaluation Phase will start on April 30, 2025.

### Submission Guidelines
- Each team must create only one account in CodaLab and submit their predictions exclusively through that account.
- Make sure your account name matches that used during CLEF registration.
- The last submission will be considered as the final submission!
- The name of the file containing the predictions (for both Development and Evaluation phase) must be predictions.csv
- The predictions.csv file must contain the following columns: "index", "cat1_pred", "cat2_pred", and "cat3_pred".
- You must zip the predictions.csv to predictions.zip before you upload your submission to CodaLab.
- You are allowed to submit max 50 submissions per day per team.

## Related Work

Information regarding the three categories can be found in the original [publication](https://dl.acm.org/doi/10.1145/3511808.3557693)

## Contact
Please contact sebastian.schellhammer@gesis.org

## Credits
Please find it on the task website: https://checkthat.gitlab.io/clef2025/task4/
