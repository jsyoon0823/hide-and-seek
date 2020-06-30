# Codebase for NeurIPS "Hide-and-Seek Privacy Challenge" Competition 

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: June 30th 2020
Code author: Jinsung Yoon
Contact: jsyoon0823@gmail.com

This directory contains implementations of NeurIPS 2020 Hide-and-Seek benchmarks and the evaluation framework. This includes 2 benchmarks for generating private synthetic data (hiders) and 2 for reidentifying the original data (seekers) using real-world datasets.

-   Amsterdam data: https://amsterdammedicaldatascience.nl/ (private data that needs approval through the website)
-   Stock data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG

To run the pipeline for training and evaluation on NeurIPS Hide-and-Seek competition framwork, simply run 
python3 -m main_hide-and-seek.py.

### Code explanation

(1) data
- public_data directory: this directory contains dummy public data to be used prior to data access being granted
- amsterdam_data directory: amsterdam data should be saved in this directory with the file name set to train_longitudinal_data.csv (main data)
- data_utils.py: used to divide data into train/test splits
- data_preprocess.py: data preprocessing tools for Amsterdam database

(2) hider
- add_noise: a simple model that adds Gaussian noise to the original data to create the synthetic data
- timegan: TimeGAN (Yoon et al., NeurIPS 2019) - model based on GANs to generate synthetic data

(3) master
- main_master.py: main file used in the competition backend to evaluate algorithms
- appropriate data needs to be saved as test_longitudinal_data.csv in data/amsterdam directory for main_master.py to run

(4) metrics
- general_rnn.py: general rnn models used to compute metrics
- metric_utils.py: feature prediction & one-step ahead prediction & reidentification score - all submissions will be ranked based on reidentification score, while hider submissions must meet a minimum threshold in feature and one-step ahead precition

(5) seeker
- knn: use the distance between original and synthetic data for reidentifying the real data
- binary_predictor: use a binary classifier to classify genereated data and the (enlarged) real data.
                    Reidentify by selecting the subset of the enlarged data with the highest classification scores (i.e. that is "most" mis-classified as generated).

(6) main_hide-and-seek.py
- main file that competition participants (both hider and seeker) can use to run the benchmarks against each other or their own models by substituting their model for the appropriate model.

### Command inputs:

-   data_name: amsterdam or stock
-   max_seq_len: maximum sequence length
-   train_rate: ratio of training data
-   feature_prediction_no: the number of features to be predicted for evaluation
-   seed: random seed for train / test data division
-   hider_model: timegan or add_noise
-   noise_size: size of the noise for add_noise hider
-   seeker_model: binary_predictor or knn

### Example command

```shell
$ python3 main_hide-and-seek.py --data_name amsterdam --max_seq_len 10
--train_rate 0.8 --feature_prediction_no 2 --seed 0 --hider_model timegan 
--noise_size 0.1 --seeker_model binary_predictor
```

### Outputs

-   feat_pred: feature prediction results (original & new) 
-   step_ahead_pred: step ahead prediction results (original & new)
-   reidentification_score: reidentification score between hider and seeker
