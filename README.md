# Codebase for NeurIPS "Hide-and-Seek Privacy Challenge" Competition 

Authors: James Jordon,Jinsung Yoon,  Mihaela van der Schaar

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Paper 

Contact: jsyoon0823@gmail.com

This directory contains implementations of NeurIPS Hide-and-Seek competition framework for generating private synthetic data (hider)
and reidentifying the original data (seeker) using a real-world dataset.

-   Stock data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG

To run the pipeline for training and evaluation on NeurIPS Hide-and-Seek competition framwork, simply run 
python3 -m main_hide-and-seek.py.


### Code explanation

(1) data
- public_data directory: public data to be released to the participants
- data_utils.py: train/test data division 

(2) hider
- add_noise: add Gaussian noise on the original data and use it as the synthetic data
- timegan: TimeGAN (Yoon et al., NeurIPS 2019) model for generating synthetic time-series data

(3) master (only for the master (competition owner))
- private_data: private data for evaluating submitted models
- raw_data: original data (public data U private data)
- data_release.py: data preprocessing code
- main_master.py: main file for the evaluation

(4) metrics
- general_rnn.py: general rnn models for evaluation
- metric_utils.py: feature prediction & one-step ahead prediction & reidentification score

(5) seeker
- knn: use the distance between original and synthetic data for reidentifying the real data

(6) main_hide-and-seek.py
- main file that competition participants (both hider and seeker) can use.

### Command inputs:

-   data_name: stock
-   train_rate: ratio of training data
-   feature_prediction_no: the number of features to be predicted for evaluation
-   seed: random seed for train / test data division
-   hider_model: timegan or add_noise
-   noise_size: size of the noise for add_noise hider

### Example command

```shell
$ python3 main_hide-and-seek.py --data_name stock --train_rate 0.8 
--feature_prediction_no 2 --seed 0 --hider_model timegan --noise_size 0.1 
```

### Outputs

-   feat_pred: feature prediction results (original & new) 
-   step_ahead_pred: step ahead prediction results (original & new)
-   reidentification_score: reidentification score between hider and seeker