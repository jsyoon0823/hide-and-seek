#!/usr/bin/env python3

import numpy as np
import os
import sys

from metrics.metric_utils import feature_prediction, one_step_ahead_prediction, reidentify_score


FEATURE_PREDICTION_NO = 2


def _score_hider(hider_output, scores_output):
    with np.load(hider_output) as data:
        train_data = data['train_data']
        test_data = data['test_data']
        generated_data = data['generated_data']
        enlarge_data = data['enlarge_data']
        enlarge_data_label = data['enlarge_data_label']

    with open(scores_output, "w") as output:
        # 1. Feature prediction
        feat_idx = np.random.permutation(train_data.shape[2])[:FEATURE_PREDICTION_NO]
        ori_feat_pred_perf = feature_prediction(train_data, test_data, feat_idx)
        new_feat_pred_perf = feature_prediction(generated_data, test_data, feat_idx)

        feat_pred = [ori_feat_pred_perf, new_feat_pred_perf]

        print('Feature prediction results: ' +
              '(1) Ori: ' + str(np.round(ori_feat_pred_perf, 4)) +
              '(2) New: ' + str(np.round(new_feat_pred_perf, 4)))

        print('Feature: {}'.format(new_feat_pred_perf[0]), file=output)

        # 2. One step ahead prediction
        ori_step_ahead_pred_perf = one_step_ahead_prediction(train_data, test_data)
        new_step_ahead_pred_perf = one_step_ahead_prediction(generated_data, test_data)

        step_ahead_pred = [ori_step_ahead_pred_perf, new_step_ahead_pred_perf]

        print('One step ahead prediction results: ' +
              '(1) Ori: ' + str(np.round(ori_step_ahead_pred_perf, 4)) +
              '(2) New: ' + str(np.round(new_step_ahead_pred_perf, 4)))

        print('One_step_ahead: {}'.format(new_step_ahead_pred_perf), file=output)


def _score_seeker(seeker_output, scores_output):
    with np.load(seeker_output) as data:
        train_data = data['train_data']
        test_data = data['test_data']
        generated_data = data['generated_data']
        enlarge_data = data['enlarge_data']
        enlarge_data_label = data['enlarge_data_label']
        reidentified_data = data['reidentified_data']

    with open(scores_output, "w") as output:
        # 3. Reidentification score
        reidentification_score = reidentify_score(enlarge_data_label, reidentified_data)

        print('Reidentification score: ' + str(np.round(reidentification_score, 4)))

        print('Reidentification: {}'.format(reidentification_score), file=output)


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    scores_output = os.path.join(output_dir, "scores.txt")

    seeker_output = os.path.join(input_dir, "res", "seeker_output.npz")
    if os.path.isfile(seeker_output):
        _score_seeker(seeker_output, scores_output)
    else:
        hider_output = os.path.join(input_dir, "res", "hider_output.npz")
        _score_hider(hider_output, scores_output)


if __name__ == "__main__":
    main()
