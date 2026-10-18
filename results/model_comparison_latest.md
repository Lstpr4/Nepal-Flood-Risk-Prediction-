# Model comparison (latest runs)

Same temporal CV, horizon, and alert settings per script.

| model         | macro_roc_auc   | macro_f1   | macro_precision   | macro_recall   | micro_roc_auc   | micro_f1   | micro_precision   | micro_recall   |   false_alarm_rate |
|:--------------|:----------------|:-----------|:------------------|:---------------|:----------------|:-----------|:------------------|:---------------|-------------------:|
| XGBoost       | 95.5%           | 72.8%      | 80.2%             | 72.8%          | 93.3%           | 65.7%      | 83.6%             | 54.2%          |           0.164286 |
| Random Forest | 95.8%           | 71.8%      | 78.8%             | 72.7%          | 95.0%           | 67.7%      | 83.1%             | 57.1%          |           0.169231 |
| PyTorch LSTM  | 92.0%           | 55.4%      | 62.8%             | 66.7%          | 89.4%           | 52.6%      | 60.1%             | 46.8%          |           0.399491 |

CSV: `model_comparison_latest.csv`