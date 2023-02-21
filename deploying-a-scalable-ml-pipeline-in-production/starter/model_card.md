# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Use `RandomForestClassifier` with default parameters for classified problem.

## Intended Use

Classify employee's salary into `<=50K` and `>50K` based on some information.

## Training Data

Census Bureau dataset is used for training and evaluation.

## Evaluation Data

20% of original data is used for evaluation. `train_test_split` with `random_state=42` split data into training and evaluation dataset.

## Metrics
Metrics used for evaluate model are `Precision`, `Recall`, and `Fbeta`.
- Precision: 0.744
- Recall: 0.631
- Fbeta: 0.683

## Ethical Considerations

This project was used for educational purposes only.

## Caveats and Recommendations

Model hyperparameters and selected features are default. Maybe try tuning the hyperparameters and choosing better features to improve the model's performance.