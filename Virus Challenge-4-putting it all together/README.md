## One model may not predict well so i will use separate models for predicting each of the three labels' components, virus, risk, and spreaders
###### I have done the following steps:
1. Load the data and prepare it
2. Train and evaluate five different models for each component (ensemble-based models have been used)
3. Select the most accurate model **automatically** and use it to predict the classes for the new unlabeled data
