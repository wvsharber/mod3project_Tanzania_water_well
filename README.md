# Predicting Water Well Pump Functionality in Tanzania

## Summary and Motivation
Easy access to water is one of the most influential ways to increase quality of living in rural areas. In order to identify pumps that are a good candidate for updating, repairing, or installing, we create a model that predicts existing pump functionality in communities across Tanzania. Our predictive model, which implements a consensus method based on 4 individual classifiers, is able to predict pump functionality with 82% accuracy. 

## Data
Our predictive model is trained on data from Taarifa, which aggregates the data from the Tanzania Ministry of Water. You may find and download the data here https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/ after signing up for the competition. The training data consists of over 50,000 records of Tanzanian water well pumps across the country and includes features about their location, when they were installed, who installed them, what kind of pump they are, etc., as well as their current functioning status - functional, non-functional, or functional but needs repair. 

## Model Development and Results
We trained four separate classifier models: Logistic Regression, K-Nearest Neighbor, Random Forest, and Gradient Boosted Forest. Individually, these models performed well, ~ 80% accuracy on the training data, but we were able to achieve slightly higher accuracy (82%) by combining the four models into a single Soft Voting Classifier, where the weighted consensus was calculated for each prediction. We support the use of this consensus model to avoid any biases the individual models may introduce.

## Future Directions
Our current model could be improved by appropriately dealing with the smallest class, functional but needs repair, which is underrepresented in our dataset. Given more time, we would implement methods to deal with this class imbalance. Given the potential of this model to improve quality of life for Tanzanian communities, we also suggest adding functionality to the usage of this model to prioritize pump replacement/installation in communities with larger populations or where there is not already a functioning well.