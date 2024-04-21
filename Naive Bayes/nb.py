import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self):
        self.class_priors = None
        self.num_feature_stats = {}
        self.cat_feature_stats = {}
        self.num_cols = []
        self.cat_cols = []

    def fit(self, X, y):
        self.class_priors = {}
        self.num_feature_stats = {}
        self.cat_feature_stats = {}

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        for c, count in zip(classes, counts):
            self.class_priors[c] = count / total_samples

        self.num_cols = list(X.select_dtypes(include=np.number).columns)
        self.cat_cols = list(X.select_dtypes(exclude=np.number).columns)

        for c in classes:
            self.num_feature_stats[c] = {
                'mean': X.loc[y == c, self.num_cols].mean(),
                'std': X.loc[y == c, self.num_cols].std()
            }

        for c in classes:
            cat_stats = {}
            for col in self.cat_cols:
                cat_stats[col] = X.loc[y == c, col].value_counts(normalize=True)
            self.cat_feature_stats[c] = cat_stats

    def predict(self, X):
        predictions = []

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        for _, x in X.iterrows():
            posteriors = {}
            for c in self.class_priors:
                prior = self.class_priors[c]

                num_likelihood = 0
                for col in self.num_cols:
                    mean = self.num_feature_stats[c]['mean'][col]
                    std = self.num_feature_stats[c]['std'][col]
                    num_likelihood += self.gaussian_pdf(x[col], mean, std)

                cat_likelihood = 0
                for col in self.cat_cols:
                    value = x[col]
                    if value in self.cat_feature_stats[c][col]:
                        cat_likelihood += np.log(self.cat_feature_stats[c][col][value])

                posterior = np.log(prior) + num_likelihood + cat_likelihood
                posteriors[c] = posterior

            predictions.append(max(posteriors, key=posteriors.get))

        return predictions

    def gaussian_pdf(self, x, mean, std):
        exponent = -((x - mean) ** 2) / (2 * (std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(exponent)