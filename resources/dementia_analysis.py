# -*- coding: utf-8 -*-
"""Defining functions and classes for classification of dementia features.

The contents of this module define functions and classes for cleaning,
visualizing, and making predictions based on clinical datasets collected by
the Open Access Series of Imaging Studies (OASIS) project.

Explore this repository at:
    https://github.com/chance-alvarado/dementia-classifier

Author:
    Chance Alvarado
        LinkedIn: https://www.linkedin.com/in/chance-alvarado/
        GitHub: https://github.com/chance-alvarado/
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, plot_roc_curve


class DataCleaning():
    """Class for DataFrame construction and cleaning."""

    def create_dataframes(self, cross_sectional_path, longitudinal_path):
        """Construct DataFrames for both datasets."""
        cs_df = pd.read_csv(cross_sectional_path)
        l_df = pd.read_csv(longitudinal_path)

        return [cs_df, l_df]

    def create_master_dataframe(self, cs_df, l_df):
        """Combine DataFrames into one master DataFrame."""
        # Grab only the first visit from l_df
        l_df = l_df[l_df.Visit == 1].copy(deep=True)

        # Drop different columns
        l_df.drop(columns=['Subject ID', 'MRI ID', 'Group',
                           'Visit', 'MR Delay', 'Hand'
                           ],
                  inplace=True
                  )
        cs_df.drop(columns=['ID', 'Delay', 'Hand'], inplace=True)

        # Rename 'EDUC' column for a proper merge
        l_df.rename(columns={'EDUC': 'Educ'}, inplace=True)

        # Construct master DataFrame
        df = pd.concat([cs_df, l_df], ignore_index=True, sort=False)

        # Reset index
        df.reset_index(drop=True, inplace=True)

        # Rename columns for ease of use
        df.rename(columns={'M/F': 'sex',
                           'Age': 'age',
                           'Educ': 'education',
                           'SES': 'economic_status',
                           'MMSE': 'mental_state_eval',
                           'CDR': 'dementia_rating',
                           'eTIV': 'intracranial_vol',
                           'nWBV': 'brain_vol',
                           'ASF': 'scaling_factor',
                           },
                  inplace=True
                  )

        return df

    def clean_master_dataframe(self, df):
        """Clean master DataFrame for visualization and prediction."""
        # Remove rows where no target information exists
        df = df[~df.dementia_rating.isnull()].copy(deep=True)

        # Convert sex info to binary: M=1, F=0
        df.sex.replace('F', 0, inplace=True)
        df.sex.replace('M', 1, inplace=True)

        # Fill na with column median
        df.fillna(df.median(), inplace=True)

        # Reset index
        df.reset_index(drop=True, inplace=True)

        # Add target column
        df['target'] = 1
        for i, val in enumerate(df.dementia_rating):
            if val == 0:
                df.at[i, 'target'] = 0

        return df


class DataVisualization():
    """Class for data exploration through visualization."""

    def __init__(self):
        """Itialize with proper themes."""
        self._blue = '#0834CB'
        self._red = '#B22222'
        self._main = '#ebeaf3'
        self._blue_opaque = '#b7bde5'
        self._red_opaque = '#ddbac2'

        sns.set_style("darkgrid")
        sns.set_palette(sns.color_palette([self._blue, self._red]), desat=.75)

    def donut_plot(self, df):
        """Create donut plot of sex breakdown."""
        # Manipulate data
        total_male = df[df.sex == 1].shape[0]
        dementia_male = df[df.sex == 1].target.sum()

        total_female = df[df.sex == 0].shape[0]
        dementia_female = df[df.sex == 0].target.sum()

        # Create plot
        fig, ax = plt.subplots(figsize=(7, 7))

        ax.pie([total_male, total_female], colors=['#C97276', '#90A1E3'],
               labels=['Male', 'Female'], shadow=True, radius=1,
               textprops={'fontsize': 14},
               wedgeprops=dict(width=.4, edgecolor=self._red),
               )

        wedges, texts = ax.pie([total_male - dementia_male, dementia_male,
                                total_female - dementia_female,
                                dementia_female
                                ],
                               colors=[self._blue_opaque, self._red_opaque,
                                       self._blue_opaque, self._red_opaque
                                       ],
                               shadow=True, radius=.7,
                               textprops={'fontsize': 14},
                               wedgeprops=dict(width=.15,
                                               edgecolor=self._blue
                                               ),
                               )

        # Plot attributes
        ax.legend(wedges[0:2], ['No Dementia', 'Dementia'],
                  loc="center left",
                  fontsize=14,
                  bbox_to_anchor=(.8, .9))

        fig.suptitle('Breakdown of Sexes', fontsize=18, x=.52, y=.9)

        # Show plot
        plt.show()

    def age_sex_kde_plot(self, df):
        """Create KDE plots of sex and age features."""
        # Create plot
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        sns.kdeplot(df[(df.sex == 1) & (df.target == 0)].age, shade=True,
                    ax=ax[0], legend=False)
        sns.kdeplot(df[(df.sex == 1) & (df.target == 1)].age, shade=True,
                    ax=ax[0], legend=False)

        sns.kdeplot(df[(df.sex == 0) & (df.target == 0)].age, shade=True,
                    ax=ax[1], legend=False)
        sns.kdeplot(df[(df.sex == 0) & (df.target == 1)].age, shade=True,
                    ax=ax[1], legend=False)

        # Plot attributes
        fig.suptitle('Kernal Density Estimates of Varying Populations',
                     fontsize=14)
        ax[0].set_xlabel('Male', fontsize=14)
        ax[1].set_xlabel('Female', fontsize=14)
        ax[1].legend(['No Dementia', 'Dementia'], fontsize=14,
                     bbox_to_anchor=(1, 1))

        # Show plot
        plt.show()

    def kde_plot(self, df):
        """Create KDE plot of relevant features."""
        # Create figure and specify subplots
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))

        # Attributes for all KDE plots
        plot_attrs = {'shade': True, 'legend': False}

        # Create KDE plots
        sns.kdeplot(df[df.target == 0].age, ax=ax[0, 0], **plot_attrs)
        sns.kdeplot(df[df.target == 1].age, ax=ax[0, 0], **plot_attrs)

        sns.kdeplot(df[df.target == 0].education, ax=ax[0, 1], **plot_attrs)
        sns.kdeplot(df[df.target == 1].education, ax=ax[0, 1], **plot_attrs)

        sns.kdeplot(df[df.target == 0].economic_status, ax=ax[0, 2],
                    **plot_attrs
                    )
        sns.kdeplot(df[df.target == 1].economic_status, ax=ax[0, 2],
                    **plot_attrs
                    )

        sns.kdeplot(df[df.target == 0].intracranial_vol, ax=ax[1, 0],
                    **plot_attrs
                    )
        sns.kdeplot(df[df.target == 1].intracranial_vol, ax=ax[1, 0],
                    **plot_attrs
                    )

        sns.kdeplot(df[df.target == 0].brain_vol, ax=ax[1, 1], **plot_attrs)
        sns.kdeplot(df[df.target == 1].brain_vol, ax=ax[1, 1], **plot_attrs)

        sns.kdeplot(df[df.target == 0].mental_state_eval, ax=ax[1, 2],
                    **plot_attrs
                    )
        sns.kdeplot(df[df.target == 1].mental_state_eval, ax=ax[1, 2],
                    **plot_attrs
                    )

        # Update plot attributes
        fig.suptitle('Kernal Density Estimates', fontsize=18)

        ax[0, 2].legend(['No Dementia', 'Dementia'], fontsize=14,
                        bbox_to_anchor=(1, 1))

        ax[0, 0].set_title('Age', fontsize=14)
        ax[0, 1].set_title('Years of Education', fontsize=14)
        ax[0, 2].set_title('Socioeconomic Status', fontsize=14)
        ax[1, 0].set_title('Estimated Intracranial Volume', fontsize=14)
        ax[1, 1].set_title('Normalized Whole Brain Volume', fontsize=14)
        ax[1, 2].set_title('Mini Mental State Evaluation', fontsize=14)

        # Show plot
        plt.show()

    def scatter_plot(self, df):
        """Create scatter plot of scaling factor and intracranial volume."""
        # Create plot
        sns.scatterplot(x='scaling_factor', y='intracranial_vol',
                        hue='target', data=df, alpha=0.4,
                        s=50, legend=False
                        )

        # Update plot attributes
        plt.title('Atlas Scaling Factor vs. Estimated Intracranial Volume',
                  fontsize=14)
        plt.xlabel('Atlas Scaling Factor', fontsize=12)
        plt.ylabel('Estimated Intracranial Volume', fontsize=12)

        # Show plot
        plt.show()
        plt.show

    def pair_plot(self, df):
        """Create pair plot to examine clustering of multiple features."""
        # Create grid and populate with plots
        g = sns.PairGrid(df[['age', 'intracranial_vol',
                             'brain_vol', 'mental_state_eval',
                             'target']],
                         hue='target', height=3.5, aspect=1.5,
                         )

        g = g.map_diag(plt.hist, alpha=0.5)
        g = g.map_offdiag(plt.scatter, alpha=0.5, s=80)

        # Update plot attributes
        labels = {'age': 'Age',
                  'intracranial_vol': 'Intracranial Volume',
                  'brain_vol': 'Whole Brain Volume',
                  'mental_state_eval': 'Mini Mental State Evaluation'
                  }

        for i in range(len(labels)):
            for j in range(len(labels)):
                xlabel = g.axes[i][j].get_xlabel()
                ylabel = g.axes[i][j].get_ylabel()
                if xlabel in labels.keys():
                    g.axes[i][j].set_xlabel(labels[xlabel], fontsize=20)
                if ylabel in labels.keys():
                    g.axes[i][j].set_ylabel(labels[ylabel], fontsize=20)

        g.fig.suptitle('Clustering between Features', fontsize=28, y=1.04)

        # Show plot
        plt.show()


class PredictiveModel():
    """Create multiple predictive models and find the best fit for the data."""

    def feature_target_split(self, df):
        """Seperate feature matrix and target vector from DataFrame."""
        # Labels
        labels = ['sex', 'age', 'education', 'economic_status',
                  'mental_state_eval', 'intracranial_vol', 'brain_vol'
                  ]

        # Feature matrix
        X = df[labels].to_numpy()

        # Target Vector
        y = df['target'].to_numpy()

        # Normalize the feature matrix
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)

        return [X, y]

    def split(self, X, y):
        """Split data into training and testing sets with an 70/30 split."""
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=.3,
                                                            random_state=1
                                                            )

        return [X_train, X_test, y_train, y_test]

    def logistic_regression(self, X_train, X_test, y_train, y_test):
        """Fit the data to a logistic regression classifier and test."""
        # Create and fit the model
        classifier = LogisticRegression(random_state=2)
        classifier.fit(X_train, y_train)

        # Print score on test data
        print('Logistic Regression Accuracy: ',
              classifier.score(X_test, y_test)
              )

        return classifier

    def random_forest(self, X_train, X_test, y_train, y_test):
        """Fit the data to a random forest classifier and test."""
        # Create and fit the model
        classifier = RandomForestClassifier(random_state=3)
        classifier.fit(X_train, y_train)

        # Print score on test data
        print('Random Forest Accuracy: ',
              classifier.score(X_test, y_test)
              )

        return classifier

    def svc(self, X_train, X_test, y_train, y_test):
        """Fit the data to an SVC classifier and test."""
        # Create and fit the model
        classifier = SVC(kernel='poly', C=0.5, random_state=4)
        classifier.fit(X_train, y_train)

        # Print score on test data
        print('SVC Accuracy: ',
              classifier.score(X_test, y_test)
              )

        return classifier


class ResultsVisualization():
    """Class for visualizing results from classifications models."""

    def __init__(self):
        """Initialize class with proper themes."""
        self._blue = '#0834CB'
        self._red = '#B22222'

        self._labels = ['Sex', 'Age', 'Education', 'Economic Status',
                        'Mental State Evaluation', 'Intracranial Volume',
                        'Brain Volume'
                        ]

        sns.set_style("darkgrid")
        sns.set_palette(sns.color_palette([self._blue, self._red]), desat=.75)

    def roc_plot(self, classifier, X_test, y_test):
        """Create ROC plot for given classifier."""
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        plot_roc_curve(classifier, X_test, y_test, alpha=0.6, ax=ax,
                       linewidth=3, c=self._blue
                       )
        ax.plot([0, 1], [0, 1], linestyle='dashed', linewidth=3,
                c=self._red, alpha=0.6
                )

        # Plot attributes
        ax.set_title('Receiver Operating Charecteristic Curve',
                     fontsize=18)

        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Postive Rate', fontsize=14)

        ax.get_legend().remove()

        # Show plot
        plt.show

    def feature_importance_plot(self, classifier):
        """Create bar plot of feature importances for given classifier."""
        # Manipulate data
        feature_importances = classifier.feature_importances_
        sort = np.argsort(feature_importances)

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.barh(np.array(self._labels)[sort], feature_importances[sort],
                color=self._blue, alpha=0.6
                )

        # Plot attributes
        ax.set_title('Feature Importances', fontsize=18)
        ax.tick_params(labelsize=14)

        # Show plot
        plt.show()

    def confusion_matrix(self, classifier, X_test, y_test):
        """Create confusion matrix for given classifier."""
        # Make predicitons from test features
        y_prediction = classifier.predict(X_test)

        # Create plot
        ax = plt.subplot()
        sns.heatmap(confusion_matrix(y_test, y_prediction),
                    annot=True, cmap=plt.get_cmap('Blues'),
                    alpha=0.8, ax=ax, fmt="g", cbar=False,
                    annot_kws={"size": 16}
                    )

        # Plot attributes
        ax.set_title('Confusion Matrix', fontsize=18)

        ax.set_xlabel('Predicted labels', fontsize=14)
        ax.set_ylabel('True labels', fontsize=14)

        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.xaxis.set_ticklabels(['No Dementia', 'Dementia'])
        ax.yaxis.set_ticklabels(['No Dementia', 'Dementia'])

        # Show plot
        plt.show()
