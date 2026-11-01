# Import the libraries needed for data handling, model training, evaluation, and plotting.
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Use a fixed seed so the same split and model training results are reproducible
# across runs. This is helpful when comparing different hyperparameter values.
RANDOM_STATE = 55


# Load the heart disease dataset from disk into a pandas DataFrame.
# The file contains both input features and the target column, HeartDisease.
# pd.read_csv() reads the CSV file and stores it in a table-like structure
# called a DataFrame, which makes it easy to inspect, filter, and analyze data.
df = pd.read_csv("Tree Ensembles\\data\\heart.csv")
print(df.head())

# These columns store categorical values, so we convert them to one-hot encoded
# numeric columns. This allows the decision tree to work with categorical data
# without manually mapping each category to integers.
cat_variables = ['Sex',
'ChestPainType',
'RestingECG',
'ExerciseAngina',
'ST_Slope'
]

# One-hot encoding creates a separate binary column for each category in the listed
# variables while preserving the remaining columns unchanged.
# pd.get_dummies() converts each categorical column into multiple 0/1 columns,
# allowing machine learning models to treat categories numerically without imposing
# a false ordering between categories.
df = pd.get_dummies(data=df,
                         prefix=cat_variables,
                         columns=cat_variables,
                         dtype=int)

print(df.head())
print(df.columns)

# Exclude the target variable from the feature list; the model should learn from
# the remaining columns only.
features = [x for x in df.columns if x != 'HeartDisease']

print(len(features))

# Split the data into training and validation sets. Keeping 80% for training and
# 20% for validation helps estimate how well the model generalizes to unseen data.
# train_test_split() creates separate feature and target arrays for training and
# validation, so we can fit the model on one subset and evaluate it on another.
X_train, X_val, y_train, y_val = train_test_split(
    df[features],
    df['HeartDisease'],
    train_size=0.8,
    random_state=RANDOM_STATE,
)

# These prints summarize the split size and class balance for a quick sanity check.
print(f'train samples: {len(X_train)}')
print(f'validation samples: {len(X_val)}')
print(f'target proportion: {sum(y_train)/len(y_train):.4f}')

# Explore how the minimum number of samples required to split a node affects tree
# complexity and validation accuracy. Smaller values often create deeper, more
# complex trees, while larger values create simpler trees that may generalize better.
min_samples_split_list = [2, 10, 30, 50, 100, 200, 300, 700]

# Explore how limiting tree depth impacts model fitting. A shallow tree avoids
# overfitting, while a deeper tree can capture more complex patterns in the data.
max_depth_list = [1, 2, 3, 4, 8, 16, 32, 64, None]  # None means no depth limit.

# Store the training and validation scores for each hyperparameter value so we can
# compare model behavior visually.
accuracy_list_train = []
accuracy_list_val = []
for min_samples_split in min_samples_split_list:
    # Train a decision tree with a specific split threshold and then evaluate it on
    # both the training and validation sets.
    # DecisionTreeClassifier() creates the model object, and .fit() trains it using
    # the training features and labels. This is the core learning step.
    model = DecisionTreeClassifier(
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE,
    ).fit(X_train, y_train)

    # .predict() uses the learned decision rules to estimate the class for each
    # observation in the train and validation sets.
    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)

    # accuracy_score() compares the predicted labels with the true labels and returns
    # the fraction of correct predictions, which measures predictive performance.
    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_val = accuracy_score(predictions_val, y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

# Plot the effect of min_samples_split on learning quality.
# plt.title(), plt.xlabel(), and plt.ylabel() label the chart so the result is easy
# to interpret. plt.xticks() sets the x-axis values to the tested split thresholds.
# plt.plot() draws the training and validation accuracy curves, and plt.show()
# displays the chart for visual inspection.
plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks=range(len(min_samples_split_list)), labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train', 'Validation'])
plt.show()

# Repeat the same process for max_depth. This helps identify whether a tree is
# growing too deep and overfitting the training data.
accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    # We test different tree depths to see how model complexity affects performance.
    # A deeper tree can fit the training data more closely, but may overfit and
    # perform worse on validation data.
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=RANDOM_STATE,
    ).fit(X_train, y_train)

    predictions_train = model.predict(X_train)
    predictions_val = model.predict(X_val)
    accuracy_train = accuracy_score(predictions_train, y_train)
    accuracy_val = accuracy_score(predictions_val, y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

# Visualize the validation curve to choose a balanced depth for the final model.
# This graph helps identify whether the model is underfitting or overfitting as the
# tree gets deeper. The best depth is usually the one with strong validation accuracy
# without a large gap between train and validation performance.
plt.title('Train x Validation metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks=range(len(max_depth_list)), labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train', 'Validation'])
plt.show()

# Final model configuration after inspecting the tuning curves. The chosen values
# aim to keep the tree simple enough to generalize well while still capturing the
# key patterns in the heart disease data.
# This final DecisionTreeClassifier is trained on the training set using the tuned
# hyperparameters, and then evaluated on both training and validation data.
decision_tree_model = DecisionTreeClassifier(
    min_samples_split=50,
    max_depth=4,
    random_state=RANDOM_STATE,
).fit(X_train, y_train)

# We print the final accuracy values to summarize how well the trained tree performs
# on seen data (training set) versus unseen data (validation set).
print(f"Metrics train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_train), y_train):.4f}")
print(f"Metrics validation:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_val), y_val):.4f}")