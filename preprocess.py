import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import numpy as np

column_names = ['author', 'Aggressive', 'AgreeBut',
                'AgreeToDisagree', 'Alternative', 'Answer', 'AttackValidity', 'BAD',
                'Clarification', 'Complaint', 'Convergence', 'CounterArgument',
                'CriticalQuestion', 'DirectNo', 'DoubleVoicing', 'Extension',
                'Irrelevance', 'Moderation', 'NegTransformation', 'Nitpicking',
                'NoReasonDisagreement', 'Personal', 'Positive', 'Repetition',
                'RephraseAttack', 'RequestClarification', 'Ridicule', 'Sarcasm',
                'Softening', 'Sources', 'ViableTransformation', 'WQualifiers',
                'Untagged']

tag_columns = ['Aggressive', 'AgreeBut', 'AgreeToDisagree',
               'Alternative', 'Answer', 'AttackValidity', 'BAD', 'Clarification',
               'Complaint', 'Convergence', 'CounterArgument', 'CriticalQuestion',
               'DirectNo', 'DoubleVoicing', 'Extension', 'Irrelevance', 'Moderation',
               'NegTransformation', 'Nitpicking', 'NoReasonDisagreement', 'Personal',
               'Positive', 'Repetition', 'RephraseAttack', 'RequestClarification',
               'Ridicule', 'Sarcasm', 'Softening', 'Sources', 'ViableTransformation',
               'WQualifiers', 'Untagged']

forum_features = None
default_user = None


def calculate_untagged(row):
    if row[tag_columns].sum() == 0:
        return 1
    else:
        return 0


def normalize(df, col_name, axis=0, cols=None):
    if axis == 0:
        # Normalize by column (along the columns)
        if (df[col_name].max() - df[col_name].min() == 0):
            normalized_column = [1] * df.shape[0]
        else:
            normalized_column = (df[col_name] - df[col_name].min()) / (df[col_name].max() - df[col_name].min())
    elif axis == 1:
        # Normalize by row (along the rows)
        normalized_column = (df[col_name] - df[cols].min(axis=1)) / (df[cols].max(axis=1) - df[cols].min(axis=1))
    else:
        raise ValueError("Invalid axis. Use 0 for column-wise normalization or 1 for row-wise normalization.")

    # Create a new DataFrame with the normalized column
    df_normalized = df.copy()
    df_normalized[col_name] = normalized_column

    return df_normalized


def count_words(text):
    return len(text.split())


def normalize_df(df, name):
    for tag in tag_columns:
        df = normalize(df, tag, axis=1, cols=tag_columns)
    df = normalize(df, 'num_comment', axis=0)
    df = normalize(df, 'min_timestamp', axis=0)
    df = normalize(df, 'sending_rate', axis=0)
    df = normalize(df, 'avg_tag_msg', axis=0)
    df = normalize(df, 'avg_word_text', axis=0)

    df['min_timestamp'] = 1 - df['min_timestamp']
    df.rename(columns={'min_timestamp': 'duration'}, inplace=True)

    new_columns = [f'{col}_{name}' for col in df.columns]
    df.columns = new_columns

    return df


def feature_extraction(df, current_message=False):
    # Drop unnecessary columns at the beginning to simplify processing
    df_cleaned = df.drop(['Unnamed: 0', 'parent'], axis=1, errors='ignore')

    # Calculate average word count per author's text
    df_cleaned['word_count'] = df_cleaned['text'].apply(count_words)
    avg_word_counts = df_cleaned.groupby('author')['word_count'].mean().reset_index(name='avg_word_text')

    # Calculate number of comments, min and max timestamps per author
    agg_operations = {
        'timestamp': ['min', 'max'],
        'text': 'size'  # This will count the number of texts/comments per author
    }
    agg_df = df_cleaned.groupby('author').agg(agg_operations)
    agg_df.columns = ['min_timestamp', 'max_timestamp', 'num_comment']  # Flatten MultiIndex columns

    # Calculate sending rate
    agg_df['sending_rate'] = (agg_df['max_timestamp'] - agg_df['min_timestamp']) / agg_df['num_comment']

    # Merge with avg_word_counts
    merged_df = pd.merge(agg_df.reset_index(), avg_word_counts, on='author')

    # Sum of tag columns
    if current_message:
        current_author = df_cleaned['author'].iloc[0]
        df_cleaned = df_cleaned.iloc[1:]

        if current_author not in df_cleaned['author'].values:
            # This block executes if, after the operation, there are no rows left for 'current_author'
            # Now, check if 'current_author' exists in 'forum_features' to proceed with fetching and processing their data
            if current_author in forum_features.index:
                user_dataframe = pd.DataFrame(forum_features.loc[current_author]).T
            else:
                # Handle the case where 'current_author' is not found in 'forum_features'
                # You might want to create an empty DataFrame or use 'default_user' as fallback
                user_dataframe = pd.DataFrame(default_user).T  # Ensure 'default_user' is defined appropriately
            new_columns = [col[:-6] for col in user_dataframe.columns]  # Rename columns by removing the last 6 chars
            user_dataframe.columns = new_columns
            user_dataframe['author'] = current_author
            df_cleaned = pd.concat([user_dataframe[column_names], df_cleaned[column_names]], axis=0)


    tag_sum_df = df_cleaned.groupby('author')[tag_columns].sum().reset_index()
    merged_df = pd.merge(merged_df, tag_sum_df, on='author')

    # Calculate diversity and average tag per message
    merged_df["diversity"] = 1 - (merged_df[tag_columns].max(axis=1) / merged_df[tag_columns].sum(axis=1))
    merged_df["diversity"].fillna(0, inplace=True)  # Handle division by zero
    merged_df["avg_tag_msg"] = merged_df[tag_columns].sum(axis=1) / merged_df['num_comment']
    merged_df["avg_tag_msg"].fillna(0, inplace=True)  # Handle division by zero

    # Drop max_timestamp as it's no longer needed
    merged_df.drop(['max_timestamp'], axis=1, inplace=True)

    return merged_df


# Define the TreeNode class
class TreeNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.children = []
        self.parent = None

    def add_child(self, child):
        self.children.append(child)
        child.parent = self


# Initialize a dictionary to hold node_id as key and TreeNode as value


# trees now contains the root nodes of each tree in your dataset
def find_leaves(node):
    if not node.children:
        return [node]
    leaves = []
    for child in node.children:
        leaves.extend(find_leaves(child))
    return leaves


# Function to trace back the path from a leaf to the root
def trace_path(node):
    path = []
    while node:
        path.append(node.node_id)
        node = node.parent
    return path[::-1]  # Reverse the path to start from the root


# df.set_index('Unnamed: 0', inplace=True)
def Branch_feature_extraction(df, name):
    # Preallocate a list to collect rows for a single concatenation later if needed

    paths_dict = Extract_paths(df)
    new_df = pd.DataFrame()

    # Wrap df.iterrows() with tqdm() to display a progress bar
    for _, row in tqdm(df.iterrows()):
        author = row['author']
        df_cur_branch = paths_dict[row['Unnamed: 0']]
        features_cur_branch = feature_extraction(df_cur_branch, current_message=True)
        features_cur_branch = normalize_df(features_cur_branch, name)
        author_row = features_cur_branch[features_cur_branch[f'author_{name}'] == author].iloc[0]
        author_row.drop(f'author_{name}', inplace=True)
        # author_row.drop(f'text_{name}', inplace=True)
        # author_row.drop(f'node_id_{name}', inplace=True)
        # author_row.drop(f'tree_id_{name}', inplace=True)
        new_df = pd.concat([new_df, author_row.to_frame().T], axis=0)

    return new_df


def Extract_paths(df):
    paths_dict = {}

    for _, row in df.iterrows():
        myPath = pd.DataFrame()

        myPath = pd.concat([myPath, row.T], axis=1)
        myPath = myPath.T
        parent = row['parent']

        while parent != -1:
            parentRow = df[df['Unnamed: 0'] == parent].iloc[0]
            myPath = pd.concat([myPath, parentRow.to_frame().T], axis=0)
            parent = parentRow['parent']

        paths_dict[row['Unnamed: 0']] = myPath

    return paths_dict


def Forum_feature_extraction(df, byUser=False, name='forum'):
    feature_forum = feature_extraction(df)
    feature_forum = normalize_df(feature_forum, name)
    # feature_forum.drop(columns=[f'text_{name}', f'node_id_{name}', f'tree_id_{name}'], inplace=True)

    return feature_forum


def Tree_feature_extraction(df, name):
    tree_dict = df.groupby('tree_id')['Unnamed: 0'].apply(list).to_dict()
    # Preallocate a list to collect rows for a single concatenation later if needed
    new_df = pd.DataFrame()

    for key, tree_nodes in tqdm(tree_dict.items()):
        cur_tree_df = pd.DataFrame()
        for node_id in tree_nodes:
            # Access the row directly using the index, avoiding the search in each iteration
            new_row = df[df['Unnamed: 0'] == node_id]
            author = new_row['author'].values[0]  # Access the scalar value from the Series

            copy_new_row = new_row.copy()

            df_cur_tree = feature_extraction(pd.concat([copy_new_row, cur_tree_df]), current_message=True)
            df_cur_tree = normalize_df(df_cur_tree, name)

            cur_tree_df = pd.concat([cur_tree_df, new_row])

            author_row = df_cur_tree[df_cur_tree[f'author_{name}'] == author].iloc[0]
            new_df = pd.concat([new_df, author_row.to_frame().T], axis=0)

    # new_df.drop(columns=[f'author_{name}', f'text_{name}', f'node_id_{name}', f'tree_id_{name}'], inplace=True)
    new_df.drop(columns=[f'author_{name}'], inplace=True)
    return new_df


def List_last_user_feature_extraction(df):
    list_last_user = []

    # Create a mapping from 'Unnamed: 0' to row index
    index_mapping = {id_val: index for index, id_val in enumerate(df['Unnamed: 0'])}

    for _, row in df.iterrows():
        parent = row['parent']

        # Look up the row index using the mapping
        parent_index = index_mapping.get(parent, -1)  # Returns -1 if parent not found
        list_last_user.append(parent_index)

    return list_last_user


def extract_features(df, isTrain, indx_cross_val, byUser=True):
    global forum_features, default_user
    TrainTest = "train" if isTrain else "test"

    # Ensure the directory exists before writing the file
    os.makedirs('features', exist_ok=True)

    if TrainTest == "train":
        if not os.path.exists(f'features/{indx_cross_val}_forum_{TrainTest}.pkl'):
            forum_features = Forum_feature_extraction(df, byUser, 'forum')
            with open(f'features/{indx_cross_val}_forum_{TrainTest}.pkl', 'wb') as file:
                pickle.dump(forum_features, file)

            forum_features.set_index('author_forum', inplace=True)
            default_user = forum_features.mean()
        else:
            if os.path.exists(f'features/{indx_cross_val}_forum_train.pkl'):
                forum_features = pd.read_pickle(f'features/{indx_cross_val}_forum_train.pkl')
                forum_features.set_index('author_forum', inplace=True)
                default_user = forum_features.mean()

    if not os.path.exists(f'features/{indx_cross_val}_branch_{TrainTest}.pkl'):
        branch_features = Branch_feature_extraction(df, 'branch')
        with open(f'features/{indx_cross_val}_branch_{TrainTest}.pkl', 'wb') as file:
            pickle.dump(branch_features, file)

    if not os.path.exists(f'features/{indx_cross_val}_tree_{TrainTest}.pkl'):
        tree_features = Tree_feature_extraction(df, 'tree')
        with open(f'features/{indx_cross_val}_tree_{TrainTest}.pkl', 'wb') as file:
            pickle.dump(tree_features, file)



def extract_last_user(df, isTrain, indx_cross_val):
    # Ensure the directory exists before writing the file
    os.makedirs('features', exist_ok=True)

    TrainTest = "train" if isTrain else "test"

    if not os.path.exists(f'features/{indx_cross_val}_last_user_{TrainTest}.pkl'):

        list_last_user_feature = List_last_user_feature_extraction(df)

        with open(f'features/{indx_cross_val}_last_user_{TrainTest}.pkl', 'wb') as file:
            pickle.dump(list_last_user_feature, file)

