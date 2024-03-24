import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import numpy as np



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


# Assuming trees construction from the previous steps...

# Initialize the dictionary to store paths
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


def Branch_feature_extraction(df, name):
    paths_dict = Extract_paths(df)
    new_df = pd.DataFrame()

    # Wrap df.iterrows() with tqdm() to display a progress bar
    for _, row in tqdm(df.iterrows()):
        author = row['author']
        df_cur_branch = get_df_mean_by_user(paths_dict[row['Unnamed: 0']], name)
        author_row = df_cur_branch[df_cur_branch['author'] == author].iloc[0]
        author_row.drop('author', inplace=True)
        new_df = pd.concat([new_df, author_row.to_frame().T], axis=0)

    return new_df




# def get_df_mean_by_user(df, name):
#     dfnew = df.drop(columns=['Unnamed: 0', 'tree_id', 'parent', 'node_id'])
#     GroupByUser = dfnew.groupby("author").mean()
#     return pd.merge(df[['author']], GroupByUser, on='author', suffixes=('', f'_{name}'))


def get_df_mean_by_user(df, name):
    # Drop columns that are not needed for the mean calculation
    dfnew = df.drop(columns=['Unnamed: 0', 'tree_id', 'parent', 'node_id'])

    # Calculate the mean of the remaining columns grouped by 'author'
    GroupByUser = dfnew.groupby("author", as_index=False).mean()

    # Ensure column names in GroupByUser are suffixed accordingly, except 'author'
    GroupByUser = GroupByUser.rename(columns={col: f'{col}_{name}' for col in GroupByUser.columns if col != 'author'})

    # Perform a left join merge to maintain the order and size of the original df
    result_df = pd.merge(df[['author']], GroupByUser, on='author', how='left')

    return result_df


def Forum_feature_extraction(df, name='forum'):
    return get_df_mean_by_user(df, name)


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
            cur_tree_df = pd.concat([cur_tree_df, new_row])
            df_cur_tree = get_df_mean_by_user(cur_tree_df, name)

            author_row = df_cur_tree[df_cur_tree[f'author'] == author].iloc[0]
            author_row.drop(f'author', inplace=True)
            new_df = pd.concat([new_df, author_row.to_frame().T], axis=0)

    return new_df


def extract_features(df, isTrain, indx_cross_val, embedding_method_current_message):
    TrainTest = "train" if isTrain else "test"

    folder_path = "featuresEmbeddings"

    # Ensure the directory exists before writing the file
    os.makedirs(folder_path, exist_ok=True)

    folder_path = f"{folder_path}/{embedding_method_current_message}"

    os.makedirs(folder_path, exist_ok=True)

    if TrainTest == "train":
        if not os.path.exists(f'{folder_path}/{indx_cross_val}_forum_{TrainTest}.pkl'):
            forum_features = Forum_feature_extraction(df, 'forum')
            with open(f'{folder_path}/{indx_cross_val}_forum_{TrainTest}.pkl', 'wb') as file:
                pickle.dump(forum_features, file)

    if not os.path.exists(f'{folder_path}/{indx_cross_val}_branch_{TrainTest}.pkl'):
        branch_features = Branch_feature_extraction(df, 'branch')
        with open(f'{folder_path}/{indx_cross_val}_branch_{TrainTest}.pkl', 'wb') as file:
            pickle.dump(branch_features, file)

    if not os.path.exists(f'{folder_path}/{indx_cross_val}_tree_{TrainTest}.pkl'):
        tree_features = Tree_feature_extraction(df, 'tree')
        with open(f'{folder_path}/{indx_cross_val}_tree_{TrainTest}.pkl', 'wb') as file:
            pickle.dump(tree_features, file)
