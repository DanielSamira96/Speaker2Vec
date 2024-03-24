import os
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import torch
import gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support, f1_score
import numpy as np
from tensorflow_addons.optimizers import AdamW
import preprocess
import preprocessEmbeddings
import pickle
import datetime
from transformers import BertTokenizer, BertModel
import json
from tensorflow.keras.layers import Layer, Dense, Activation, Input
from tensorflow.keras.models import Model
import tensorflow as tf


os.environ["TOKENIZERS_PARALLELISM"] = "false"


os.environ['TF_DETERMINISTIC_OPS'] = '1'


def get_embedding_df(embedding_name):
    if embedding_name == 'clip':
        if os.path.exists("embeddings/clip_embeddings.pkl"):
            embeddings_df = pd.read_pickle("text_embeddings.pkl")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            Clip_name = "openai/clip-vit-large-patch14"
            processorClip = CLIPProcessor.from_pretrained(Clip_name)
            modelClip = CLIPModel.from_pretrained(Clip_name).to(device)

            def get_clip_text_embedding(text):
                with torch.no_grad():
                    inputs = processorClip(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
                    outputs = modelClip.get_text_features(**inputs)
                return outputs.cpu()

            text_embeddings = torch.vstack([get_clip_text_embedding(text) for text in data['text']]).squeeze()

            embeddings_df = pd.DataFrame(text_embeddings.numpy())

            embeddings_df.to_pickle("embeddings/text_embeddings.pkl")

    elif embedding_name == 'bert':
        if os.path.exists("embeddings/bert_embeddings.pkl"):
            embeddings_df = pd.read_pickle("embeddings/bert_embeddings.pkl")
        else:
            bert_name = "bert-base-uncased"
            tokenizer = BertTokenizer.from_pretrained(bert_name)
            model = BertModel.from_pretrained(bert_name)

            def get_bert_text_embedding(text):
                encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                with torch.no_grad():  # Disable gradient calculation for inference
                    outputs = model(**encoded_input)

                return outputs.pooler_output

            text_embeddings = torch.vstack([get_bert_text_embedding(text) for text in data['text']]).squeeze()

            embeddings_df = pd.DataFrame(text_embeddings.numpy())

            embeddings_df.to_pickle("embeddings/bert_embeddings.pkl")

    return embeddings_df


def main():
    file_path = 'annotated_trees_101.csv'
    data = pd.read_csv(file_path)
    random_seed = 42

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    epochs = 50

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    tf.random.set_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    tag_columns = ['Aggressive', 'AgreeBut', 'AgreeToDisagree',
                   'Alternative', 'Answer', 'AttackValidity', 'BAD', 'Clarification',
                   'Complaint', 'Convergence', 'CounterArgument', 'CriticalQuestion',
                   'DirectNo', 'DoubleVoicing', 'Extension', 'Irrelevance', 'Moderation',
                   'NegTransformation', 'Nitpicking', 'NoReasonDisagreement', 'Personal',
                   'Positive', 'Repetition', 'RephraseAttack', 'RequestClarification',
                   'Ridicule', 'Sarcasm', 'Softening', 'Sources', 'ViableTransformation',
                   'WQualifiers']

    tag_to_category = {
        'Moderation': 'Promoting Discussion',
        'RequestClarification': 'Promoting Discussion',
        'AttackValidity': 'Promoting Discussion',
        'Clarification': 'Promoting Discussion',
        'Answer': 'Promoting Discussion',
        'CounterArgument': 'Promoting Discussion',
        'Extension': 'Promoting Discussion',
        'ViableTransformation': 'Promoting Discussion',
        'Personal': 'Promoting Discussion',
        'BAD': 'Low Responsiveness',
        'Repetition': 'Low Responsiveness',
        'NegTransformation': 'Low Responsiveness',
        'NoReasonDisagreement': 'Low Responsiveness',
        'Convergence': 'Low Responsiveness',
        'AgreeToDisagree': 'Low Responsiveness',
        'Aggressive': 'Tone and Style',
        'Ridicule': 'Tone and Style',
        'Complaint': 'Tone and Style',
        'Sarcasm': 'Tone and Style',
        'Positive': 'Tone and Style',
        'WQualifiers': 'Tone and Style',
        'Softening': 'Disagreement Strategies',
        'AgreeBut': 'Disagreement Strategies',
        'DoubleVoicing': 'Disagreement Strategies',
        'Sources': 'Disagreement Strategies',
        'RephraseAttack': 'Disagreement Strategies',
        'CriticalQuestion': 'Disagreement Strategies',
        'Alternative': 'Disagreement Strategies',
        'DirectNo': 'Disagreement Strategies',
        'Irrelevance': 'Disagreement Strategies',
        'Nitpicking': 'Disagreement Strategies',
        'Untagged': 'Untagged'
    }

    def calculate_untagged(row):
        if row[tag_columns].sum() == 0:
            return 1
        else:
            return 0

    embedding_methods_current_message = ['clip', 'bert']
    embedding_methods_user_message = ['clip', 'bert']

    methods = ['one_hot_vector', 'fetuare_extraction', 'text_embedding', 'combined']
    # methods = ['combined']
    # methods = ['one_hot_vector', 'fetuare_extraction']
    # methods = ['combined']

    data['Untagged'] = data.apply(calculate_untagged, axis=1)

    results = {}

    history = {}

    for embedding_method_current_message in embedding_methods_current_message:
        for embedding_method_user_message in embedding_methods_user_message:

            for method in methods:
                if method == 'one_hot_vector' or method == 'fetuare_extraction':
                    if f'{method}_{embedding_method_current_message}' in history:
                        continue
                    else:
                        history[f'{method}_{embedding_method_current_message}'] = True
                print(f"Training Method: {method}\n"
                      f"Embedding Method Current Message:{embedding_method_current_message}\n"
                      f"Embedding Method User Message: {embedding_method_user_message}")
                if method == 'one_hot_vector':
                    X_method_one_hot_vector = pd.get_dummies(data['author']).astype(int)

                embedding_df_current_message = get_embedding_df(embedding_method_current_message)

                X_non_text = data.iloc[:, :7]
                labels = data.iloc[:, 7:]

                tag_names = labels.columns.tolist()


                gkf = GroupKFold(n_splits=5)

                groups = data['tree_id'].values

                folds_info = list(gkf.split(X=data, y=labels, groups=groups))


                def asymmetric_loss(y_true, y_pred):
                    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

                class GatedResidualNetwork(Layer):
                    def __init__(self, units, **kwargs):
                        super(GatedResidualNetwork, self).__init__(**kwargs)
                        self.units = units
                        # Transformation layers for each input
                        self.linear_transform_message = Dense(units)
                        self.linear_transform_current_user = Dense(units)
                        self.linear_transform_last_message_user = Dense(units)

                        # Transformation layers for original inputs to match the combined size
                        self.transform_original_message = Dense(units)
                        self.transform_original_current_user = Dense(units)
                        self.transform_original_last_message_user = Dense(units)

                        # Gate for each input
                        self.gate_message = Dense(units, activation='sigmoid')
                        self.gate_current_user = Dense(units, activation='sigmoid')
                        self.gate_last_message_user = Dense(units, activation='sigmoid')

                    def call(self, inputs):
                        # Assume inputs is a list of three tensors: [message, current_user, last_message_user]
                        message, current_user, last_message_user = inputs
                        # Transform each input
                        message_transformed = self.linear_transform_message(message)
                        current_user_transformed = self.linear_transform_current_user(current_user)
                        last_message_user_transformed = self.linear_transform_last_message_user(last_message_user)
                        # Apply gates
                        message_gate = self.gate_message(message_transformed)
                        current_user_gate = self.gate_current_user(current_user_transformed)
                        last_message_user_gate = self.gate_last_message_user(last_message_user_transformed)
                        # Combine inputs with gating
                        combined = (message_transformed * message_gate +
                                    current_user_transformed * current_user_gate +
                                    last_message_user_transformed * last_message_user_gate)

                        # Transform original inputs to match combined size and apply gates
                        message_res = self.transform_original_message(message) * message_gate
                        current_user_res = self.transform_original_current_user(current_user) * current_user_gate
                        last_message_user_res = self.transform_original_last_message_user(last_message_user) * last_message_user_gate

                        # Combine everything with residual connection
                        combined_with_residuals = combined + message_res + current_user_res + last_message_user_res

                        return combined_with_residuals

                def create_model(input_dims, output_dim, loss_function):
                    # Define three inputs for the model
                    input_message = Input(shape=(input_dims[0],), name='current_message')
                    input_current_user = Input(shape=(input_dims[1],), name='user_profile')
                    input_last_message_user = Input(shape=(input_dims[2],), name='last_user_profile')

                    # GRN layer takes a list of inputs
                    x = GatedResidualNetwork(1024)([input_message, input_current_user, input_last_message_user])
                    x = Activation('relu')(x)

                    # Following layers
                    x = Dense(1024, activation='relu')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Dropout(0.3)(x)
                    x = Dense(512, activation='relu')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Dropout(0.3)(x)
                    x = Dense(256, activation='relu')(x)
                    x = tf.keras.layers.Dropout(0.3)(x)
                    x = Dense(128, activation='relu')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Dropout(0.3)(x)
                    x = Dense(64, activation='relu')(x)
                    x = tf.keras.layers.Dropout(0.3)(x)

                    # Output layer
                    output = Dense(output_dim, activation='sigmoid')(x)  # Adjust activation based on your specific problem

                    # Create and compile the model
                    model = Model(inputs=[input_message, input_current_user, input_last_message_user], outputs=output)
                    model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4),
                                  loss=loss_function,
                                  metrics=['accuracy'])
                    return model

                # Initialize storage for results and metrics
                fold_metrics = {
                    'AL': {'loss': [], 'accuracy': [], 'f1_macro': [], 'f1_weighted': [], 'category_f1_macro': {},
                           'category_f1_weighted': {}},
                    'BCE': {'loss': [], 'accuracy': [], 'f1_macro': [], 'f1_weighted': [], 'category_f1_macro': {},
                            'category_f1_weighted': {}}
                }

                output_dim = labels.shape[1]

                for fold, (train_idx, test_idx) in enumerate(folds_info):
                    print(f"Training on fold {fold + 1}/{len(folds_info)}...")

                    df_train = data.iloc[train_idx]
                    df_test = data.iloc[test_idx]

                    preprocess.extract_last_user(df_train, True, fold)
                    preprocess.extract_last_user(df_test, False, fold)

                    if method == 'one_hot_vector':
                        X_train, X_test = X_method_one_hot_vector.iloc[train_idx], X_method_one_hot_vector.iloc[test_idx]

                    if method == 'fetuare_extraction' or method == 'combined':
                        preprocess.extract_features(df_train, True, fold)
                        preprocess.extract_features(df_test, False, fold)

                        if os.path.exists(f'features/{fold}_forum_train.pkl'):
                            df_train_forum = pd.read_pickle(f'features/{fold}_forum_train.pkl')
                        if os.path.exists(f'features/{fold}_tree_train.pkl'):
                            df_train_tree = pd.read_pickle(f'features/{fold}_tree_train.pkl')
                        if os.path.exists(f'features/{fold}_branch_train.pkl'):
                            df_train_branch = pd.read_pickle(f'features/{fold}_branch_train.pkl')

                        df_train_forum.set_index('author_forum', inplace=True)

                        df_train_forum_list = []

                        for i, row in df_train.iterrows():
                            author_name = row['author']

                            df_train_forum_list.append(pd.Series(df_train_forum.loc[author_name]))

                        df_train_forum_db = pd.DataFrame(df_train_forum_list)

                        X_train = pd.concat([df_train_forum_db.reset_index(drop=True), df_train_tree.reset_index(drop=True), df_train_branch.reset_index(drop=True)], axis=1).reset_index(drop=True)

                        X_train = X_train.astype('float32')

                        X_train_fetuare_extraction = X_train

                        if os.path.exists(f'features/{fold}_tree_test.pkl'):
                            df_test_tree = pd.read_pickle(f'features/{fold}_tree_test.pkl')
                        if os.path.exists(f'features/{fold}_branch_test.pkl'):
                            df_test_branch = pd.read_pickle(f'features/{fold}_branch_test.pkl')

                        default_user = df_train_forum.mean()

                        df_test_forum_list = []

                        for i, row in df_test.iterrows():
                            author_name = row['author']
                            if author_name in df_train_forum.index:
                                df_test_forum_list.append(pd.Series(df_train_forum.loc[author_name]))
                            else:
                                df_test_forum_list.append(default_user)

                        df_test_forum = pd.DataFrame(df_test_forum_list)

                        X_test = pd.concat([df_test_forum.reset_index(drop=True), df_test_tree.reset_index(drop=True), df_test_branch.reset_index(drop=True)], axis=1).reset_index(drop=True)

                        X_test = X_test.astype('float32')

                        X_test_fetuare_extraction = X_test
                        default_user = pd.concat([default_user, default_user, default_user], ignore_index=True)
                        default_user_fetuare_extraction = default_user

                    if method == 'text_embedding' or method == 'combined':
                        embedding_df_user_message = get_embedding_df(embedding_method_user_message)

                        X_train_user_message, X_test_user_message = embedding_df_user_message.iloc[train_idx], embedding_df_user_message.iloc[test_idx]

                        preprocessEmbeddings.extract_features(pd.concat([X_train_user_message, df_train[['Unnamed: 0', 'author', 'tree_id', 'parent', 'node_id']]], axis=1), True, fold, embedding_method_user_message)
                        preprocessEmbeddings.extract_features(pd.concat([X_test_user_message, df_test[['Unnamed: 0', 'author', 'tree_id', 'parent', 'node_id']]], axis=1), False, fold, embedding_method_user_message)

                        foldr_path = f"featuresEmbeddings/{embedding_method_user_message}"

                        if os.path.exists(f'{foldr_path}/{fold}_forum_train.pkl'):
                            df_train_forum = pd.read_pickle(f'{foldr_path}/{fold}_forum_train.pkl')
                        if os.path.exists(f'{foldr_path}/{fold}_tree_train.pkl'):
                            df_train_tree = pd.read_pickle(f'{foldr_path}/{fold}_tree_train.pkl')
                        if os.path.exists(f'{foldr_path}/{fold}_branch_train.pkl'):
                            df_train_branch = pd.read_pickle(f'{foldr_path}/{fold}_branch_train.pkl')

                        X_train = pd.concat([df_train_forum.drop(columns='author').reset_index(drop=True), df_train_tree.reset_index(drop=True), df_train_branch.reset_index(drop=True)], axis=1).reset_index(drop=True)

                        X_train = X_train.astype('float32')

                        X_train_text_embedding = X_train

                        if os.path.exists(f'{foldr_path}/{fold}_tree_test.pkl'):
                            df_test_tree = pd.read_pickle(f'{foldr_path}/{fold}_tree_test.pkl')
                        if os.path.exists(f'{foldr_path}/{fold}_branch_test.pkl'):
                            df_test_branch = pd.read_pickle(f'{foldr_path}/{fold}_branch_test.pkl')

                        users_train_forum = df_train_forum.groupby('author').mean()

                        default_user = users_train_forum.mean()

                        df_test_forum_list = []

                        for i, row in df_test.iterrows():
                            author_name = row['author']
                            if author_name in users_train_forum.index:
                                df_test_forum_list.append(pd.Series(users_train_forum.loc[author_name]))
                            else:
                                df_test_forum_list.append(default_user)

                        df_test_forum = pd.DataFrame(df_test_forum_list)

                        X_test = pd.concat([df_test_forum.reset_index(drop=True), df_test_tree.reset_index(drop=True), df_test_branch.reset_index(drop=True)], axis=1).reset_index(drop=True)

                        X_test = X_test.astype('float32')

                        X_test_text_embedding = X_test
                        default_user = pd.concat([default_user, default_user, default_user], ignore_index=True)
                        default_user_text_embedding = default_user

                    if method == 'combined':
                        X_train = pd.concat([X_train_fetuare_extraction, X_train_text_embedding], axis=1).reset_index(drop=True)
                        X_test = pd.concat([X_test_fetuare_extraction, X_test_text_embedding], axis=1).reset_index(drop=True)

                        default_user = pd.concat([default_user_fetuare_extraction, default_user_text_embedding]).reset_index(drop=True)


                    X_train_current_message, X_test_current_message = embedding_df_current_message.iloc[train_idx], embedding_df_current_message.iloc[test_idx]

                    if os.path.exists(f'features/{fold}_last_user_train.pkl'):
                        list_train_last_user = pd.read_pickle(f'features/{fold}_last_user_train.pkl')
                    if os.path.exists(f'features/{fold}_last_user_test.pkl'):
                        list_test_last_user = pd.read_pickle(f'features/{fold}_last_user_test.pkl')

                    # Initialize a list to collect rows
                    rows_to_append = []

                    if method == 'one_hot_vector':
                        default_user = np.zeros((X_train.shape[1],))
                    else:
                        default_user.index = X_train.columns

                    for idx in list_train_last_user:
                        if idx != -1:
                            # When idx is valid, append the row from X_test as a Series to maintain column alignment
                            rows_to_append.append(X_train.iloc[idx])
                        else:
                            # When idx is -1, append the default_user Series
                            rows_to_append.append(default_user)

                    # Convert the list of Series to a DataFrame
                    X_train_last_user = pd.DataFrame(rows_to_append, columns=X_train.columns).reset_index(drop=True)


                    # Initialize a list to collect rows
                    rows_to_append = []

                    for idx in list_test_last_user:
                        if idx != -1:
                            # When idx is valid, append the row from X_test as a Series to maintain column alignment
                            rows_to_append.append(X_test.iloc[idx])
                        else:
                            # When idx is -1, append the default_user Series
                            rows_to_append.append(default_user)

                    # Convert the list of Series to a DataFrame
                    X_test_last_user = pd.DataFrame(rows_to_append).reset_index(drop=True)

                    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

                    # Assuming you have determined the input dimensions beforehand
                    input_dims = (X_train_current_message.shape[1], X_train.shape[1], X_train_last_user.shape[1])


                    # Create and train the AL model
                    model_al = create_model(input_dims, output_dim, asymmetric_loss)
                    model_al.fit([X_train_current_message, X_train, X_train_last_user], y_train, epochs=epochs,batch_size=32)
                    loss_al, accuracy_al = model_al.evaluate([X_test_current_message, X_test, X_test_last_user], y_test, verbose=0)
                    y_pred_al = model_al.predict([X_test_current_message, X_test, X_test_last_user])
                    y_pred_bin_al = (y_pred_al > 0.5).astype(int)
                    precision_al, recall_al, f1_score_al, _ = precision_recall_fscore_support(y_test, y_pred_bin_al, average=None)


                    # Train and evaluate BCE model
                    model_bce = create_model(input_dims, output_dim, 'binary_crossentropy')
                    model_bce.fit([X_train_current_message, X_train, X_train_last_user], y_train, epochs=epochs,batch_size=32)
                    loss_bce, accuracy_bce = model_bce.evaluate([X_test_current_message, X_test, X_test_last_user], y_test, verbose=0)
                    y_pred_bce = model_bce.predict([X_test_current_message, X_test, X_test_last_user])
                    y_pred_bin_bce = (y_pred_bce > 0.5).astype(int)
                    precision_bce, recall_bce, f1_score_bce, _ = precision_recall_fscore_support(y_test, y_pred_bin_bce, average=None)

                    # Calculate macro and weighted F1-scores
                    macro_f1_al = f1_score(y_test, y_pred_bin_al, average='macro')
                    weighted_f1_al = f1_score(y_test, y_pred_bin_al, average='weighted')
                    macro_f1_bce = f1_score(y_test, y_pred_bin_bce, average='macro')
                    weighted_f1_bce = f1_score(y_test, y_pred_bin_bce, average='weighted')

                    # Store overall metrics
                    fold_metrics['AL']['loss'].append(loss_al)
                    fold_metrics['AL']['accuracy'].append(accuracy_al)
                    fold_metrics['AL']['f1_macro'].append(macro_f1_al)
                    fold_metrics['AL']['f1_weighted'].append(weighted_f1_al)

                    fold_metrics['BCE']['loss'].append(loss_bce)
                    fold_metrics['BCE']['accuracy'].append(accuracy_bce)
                    fold_metrics['BCE']['f1_macro'].append(macro_f1_bce)
                    fold_metrics['BCE']['f1_weighted'].append(weighted_f1_bce)

                    # Calculate tag supports and category-wise F1 scores
                    tag_supports = y_test.sum(axis=0)
                    tag_supports = {tag: support for tag, support in zip(tag_names, tag_supports)}

                    # Initialize support counts and weights for categories
                    category_supports = {}
                    tag_weights = {}

                    # Calculate total support for each category
                    for tag, support in tag_supports.items():
                        category = tag_to_category[tag]
                        if category not in category_supports:
                            category_supports[category] = 0
                        category_supports[category] += support

                    # Calculate weights for each tag within its category
                    for tag, support in tag_supports.items():
                        category = tag_to_category[tag]
                        if category_supports[category] > 0:
                            tag_weights[tag] = support / category_supports[category]
                        else:
                            # Handle the case where category support is 0;
                            # you might want to assign a default value or skip the calculation
                            tag_weights[tag] = 0  # or use an appropriate default value

                    # Aggregate F1 scores by category
                    for tag_index, tag in enumerate(tag_names):
                        category = tag_to_category.get(tag)
                        if category:
                            if category not in fold_metrics['AL']['category_f1_macro']:
                                fold_metrics['AL']['category_f1_macro'][category] = []
                                fold_metrics['AL']['category_f1_weighted'][category] = []
                                fold_metrics['BCE']['category_f1_macro'][category] = []
                                fold_metrics['BCE']['category_f1_weighted'][category] = []

                            macro_f1_al = np.mean(f1_score_al[tag_index])  # Adjusted to use f1_score_al correctly
                            weighted_f1_al = np.sum(f1_score_al[tag_index] * tag_weights[tag])  # Adjusted to use f1_score_al correctly

                            macro_f1_bce = np.mean(f1_score_bce[tag_index])  # Adjusted to use f1_score_bce correctly
                            weighted_f1_bce = np.sum(f1_score_bce[tag_index] * tag_weights[tag])  # Adjusted to use f1_score_bce correctly

                            fold_metrics['AL']['category_f1_macro'][category].append(macro_f1_al)
                            fold_metrics['AL']['category_f1_weighted'][category].append(weighted_f1_al)
                            fold_metrics['BCE']['category_f1_macro'][category].append(macro_f1_bce)
                            fold_metrics['BCE']['category_f1_weighted'][category].append(weighted_f1_bce)

                if method == 'one_hot_vector' or method == 'fetuare_extraction':
                    result_name = f'{method}_{embedding_method_current_message}'
                else:
                    result_name = f'{method}_{embedding_method_current_message}_{embedding_method_user_message}'

                print(f"Results for {result_name}:")
                results[f'{result_name}'] = {}

                for loss_function, metrics in fold_metrics.items():
                    print(f"\nResults for {loss_function}:")
                    average_loss = np.mean(metrics['loss'])
                    average_accuracy = np.mean(metrics['accuracy'])
                    average_macro_f1 = np.mean(metrics['f1_macro'])
                    average_weighted_f1 = np.mean(metrics['f1_weighted'])

                    print(f"Average Loss: {average_loss:.4f}")
                    print(f"Average Accuracy: {average_accuracy:.4f}")
                    print(f"Average Macro F1-score: {average_macro_f1:.4f}")
                    print(f"Average Weighted F1-score: {average_weighted_f1:.4f}")

                    results[f'{result_name}'][f'{loss_function}'] = {}
                    results[f'{result_name}'][f'{loss_function}']['Loss'] = average_loss
                    results[f'{result_name}'][f'{loss_function}']['Accuracy'] = average_accuracy
                    results[f'{result_name}'][f'{loss_function}']['Macro F1-score'] = average_macro_f1
                    results[f'{result_name}'][f'{loss_function}']['Weighted F1-score'] = average_weighted_f1

                    # Print category-specific macro and weighted F1 scores
                    for category in metrics['category_f1_macro']:
                        average_category_macro_f1 = np.mean(metrics['category_f1_macro'][category])
                        average_category_weighted_f1 = np.mean(metrics['category_f1_weighted'][category])

                        print(f"Category '{category}': Macro F1: {average_category_macro_f1:.4f}, Weighted F1: {average_category_weighted_f1:.4f}")

                        results[f'{result_name}'][f'{loss_function}'][f'Macro F1-{category}'] = average_category_macro_f1
                        results[f'{result_name}'][f'{loss_function}'][f'Weighted F1-{category}'] = average_category_weighted_f1

                print("\nComplete results:", results)

    folder_name = f"NLP_results"
    os.makedirs(folder_name, exist_ok=True)
    print(results)

    filename = f"{folder_name}/results_{current_time}.json"

    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Results saved to {filename}")


if __name__ == '__main__':
    main()

