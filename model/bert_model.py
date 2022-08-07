import os
import pathlib

import mlflow
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from sklearn.model_selection import train_test_split
from tensorflow.python.saved_model import signature_constants


def initialize_bert():
    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    layer = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
    layer = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(layer)

    model = tf.keras.Model(inputs=[text_input], outputs=[layer])
    return model


def compile_model(model):
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=metrics)
    save_model(model)
    return model


def load_train_data():
    df = pd.read_csv("spam.csv")
    df_spam = df[df['Category'] == 'spam']
    df_ham = df[df['Category'] == 'ham']
    df_ham_downsampled = df_ham.sample(df_spam.shape[0])
    df_balanced = pd.concat([df_ham_downsampled, df_spam])

    df_balanced['spam'] = df_balanced['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    x_train, x_test, y_train, y_test = train_test_split(df_balanced['Message'], df_balanced['spam'],
                                                        stratify=df_balanced['spam'])

    return x_train, x_test, y_train, y_test


def save_model(model):
    tag = [tf.compat.v1.saved_model.tag_constants.SERVING]
    key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    model_path = 'spam_classifier'
    model.save(model_path)

    mlflow.tensorflow.log_model(tf_saved_model_dir=model_path,
                                tf_meta_graph_tags=tag,
                                tf_signature_def_key=key,
                                artifact_path="tf-models",
                                registered_model_name="group9-spam-detection")

