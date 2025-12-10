import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from plot_training import plot_training_history, plot_auc

df = pd.read_csv("data/processed/electron_dataset.csv")

# print(df['target'].value_counts())

X = df.drop(columns = ["target"]).to_numpy()

y = df["target"].to_numpy()

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(64),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Dense(32),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Dense(16),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights_dict = dict(enumerate(class_weights))

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.2,
    verbose = 2,
    class_weight=class_weights_dict
)

plot_training_history(history, save_path="results/training_plot.png")

plot_auc(history, save_path="results/auc_plot.png")