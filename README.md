# Toxicity_Guardrails_for_LLMs

**This project is used to detect toxicity in LLM response, flag it and automatically respond in a ethical way.**

**Dataset:** https://paperswithcode.com/dataset/toxic-comment-classification-challenge

**Dataset Description**

You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

toxic

severe_toxic

obscene

threat

insult

identity_hate

You must create a model which predicts a probability of each type of toxicity for each comment.

Since, we are not interested in the type of toxicity, we just need to detect if the text is toxic and flag it if toxic.

so we combine our target to a single column to output 1 if toxic else if non-toxic tehn output 0.

**Preprocessing:**

1. convert to lowercase

2. remove digits and numbers

3. remove stopwords

**Model Architecture:**

from tensorflow.keras import layers

model = tf.keras.Sequential([

    vectorize_layer,
    
    layers.Embedding(max_tokens, 128, mask_zero=True),
    
    layers.SpatialDropout1D(0.3),
    
    layers.Bidirectional(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    
    layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    
    layers.Dropout(0.5),
    
    layers.Dense(1, activation='sigmoid')
    
])

**TextVectorization:** Turns raw strings → fixed‑length integer sequences (token IDs).

**Embedding(max_tokens, 128):** Learns a 128‑dimensional vector for each word ID so similar words sit close in space (semantic geometry).

**SpatialDropout1D(0.3):** Randomly “masks out” entire word embeddings each step, forcing the model to rely on context—not individual tokens.

**Bidirectional LSTM(64):** Reads the sentence left→right and right→left, capturing long‑range dependencies (“not good” ≠ “good”).

**Dense(32, relu) + Dropout(0.5):** Condenses the LSTM’s 128‑d context vector, adds non‑linearity, and regularises.

**Dense(1, sigmoid):** Outputs a probability of toxicity.
