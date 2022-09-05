---
layout: post
title:  "Dimension Problem"
date:   2022-09-05 20:40:27 +0700
categories: coding
permalink: "/coding/dimension-problem_20220905.html"
---

I make a NN using the method from [keras example](https://keras.io/examples/nlp/text_classification_from_scratch/) mixed with [tensorflow dataset](https://www.tensorflow.org/datasets/overview).

I tired to implement [TextVectorization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization?hl=en) layer.

{% highlight python%}
# Fetch Dataset
# Construct a tf.data.Dataset
split_mode = ['train[:10%]', 'train[10%:15%]', 'test[:20%]']
(raw_train_ds, raw_val_ds, raw_test_ds), ds_info = tfds.load('ag_news_subset', 
                                                   split= split_mode,
                                                   shuffle_files=True,
                                                   batch_size=32,
                                                   as_supervised = True,
                                                   with_info=True)


{% endhighlight %}

After dataset is fetch I follow the step in [keras nlp example](https://keras.io/examples/nlp/text_classification_from_scratch/#prepare-the-data) data preparation with minor edit, to adapt with different dataset.
Add some regex magic i found from stackoverflow to remove stopword.

{% highlight python%}
def custom_standardization(input_data):
    # lowering
    lowercase = tf.strings.lower(input_data)
    # strip xml text
    stripped = tf.strings.regex_replace(lowercase, " #39;", "\'")
    stripped = tf.strings.regex_replace(stripped, " quot;", "\"")
    # remove stopword
    # regex magic : r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*'
    stripped = tf.strings.regex_replace(stripped, r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*',"")
    # strip punctuiation
    stripped = tf.strings.regex_replace(stripped, f"[{re.escape(string.punctuation)}]", " ")
    # remove double space
    # two or more whitespace
    stripped = tf.strings.regex_replace(stripped, r"(\s\s*)", " ")
    stripped = tf.strings.regex_replace(stripped, r"(\s+$)", "")
    return stripped
{% endhighlight %}

Then I process the dataset through keras TextVectorization layer. Using the method from keras example above.
{% highlight python%}
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label
# Vectorize the data.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)
# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10
{% endhighlight %}

With this cinfiguration as ann
{% highlight python%}
# A integer input for vocab indices.
inputs = tf.keras.Input(shape=(None,), dtype="float64")
embedding_layer = layers.Embedding(max_features, embedding_dim)(inputs)
# --------------------
# here to add autoencoder
# --------------------
# dnn
hdn  = layers.Dense(918, activation='selu') (embedding_layer)
hdn = layers.BatchNormalization(momentum=0.99, epsilon=0.0001)(hdn)
hdn  = layers.Dense(618, activation='elu') (hdn)
hdn = layers.BatchNormalization(momentum=0.99, epsilon=0.0001)(hdn)
hdn  = layers.Dense(911, activation='selu') (hdn)
hdn  = layers.Dense(713, activation='selu') (hdn)
# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(4,name="predictions")(hdn)
# Optimizer
adam_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001,
                                          amsgrad       = False)
# Set input and outpus
model = tf.keras.Model(inputs, predictions)
# configure loss
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss     = loss, 
              optimizer= adam_optimizer, 
              metrics  = [tf.keras.metrics.SparseCategoricalAccuracy()])
{% endhighlight python%}

But still i got error, huh i will need to try again tommorow.
{% highlight python%}
#logits and labels must have the same first dimension, got logits shape [64000,4] and labels shape [32]
{% endhighlight %}