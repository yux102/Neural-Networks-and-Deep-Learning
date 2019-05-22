import tensorflow as tf
import re

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    #print ("original review is:\n",review)
    #print ()
    ## start to preprocess
    #re.sub(pattern, repl, string, count=0, flags=0)¶
    review = review.lower()
    review = re.sub(r"<br />", " ", review)

    review = re.sub(r"won't", "will not ", review)
    review = re.sub(r"won't", "will not ", review)
    review = re.sub(r"don't", "do not ", review)
    review = re.sub(r"didn't", "did not ", review)
    review = re.sub(r"'s", " ", review)
    review = re.sub(r"'ve", " have ", review)
    review = re.sub(r"'ll", " will ", review)
    review = re.sub(r"'re", " are ", review)
    review = re.sub(r"'d", " ", review)
    review = re.sub(r"i'm", "i am ", review)

    review = re.sub('[,.";!?:\(\)-]+', " ", review)
    review = re.sub('[0-9]', " ", review)
    review = re.sub(' [a-z] ', " ", review)
    processed_review = [word for word in review.split() if word not in stop_words]

    #processed_review = " ".join(processed_review)
    return processed_review


def option1(input_data, dropout_keep_prob):
    """rnn + LSTM -> FC"""
    hidden_size = 128

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)

    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, input_data, dtype=tf.float32)
    W = tf.Variable(tf.random_normal((hidden_size * 2, 2), mean=0.01, stddev=0.01, seed=1))
    b = tf.Variable(tf.constant(0.01, shape=(2,)))
    logits = tf.matmul(outputs[-1], W) + b

    return logits

def option2(input_data, dropout_keep_prob):
    """Bi-LSTM -> rnn-LSTM -> FC"""
    #1. Bi-LSTM layer
    hidden_size = 128
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=dropout_keep_prob)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=dropout_keep_prob)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, input_data, dtype=tf.float32) #[batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
    print("outputs:===>", outputs)
    output_rnn = tf.concat(outputs, axis=2) #[batch_size,sequence_length,hidden_size*2]
    print("output_rnn:===>", output_rnn)

    #2. Second LSTM layer
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size*2)
    rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=dropout_keep_prob)
    _, final_state_c_h = tf.nn.dynamic_rnn(rnn_cell, output_rnn, dtype=tf.float32)
    final_state = final_state_c_h[1]
    print("final_state:===>", final_state)

    #3. FC layer
    output = tf.layers.dense(final_state, hidden_size*2, activation=tf.nn.tanh)
    print("output:===>", output)

    #4. logits(use linear layer)
    W = tf.Variable(tf.random_normal((hidden_size * 2, 2), mean=0.01, stddev=0.01, seed=1))
    b = tf.Variable(tf.constant(0.01, shape=(2,)))
    with tf.name_scope("output"): #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
        logits = tf.sigmoid(tf.matmul(output, W) + b)  # [batch_size,num_classes]
    print("logits:===>", logits)
    return logits



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    input_data = tf.placeholder(tf.float32, shape=(None, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE), name="input_data")
    labels = tf.placeholder(tf.float32, shape=(None, 2), name="labels")
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep_prob")

    logits = option2(input_data, dropout_keep_prob)

    ## define loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits), name="loss")
    print("loss:===>", loss)

    ## define optimizer
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    print("optimizer:===>", optimizer)

    ## define accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    Accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name="accuracy")
    print("Accuracy:===>", Accuracy)
    '''
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    '''
    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss

#define_graph()
