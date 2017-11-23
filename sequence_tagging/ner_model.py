import numpy as np
import tensorflow as tf


class NERModel(object):
    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx2tag = {idx: tag for tag, idx in self.config.vocab_tags.items()}

    def build(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")

        self.sequence_lenghts = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.labels = tf.placeholder(tf.int32, shape=[None, None])

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

        ###add word embeddings###
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                _word_embeddings = tf.get_variable(name="_word_embeddings", dtype=tf.float32,
                                                   shape=[self.config.nwords, self.config.dimwords])
            else:
                _word_embeddings = tf.Variable(self.config.embeddings, name="_word_embeddings", dtype=tf.float32,
                                               trainable=False)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name="word_embeddings")

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

        ####reflect to tag scores####
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_embeddings,
                                                                        sequence_length=self.sequence_lenghts,
                                                                        dtype=tf.float)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W=tf.get_variable("W",dtype=tf.float32,shape=[2*self.config.hidden_size_lstm,self.config.ntags])

            b=tf.get_variable("b",shape=[self.config.ntags],dtype=tf.float32,initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])