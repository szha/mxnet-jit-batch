from mxnet.gluon import nn
from fold.rnn import ChildSumLSTMCell

# module for distance-angle similarity
class Similarity(nn.HybridBlock):
    def __init__(self, sim_hidden_size, rnn_hidden_size, num_classes):
        super(Similarity, self).__init__()
        with self.name_scope():
            self.wh = nn.Dense(sim_hidden_size, in_units=2*rnn_hidden_size, prefix='sim_embed_')
            self.wp = nn.Dense(num_classes, in_units=sim_hidden_size, prefix='sim_out_')

    def hybrid_forward(self, F, lvec, rvec):
        # lvec and rvec will be tree_lstm cell states at roots
        mult_dist = F.broadcast_mul(lvec, rvec)
        abs_dist = F.abs(lvec-rvec)
        vec_dist = F.concat(*[mult_dist, abs_dist],dim=1)
        out = F.log_softmax(self.wp(F.sigmoid(self.wh(vec_dist))))
        return out

# putting the whole model together
class SimilarityTreeLSTM(nn.Block):
    def __init__(self, sim_hidden_size, rnn_hidden_size, embed_in_size, embed_dim, num_classes):
        super(SimilarityTreeLSTM, self).__init__()
        with self.name_scope():
            self.embed = nn.Embedding(embed_in_size, embed_dim, prefix='word_embed_')
            self.childsumtreelstm = ChildSumLSTMCell(rnn_hidden_size, input_size=embed_dim)
            self.similarity = Similarity(sim_hidden_size, rnn_hidden_size, num_classes)
            self.embed.hybridize()
            self.similarity.hybridize()

    def forward(self, F, l_inputs, r_inputs, l_tree, r_tree):
        l_len, r_len = len(l_inputs), len(r_inputs)
        embeddings = self.embed(F.concat(l_inputs, r_inputs, dim=0)).split(l_len+r_len, axis=0)
        l_embed = F.concat(*embeddings[:l_len], dim=0)
        r_embed = F.concat(*embeddings[l_len:(l_len+r_len)], dim=0)
        lstate = self.childsumtreelstm.encode(F, l_embed, l_tree)[1][1]
        rstate = self.childsumtreelstm.encode(F, r_embed, r_tree)[1][1]
        output = self.similarity(lstate, rstate)
        return output

    def fold_encode(self, fold, F, l_inputs, r_inputs, l_tree, r_tree):
        l_len, r_len = len(l_inputs), len(r_inputs)
        embeddings = self.embed(F.concat(l_inputs, r_inputs, dim=0)).split(l_len+r_len, axis=0)
        l_embed = embeddings[:l_len]
        r_embed = embeddings[l_len:(l_len+r_len)]
        lstate = self.childsumtreelstm.fold_encode(fold, F, l_embed, l_tree)[1][1]
        rstate = self.childsumtreelstm.fold_encode(fold, F, r_embed, r_tree)[1][1]
        out = fold.add(self.similarity, 0, l_inputs.context, lstate, rstate)
        return out
