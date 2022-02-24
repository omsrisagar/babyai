import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import babyai.rl
from babyai.rl.utils.supervised_losses import required_heads


# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=imm_channels,
            kernel_size=(3, 3), padding=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.bn1_1d = nn.BatchNorm1d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features,
            kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.bn2_1d = nn.BatchNorm1d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y, use_conv=True):
        if use_conv:
            x = F.relu(self.bn1(self.conv1(x)))
            # x= F.relu(self.conv1(x))
            x = self.conv2(x)
            weight = self.weight(y).unsqueeze(2).unsqueeze(3)
            bias = self.bias(y).unsqueeze(2).unsqueeze(3)
            out = x * weight + bias
            return F.relu(self.bn2(out))
            # return F.relu(out)
        else:
            # x = F.relu(x)
            x = F.relu(self.bn1_1d(x))
            weight = self.weight(y)
            bias = self.bias(y)
            out = x * weight + bias
            # return F.relu(out)
            return F.relu(self.bn2_1d(out))

class StateNetwork(nn.Module):
    def __init__(self, gat_emb_size, word_emb, vocab, vocab_kge_file, embedding_size, dropout_ratio):
        super(StateNetwork, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.embedding_size = embedding_size # 128
        self.dropout_ratio = dropout_ratio
        self.gat_emb_size = gat_emb_size
        #self.params = params
        self.gat = GAT(gat_emb_size, 3, dropout_ratio, 0.2, 1)
        self.word_emb = word_emb # 100 x 128
        self.vocab_kge = self.load_vocab_kge(vocab_kge_file)
        self.state_ent_emb = nn.Embedding.from_pretrained(torch.zeros((len(self.vocab_kge),
                                                                       self.embedding_size)), freeze=False) # 438 x 128
        self.fc1 = nn.Linear(self.state_ent_emb.weight.size()[0] * 3 * 1, self.embedding_size)
        # self.init_state_ent_emb(self.embedding_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_state_ent_emb(self, emb_size):
        embeddings = torch.zeros((len(self.vocab_kge), emb_size))
        for i in range(len(self.vocab_kge)):
            graph_node_text = self.vocab_kge[i].split('_')
            graph_node_ids = []
            for w in graph_node_text:
                graph_node_ids.append(self.vocab[w]) # each word must be in vocab
                # if w in self.vocab.keys():
                #     if self.vocab[w] < len(self.vocab) - 2:
                #         graph_node_ids.append(self.vocab[w])
                #     else:
                #         graph_node_ids.append(1)
                # else:
                #     graph_node_ids.append(1)
            graph_node_ids = torch.LongTensor(graph_node_ids)
            cur_embeds = self.word_emb(graph_node_ids)

            cur_embeds = cur_embeds.mean(dim=0)
            embeddings[i, :] = cur_embeds
        self.state_ent_emb = nn.Embedding.from_pretrained(embeddings, freeze=False)

    def load_vocab_kge(self, vocab_file):
        ent = {}
        id = 0
        with open(vocab_file, 'r') as f:
            for line in f:
                ent[id] = line.strip()
                id += 1
        return ent

    def forward(self, graph_rep):
        out = []
        for g in graph_rep:
            # node_feats, adj = g # node_feats are not used, but I think state_ent_emb and adj can express that info
            adj = g
            # adj = torch.tensor(adj, dtype=torch.int, device=self.device)
            x = self.gat.forward(self.state_ent_emb.weight, adj).view(-1)
            out.append(x.unsqueeze_(0))
        out = torch.cat(out)
        ret = self.fc1(out)
        return ret

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout)
        return x

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty((in_features, out_features), device=self.device),
                                                      gain=np.sqrt(2.0)), requires_grad=True) # 32 x 3
        self.a = nn.Parameter(nn.init.xavier_uniform_(torch.empty((2*out_features, 1), device=self.device),
                                                      gain=np.sqrt( 2.0)), requires_grad=True) # 6 x 1

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = torch.zeros_like(e)
        zero_vec = zero_vec.fill_(9e-15)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class ImageBOWEmbedding(nn.Module):
   def __init__(self, max_value, embedding_dim):
       super().__init__()
       self.max_value = max_value
       self.embedding_dim = embedding_dim
       self.embedding = nn.Embedding(3 * max_value, embedding_dim)
       self.apply(initialize_parameters)

   def forward(self, inputs):
       offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(inputs.device)
       inputs = (inputs + offsets[None, :, None, None]).long()
       return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)


class ACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128, gat_emb_size=64,agent_gat_emb_size=32,
                 dropout_ratio=0.2, vocab=None, vocab_kge_file=None, use_obs_image=True, use_agent_graph=True,
                 use_world_graph=True, no_film=False,
                 use_instr=False, lang_model="gru", use_memory=False, arch="bow_endpool_res", aux_info=None):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        endpool = 'endpool' in arch
        use_bow = 'bow' in arch
        pixel = 'pixel' in arch
        self.res = 'res' in arch

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model
        self.aux_info = aux_info
        if self.res and image_dim != 128:
            raise ValueError(f"image_dim is {image_dim}, expected 128")
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim
        self.gat_emb_size = gat_emb_size
        self.agent_gat_emb_size = agent_gat_emb_size
        self.vocab = vocab
        self.dropout_ratio = dropout_ratio
        self.use_obs_image = use_obs_image
        self.use_agent_graph = use_agent_graph
        self.use_world_graph = use_world_graph
        self.no_film = no_film

        self.obs_space = obs_space

        for part in self.arch.split('_'):
            if part not in ['original', 'bow', 'pixels', 'endpool', 'res']:
                raise ValueError("Incorrect architecture name: {}".format(self.arch))

        # if not self.use_instr:
        #     raise ValueError("FiLM architecture can be used when instructions are enabled")
        self.image_conv = nn.Sequential(*[
            *([ImageBOWEmbedding(obs_space['image'], 128)] if use_bow else []),
            *([nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=(8, 8),
                stride=8, padding=0)] if pixel else []),
            nn.Conv2d(
                in_channels=128 if use_bow or pixel else 3, out_channels=128,
                kernel_size=(3, 3) if endpool else (2, 2), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
        ])
        self.film_pool = nn.MaxPool2d(kernel_size=(7, 7) if endpool else (2, 2), stride=2)

        # Define instruction embedding
        if self.use_instr:
            if self.lang_model in ['gru', 'bigru', 'attgru']:
                self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
                if self.lang_model in ['gru', 'bigru', 'attgru']:
                    gru_dim = self.instr_dim
                    if self.lang_model in ['bigru', 'attgru']:
                        gru_dim //= 2
                    self.instr_rnn = nn.GRU(
                        self.instr_dim, gru_dim, batch_first=True,
                        bidirectional=(self.lang_model in ['bigru', 'attgru']))
                    self.final_instr_dim = self.instr_dim
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.instr_convs = nn.ModuleList([
                        nn.Conv2d(1, kernel_dim, (K, self.instr_dim)) for K in kernel_sizes])
                    self.final_instr_dim = kernel_dim * len(kernel_sizes)

            if self.lang_model == 'attgru':
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

            if not self.no_film:
                num_module = 2
                self.controllers = []
                for ni in range(num_module):
                    mod = FiLM(
                        in_features=self.final_instr_dim,
                        out_features=128 if ni < num_module-1 else self.image_dim,
                        in_channels=128, imm_channels=128)
                    self.controllers.append(mod)
                    self.add_module('FiLM_' + str(ni), mod)

        self.state_gat = StateNetwork(self.gat_emb_size, self.word_embedding, self.vocab, vocab_kge_file,
                                      self.instr_dim, self.dropout_ratio)
        self.agent_state_gat = StateNetwork(self.agent_gat_emb_size, self.word_embedding, self.vocab,
                                            vocab_kge_file, self.instr_dim, self.dropout_ratio)

        # Define memory and resize image embedding
        self.embedding_size = self.image_dim # 128
        in_dim = 0
        if self.use_obs_image:
            in_dim += self.embedding_size
        if self.use_agent_graph or self.use_world_graph: # for prev action embedding
            in_dim += self.embedding_size
        if self.use_agent_graph:
            in_dim += self.embedding_size
        if self.use_world_graph:
            in_dim += self.embedding_size
        if self.no_film: # as we need to add the instr embedding to the input of LSTM
            in_dim += self.instr_dim
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(in_dim, self.memory_dim)
            self.embedding_size = self.semi_memory_size
        else:
            self.fc_no_mem = nn.Linear(in_dim, self.semi_memory_size)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, prev_action, graph_rep, agent_graph_rep, instr_embedding=None):
        if self.use_instr and instr_embedding is None:
            instr_embedding = self._get_instr_embedding(obs.instr) # 64 x 128
        if self.use_instr and self.lang_model == "attgru":
            # outputs: B x L x D
            # memory: B x M
            mask = (obs.instr != 0).float()
            # The mask tensor has the same length as obs.instr, and
            # thus can be both shorter and longer than instr_embedding.
            # It can be longer if instr_embedding is computed
            # for a subbatch of obs.instr.
            # It can be shorter if obs.instr is a subbatch of
            # the batch that instr_embeddings was computed for.
            # Here, we make sure that mask and instr_embeddings
            # have equal length along dimension 1.
            mask = mask[:, :instr_embedding.shape[1]]
            instr_embedding = instr_embedding[:, :mask.shape[1]]

            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        final_emb = torch.empty((obs.image.size()[0], 0), device=self.device)
        # image embedding
        if self.use_obs_image:
            img_emb = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

            if 'pixel' in self.arch:
                img_emb /= 256.0
            img_emb = self.image_conv(img_emb) # 64 (num_procs) x 128 x 7 x 7
            x = img_emb
            if not self.no_film:
            # if self.use_instr:
                for controller in self.controllers:
                    out = controller(x, instr_embedding) # same dim as img_emb 64x128x7x7
                    if self.res:
                        out += x
                    x = out
            x = F.relu(self.film_pool(x)) # 64 x 128 x 1 x 1
            x = x.reshape(x.shape[0], -1) # 64 x 128
            final_emb = torch.cat([final_emb, x], dim=1)

        # previous action embedding
        if self.use_agent_graph or self.use_world_graph:
            # prev_action_emb = self.word_embedding(torch.tensor(prev_action, device=self.device))
            prev_action_emb = self.word_embedding(prev_action)
            x = prev_action_emb
            if not self.no_film: # do this only if using film i.e., no_film=False
                for controller in self.controllers:
                    out = controller(x, instr_embedding, use_conv=False) # same dim as img_emb 64x128x7x7
                    if self.res:
                        out += x
                    x = out
            final_emb = torch.cat([final_emb, x], dim=1)

        # world_graph embedding
        if self.use_world_graph:
            world_graph_emb = self.state_gat.forward(graph_rep)
            x = world_graph_emb
            if not self.no_film:
                for controller in self.controllers:
                    out = controller(x, instr_embedding, use_conv=False) # same dim as img_emb 64x128x7x7
                    if self.res:
                        out += x
                    x = out
            final_emb = torch.cat([final_emb, x], dim=1)

        # agent graph embedding
        if self.use_agent_graph:
            agent_graph_emb = self.agent_state_gat.forward(agent_graph_rep)
            x = agent_graph_emb
            if not self.no_film:
                for controller in self.controllers:
                    out = controller(x, instr_embedding, use_conv=False) # same dim as img_emb 64x128x7x7
                    if self.res:
                        out += x
                    x = out
            final_emb = torch.cat([final_emb, x], dim=1)

        # add instruction embedding if no_film:
        if self.no_film:
            final_emb = torch.cat([final_emb, instr_embedding], dim=1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(final_emb, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = self.fc_no_mem(final_emb)

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions}

    def _get_instr_embedding(self, instr):
        lengths = (instr != 0).sum(1).long()
        if self.lang_model == 'gru':
            out, _ = self.instr_rnn(self.word_embedding(instr))
            hidden = out[range(len(lengths)), lengths-1, :]
            return hidden

        elif self.lang_model in ['bigru', 'attgru']:
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]
                outputs, final_states = self.instr_rnn(self.word_embedding(instr))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            return outputs if self.lang_model == 'attgru' else final_states

        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))
