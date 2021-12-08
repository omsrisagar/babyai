import networkx as nx
import numpy as np
from fuzzywuzzy import fuzz


class StateAction(object):

    def __init__(self, spm, vocab, vocab_rev, tsv_file, tsv_file_agent):
        self.agent_graph_state = nx.DiGraph() # agent specific graph state
        self.graph_state = nx.DiGraph() # world graph state
        self.agent_graph_state_rep = []
        self.graph_state_rep = []
        self.sp = spm
        self.vocab_act = vocab
        self.vocab_act_rev = vocab_rev
        self.vocab_kge = self.load_vocab_kge(tsv_file)
        self.agent_vocab_kge = self.vocab_kge
        self.agent_adj_matrix = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        self.adj_matrix = np.zeros((len(self.agent_vocab_kge['entity']), len(self.agent_vocab_kge['entity'])))
        self.ego_rel_indx = {-1: 'agent', 0: 'left', 1: 'front', 2: 'right', 3: 'far_left', 4: 'front_left',
                        5: 'far_front', 6: 'front_right', 7: 'far_right' }
        self.ego_rel_map = np.array([4, 4, 4, 5, 6, 6, 6, 4, 4, 4, 5, 6, 6, 6,4, 4, 4, 5, 6, 6, 6,4, 4, 4, 5, 6, 6,
                                     6,4, 4, 4, 5, 6, 6, 6, 4, 4, 4, 1, 6, 6, 6,3,3,0,-1,2,7,7])
        self.room = ""

    def visualize(self, graph_state):
        # import matplotlib.pyplot as plt
        pos = nx.spring_layout(graph_state)
        edge_labels = {e: graph_state.edges[e]['rel'] for e in graph_state.edges}
        print(edge_labels)
        nx.draw_networkx_edge_labels(graph_state, pos, edge_labels)
        nx.draw(graph_state, pos=pos, with_labels=True, node_size=200, font_size=10)
        #plt.show()

    def load_vocab_kge(self, tsv_file):
        ent = {}
        rel = {}
        id = 0
        with open(tsv_file, 'r') as f:
            for line in f:
                ent[line.strip()] = id
                rel[line.strip()] = id
                id += 1
        return {'entity': ent, 'relation': rel}

    def get_obj_desc(self, obj, env):
        if obj.type == 'wall':
            return {'name': 'Wall'}
        elif obj.type == 'door':
            obj_id = env.door_to_idx(id(obj))
            return {'name': 'Door ' + str(obj_id), 'color': obj.color}
        elif obj.type == 'ball':
            obj_id = env.ball_to_idx(id(obj))
            if obj_id > 5:
                raise ValueError(f'More than 5 objects of same type, {obj.type} in a room!')
            return {'name': 'Ball ' + str(obj_id), 'color': obj.color}
        elif obj.type == 'box':
            obj_id = env.box_to_idx(id(obj))
            if obj_id > 5:
                raise ValueError(f'More than 5 objects of same type, {obj.type} in a room!')
            return {'name': 'Box ' + str(obj_id), 'color': obj.color}
        elif obj.type == 'key':
            obj_id = env.key_to_idx(id(obj))
            if obj_id > 5:
                raise ValueError(f'More than 5 objects of same type, {obj.type} in a room!')
            return {'name': 'Key ' + str(obj_id), 'color': obj.color}
        else:
            raise KeyError('Unknown obj type')

    def update_state(self, obs, prev_action, env, cache=None):

        prev_room = self.room

        graph_copy = self.graph_state.copy()
        con_cs = [graph_copy.subgraph(c) for c in nx.weakly_connected_components(graph_copy)] # connected components

        prev_room_subgraph = None
        prev_you_subgraph = None

        for con_c in con_cs:
            for node in con_c.nodes:
                node = set(str(node).split())
                if set(prev_room.split()).issubset(node):
                    prev_room_subgraph = nx.induced_subgraph(graph_copy, con_c.nodes)

        for edge in self.graph_state.edges:
            if 'you' in edge[0]:
                graph_copy.remove_edge(*edge)

        self.graph_state = graph_copy

        room = env.room_from_pos(*env.agent_pos)
        room_id = env.room_to_idx(id(room))
        self.room = 'Room ' + str(room_id+1)


        if cache is None:
            agent_rules = [] # agent specific
            rules = [] # world specific
            agent_rules.append(('You', 'in', self.room))
            if env.carrying:
                obj_desc = self.get_obj_desc(env.carrying, env)
                agent_rules.append(('You', 'has_have', obj_desc['name']))

            ego_grid = np.array(obs)
            viewable_objs_map = ego_grid != None
            viewable_objs = ego_grid[viewable_objs_map]
            viewable_objs_rel = self.ego_rel_map[viewable_objs_map]
            for obj_i in range(len(viewable_objs)):
                obj = viewable_objs[obj_i]
                rel = viewable_objs_rel[obj_i]
                obj_desc = self.get_obj_desc(obj, env)
                name = obj_desc['name']
                agent_rules.append(('You', rel, name))
                if name != 'Wall':
                    rules.append((self.room, 'has', name))
                    rules.append((name, 'color', obj_desc['color']))
                if obj.type == 'door':
                    lock_status = 'locked' if obj.is_locked else 'not_locked'
                    open_status = 'open' if obj.is_open else 'not_open'
                    rules.append((name, 'is', lock_status))
                    rules.append((name, 'is', open_status))
        else:
            agent_rules, rules = cache

        in_aliases = ['are in', 'are facing', 'are standing', 'are behind', 'are above', 'are below', 'are in front']

        in_rl = []
        in_flag = False

        # room has exit to lving room
        # you in open field; so room = open_field
        # you have ball; equivalent of carrying
        dir_dict = {0: 'right', 1:'behind', 2: 'left', 3: 'front'}

        if prev_action == env.actions.toggle and self.room != prev_room:
            rules.append((prev_room, dir_dict[env.agent_dir], self.room))
            if prev_room_subgraph is not None:
                for ed in prev_room_subgraph.edges:
                    rules.append((ed[0], prev_room_subgraph[ed]['rel'], ed[1]))

        for o in objs:
            #if o != 'all':
            rules.append((str(o), 'in', room))

        add_rules = rules

        for rule in add_rules:
            u = '_'.join(str(rule[0]).split())
            v = '_'.join(str(rule[2]).split())
            if u in self.vocab_kge['entity'].keys() and v in self.vocab_kge['entity'].keys():
                if u != 'it' and v != 'it':
                    self.graph_state.add_edge(rule[0], rule[2], rel=rule[1])

        return add_rules

    def get_state_rep_kge(self):
        ret = []
        self.adj_matrix = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))

        for u, v in self.graph_state.edges:
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())

            if u not in self.vocab_kge['entity'].keys() or v not in self.vocab_kge['entity'].keys():
                raise KeyError(f'u: {u} or v: {v} not in vocab_kge')

            u_idx = self.vocab_kge['entity'][u]
            v_idx = self.vocab_kge['entity'][v]
            self.adj_matrix[u_idx][v_idx] = 1

            ret.append(self.vocab_kge['entity'][u])
            ret.append(self.vocab_kge['entity'][v])

        return list(set(ret))

    def get_state_kge(self):
        ret = []
        self.adj_matrix = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))

        for u, v in self.graph_state.edges:
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())

            if u not in self.vocab_kge['entity'].keys() or v not in self.vocab_kge['entity'].keys():
                break

            u_idx = self.vocab_kge['entity'][u]
            v_idx = self.vocab_kge['entity'][v]
            self.adj_matrix[u_idx][v_idx] = 1

            ret.append(u)
            ret.append(v)

        return list(set(ret))

    def get_agent_state_rep_kge(self):
        ret = []
        self.agent_adj_matrix = np.zeros((len(self.agent_vocab_kge['entity']), len(self.agent_vocab_kge['entity'])))

        for u, v in self.agent_graph_state.edges:
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())

            if u not in self.agent_vocab_kge['entity'].keys() or v not in self.agent_vocab_kge['entity'].keys():
                raise KeyError(f'u: {u} or v: {v} not in vocab_kge')

            u_idx = self.agent_vocab_kge['entity'][u]
            v_idx = self.agent_vocab_kge['entity'][v]
            self.agent_adj_matrix[u_idx][v_idx] = 1

            ret.append(self.agent_vocab_kge['entity'][u])
            ret.append(self.agent_vocab_kge['entity'][v])

        return list(set(ret))

    def get_agent_state_kge(self):
        ret = []
        self.agent_adj_matrix = np.zeros((len(self.agent_vocab_kge['entity']), len(self.agent_vocab_kge['entity'])))

        for u, v in self.agent_graph_state.edges:
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())

            if u not in self.agent_vocab_kge['entity'].keys() or v not in self.agent_vocab_kge['entity'].keys():
                break

            u_idx = self.agent_vocab_kge['entity'][u]
            v_idx = self.agent_vocab_kge['entity'][v]
            self.agent_adj_matrix[u_idx][v_idx] = 1

            ret.append(u)
            ret.append(v)

        return list(set(ret))

    def get_action_rep_drqa(self, action):

        action_desc_num = 20 * [0]
        action = str(action)

        for i, token in enumerate(action.split()[:20]):
            short_tok = token[:self.max_word_len]
            action_desc_num[i] = self.vocab_act_rev[short_tok] if short_tok in self.vocab_act_rev else 0

        return action_desc_num

    def step(self, obs, prev_action, env, cache=None):
        rules = self.update_state(obs, prev_action, env, cache)
        self.graph_state_rep = self.get_state_rep_kge(), self.adj_matrix
        self.agent_graph_state_rep = self.get_agent_state_rep_kge(), self.agent_adj_matrix
        return rules


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


