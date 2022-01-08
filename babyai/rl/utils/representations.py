import networkx as nx
import numpy as np
import itertools
import pickle


class StateAction(object):

    def __init__(self, vocab_kge, vocab, vocab_rev, debug_mode):
        self.agent_graph_state = nx.DiGraph() # agent specific graph state
        self.graph_state = nx.DiGraph() # world graph state
        self.agent_graph_state_rep = []
        self.graph_state_rep = []
        self.vocab_kge = vocab_kge
        self.vocab_act = vocab
        self.vocab_act_rev = vocab_rev
        self.agent_vocab_kge = vocab_kge
        self.agent_adj_matrix = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        self.adj_matrix = np.zeros((len(self.agent_vocab_kge['entity']), len(self.agent_vocab_kge['entity'])))
        self.ego_rel_indx = {-1: 'agent', 0: 'left', 1: 'front', 2: 'right', 3: 'far_left', 4: 'front_left',
                        5: 'far_front', 6: 'front_right', 7: 'far_right' }
        self.ego_rel_map = np.array([4, 4, 4, 5, 6, 6, 6, 4, 4, 4, 5, 6, 6, 6,4, 4, 4, 5, 6, 6, 6,4, 4, 4, 5, 6, 6,
                                     6,4, 4, 4, 5, 6, 6, 6, 4, 4, 4, 1, 6, 6, 6,3,3,0,-1,2,7,7])
        self.wall_imp_locs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0])
        self.room = "" # agent's room
        self.entry_door = "" # entry door to current room
        self.door_status = {} # id to obj_desc (current door status including color)
        self.obj_to_rooms = {} # one room in case of obj, 2 in case of door
        self.pos_to_rooms = {} # pos should be in tuple
        self.env_hash = ''
        self.carrying = False
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.prev_state_pickle = pickle.dumps(('prev_env', 'pre_action')) # env, action in pickle form

    def visualize(self, graph_state, node_size=200, font_size=10):
        import matplotlib
        matplotlib.use('Qt5Agg')  # enable for interactive plots
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(graph_state)
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        edge_labels = {e: graph_state.edges[e]['rel'] for e in graph_state.edges}
        edge_labels = {k : v.split()[0] for k, v in edge_labels.items()}
        print(edge_labels)
        nx.draw_networkx_edge_labels(graph_state, pos, edge_labels)
        nx.draw(graph_state, pos=pos, with_labels=True, node_size=node_size, font_size=font_size)
        plt.tight_layout()
        plt.show()

    def get_obj_desc_old(self, obj, env):
        if obj.type == 'wall':
            return {'name': 'Wall'}
        elif obj.type == 'door':
            obj_id = env.door_to_idx(id(obj))
            return {'name': 'Door_' + str(obj_id), 'color': obj.color}
        elif obj.type == 'ball':
            obj_id = env.ball_to_idx(id(obj))
            if obj_id > 5:
                raise ValueError(f'More than 5 objects of same type, {obj.type} in a room!')
            return {'name': 'Ball_' + str(obj_id), 'color': obj.color}
        elif obj.type == 'box':
            obj_id = env.box_to_idx(id(obj))
            if obj_id > 5:
                raise ValueError(f'More than 5 objects of same type, {obj.type} in a room!')
            return {'name': 'Box_' + str(obj_id), 'color': obj.color}
        elif obj.type == 'key':
            obj_id = env.key_to_idx(id(obj))
            if obj_id > 5:
                raise ValueError(f'More than 5 objects of same type, {obj.type} in a room!')
            return {'name': 'Key_' + str(obj_id), 'color': obj.color}
        else:
            raise KeyError('Unknown obj type')

    def get_obj_desc(self, obj):
        if obj.type == 'wall':
            return 'wall'
        elif obj.type == 'door':
            return obj.color + '_door'
        elif obj.type == 'ball':
            return obj.color + '_ball'
        elif obj.type == 'box':
            return obj.color + '_box'
        elif obj.type == 'key':
            return obj.color + '_key'
        else:
            raise KeyError('Unknown obj type')

    def get_room_from_obj(self, obj, obj_id, env):
        try:
            return self.obj_to_rooms[obj_id]
        except:
            if obj.type == 'door':
                rooms = self.get_rooms_from_door(obj, env)
                self.obj_to_rooms[obj_id] = rooms
                return rooms
            elif obj.type != 'wall':
                room = self.room_from_pos(obj.cur_pos, env)
                self.obj_to_rooms[obj_id] = room
                return room
            else:
                return None

    def room_from_pos(self, pos, env):
        pos = tuple(pos)
        try:
            return self.pos_to_rooms[pos]
        except:
            room = env.env.room_from_pos(*pos)
            room = 'Room' + str(env.room_to_idx(hex(id(room))))
            self.pos_to_rooms[pos] = room
            return room

    def get_rooms_from_door(self, obj, env):
        x, y = obj.cur_pos
        sur_pos = [(x+1, y), (x-1,y) ,(x, y+1), (x, y-1)]
        rooms = []
        for pos in sur_pos:
            obj = env.env.grid.get(*pos)
            if obj == None or obj.type is not 'wall':
                rooms.append(self.room_from_pos(pos, env))
        return rooms

    def update_state(self, obs, prev_action, env, cache=None, print_obs=False):

        prev_hash = self.env_hash
        self.env_hash = env.env.hash()
        if prev_hash == self.env_hash:
            return [], [], []
        self.agent_graph_state = nx.DiGraph() # reset agent graph for every obs
        prev_room = self.room

        # img = env.env.render()
        # env.env.render(close=True)

        if print_obs:
            from matplotlib import pyplot as plt
            img = env.env.render()
            plt.imshow(img)
            plt.show()
            plt.savefig('example_grid_all_1.png')

        # graph_copy = self.graph_state.copy()
        # con_cs = [graph_copy.subgraph(c) for c in nx.weakly_connected_components(graph_copy)] # connected components
        #
        # prev_room_subgraph = None
        # prev_you_subgraph = None
        #
        # for con_c in con_cs:
        #     for node in con_c.nodes:
        #         node = set(str(node).split())
        #         if set(prev_room.split()).issubset(node):
        #             prev_room_subgraph = nx.induced_subgraph(graph_copy, con_c.nodes)
        #
        # for edge in self.graph_state.edges:
        #     if 'You' in edge[0]:
        #         raise ValueError('You should not be in grraph state')
        #         graph_copy.remove_edge(*edge)
        #
        # self.graph_state = graph_copy

        # record the previous state for debugging purposes only
        if self.debug_mode:
            prev_env, pre_action = pickle.loads(self.prev_state_pickle)
            # prev_state_pickle = pickle.dump(self.prev_state, open('prev_state.pkl', 'wb'))
            # prev_env, pre_action = pickle.load(open('prev_state.pkl', 'rb'))
            self.prev_state_pickle = pickle.dumps((env, prev_action))

        # update the previous carrying flag
        prev_carrying = self.carrying

        agent_pos = env.env.agent_pos
        self.room = self.room_from_pos(agent_pos, env)
        curr_cell = env.env.grid.get(*agent_pos)
        if curr_cell is not None and curr_cell.type == 'door':
            self.entry_door = curr_cell

        agent_rules = [] # agent specific
        add_rules = [] # world specific
        remove_rules = []
        agent_rules.append(('You', 'in', self.room))

        self.carrying = env.env.carrying # obj
        if prev_action == env.env.Actions.pickup and prev_carrying is None and self.carrying:
            obj_desc = self.get_obj_desc(self.carrying)
            obj_id = hex(id(self.carrying))
            to_be_removed_room = self.obj_to_rooms.pop(obj_id) # remove this objects room info from the lookup
            # remove this id and room from this obj_desc
            self.graph_state.nodes[obj_desc]['info'].pop(obj_id)
            if not to_be_removed_room in self.graph_state.nodes[obj_desc]['info'].values(): # entities such as room1,
                # but not [room1, # room2], where first one corr to objs and 2nd to doors. as this is self.carrying obj, only 1st are present
                remove_rules.append((to_be_removed_room, f'has {self.carrying.cur_pos}', obj_desc))
        if self.carrying:
            obj_desc = self.get_obj_desc(self.carrying)
            agent_rules.append(('You', 'have', obj_desc))


        # if you toggle a door and its status changes, you need to remove the previouos status from the graph
        front_cell = env.env.grid.get(*env.env.front_pos)
        door_in_front = front_cell != None and front_cell.type == 'door'
        if prev_action == env.env.Actions.toggle and door_in_front:
            # find previous status of this door to remove from room; current status is added as per the obs objs below
            obj_id = hex(id(front_cell))
            obj_desc = self.door_status[obj_id] # previous status
            # remove that this id and room from that node(prev_status)
            to_be_removed_rooms = self.graph_state.nodes[obj_desc]['info'].pop(obj_id)
            assoc_rooms = list(itertools.chain(*self.graph_state.nodes[obj_desc]['info'].values()))
            for room in to_be_removed_rooms:
                if not room in assoc_rooms:
                    remove_rules.append((room, f'has {front_cell.cur_pos}', obj_desc))


        ego_grid = np.array(obs)
        wall_objs = np.array([True if obj != None and obj.type=='wall' else False for obj in ego_grid])
        self.wall_imp_locs = np.array([True if x == 1 else False for x in self.wall_imp_locs])
        imp_wall_objs_map = wall_objs & self.wall_imp_locs
        imp_wall_objs = ego_grid[imp_wall_objs_map]
        imp_wall_objs_rel = self.ego_rel_map[imp_wall_objs_map]
        all_locs_except_agentloc_map = self.ego_rel_map > 0
        viewable_objs_map = (ego_grid != None) & ~wall_objs & all_locs_except_agentloc_map
        viewable_objs = ego_grid[viewable_objs_map]
        viewable_objs_rel = self.ego_rel_map[viewable_objs_map]
        all_objs = list(np.concatenate((imp_wall_objs, viewable_objs)))
        all_objs_rel = list(np.concatenate((imp_wall_objs_rel, viewable_objs_rel)))
        for obj_i in range(len(all_objs)):
            obj = all_objs[obj_i]
            rel = all_objs_rel[obj_i]
            obj_desc = self.get_obj_desc(obj)
            obj_id = hex(id(obj))
            if obj.type == 'door':
                lock_status = 'locked' if obj.is_locked else 'unlocked'
                open_status = 'open' if obj.is_open else 'closed'
                status = lock_status + '_' + open_status
                obj_desc = status + '_' + obj_desc
                self.door_status[obj_id] = obj_desc # repetitive can be improved; check if prevac is toggle
            agent_obj_desc = obj_desc + '_to_the_' + self.ego_rel_indx[rel]
            agent_rules.append(('You', 'have', agent_obj_desc))
            if obj_desc != 'wall':
                obj_room = self.get_room_from_obj(obj, obj_id, env)
                if obj_desc in self.graph_state:
                    if not obj_id in self.graph_state.nodes[obj_desc]['info'].keys():
                        self.graph_state.nodes[obj_desc]['info'][obj_id] = self.obj_to_rooms[obj_id]
                        if obj.type == 'door':
                            add_rules.append((obj_room[0], f'has {obj.cur_pos}', obj_desc))
                            add_rules.append((obj_room[1], f'has {obj.cur_pos}', obj_desc))
                        else:
                            add_rules.append((obj_room, f'has {obj.cur_pos}', obj_desc))

                else:
                    self.graph_state.add_node(obj_desc, info={obj_id: self.obj_to_rooms[obj_id]})
                    if obj.type == 'door':
                        add_rules.append((obj_room[0], f'has {obj.cur_pos}', obj_desc))
                        add_rules.append((obj_room[1], f'has {obj.cur_pos}', obj_desc))
                    else:
                        add_rules.append((obj_room, f'has {obj.cur_pos}', obj_desc))

        # room has exit to lving room
        # you in open field; so room = open_field
        # you have ball; equivalent of carrying
        dir_dict = {0: 'right', 1:'behind', 2: 'left', 3: 'front'}

        if self.room != prev_room and prev_room != '':
            add_rules.append((prev_room, 'has', self.room + '_to_the_' + dir_dict[env.env.agent_dir]))
            door_desc = self.get_obj_desc(self.entry_door)
            # obj_id = hex(id(self.entry_door))
            add_rules.append((prev_room, 'has', door_desc + '_to_' + self.room))
            add_rules.append((self.room, 'has', door_desc + '_to_' + prev_room))
            # if prev_room_subgraph is not None:
            #     for ed in prev_room_subgraph.edges:
            #         rules.append((ed[0], prev_room_subgraph[ed]['rel'], ed[1]))

        for rule in remove_rules:
            self.graph_state.remove_edge(rule[0], rule[2])

        for rule in add_rules:
            self.graph_state.add_edge(rule[0], rule[2], rel=rule[1])

        for rule in agent_rules:
            self.agent_graph_state.add_edge(rule[0], rule[2], rel=rule[1])

        # for node in self.graph_state.nodes:
        #     if 'unlocked' in node:
        #         in_degree = self.graph_state.in_degree(node)
        #         assert in_degree != 1, f'{node}: {in_degree}'
        #
        return add_rules, remove_rules, agent_rules

    def get_state_rep_kge(self):
        ret = []
        self.adj_matrix = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))

        for u, v in self.graph_state.edges:

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

            if u not in self.agent_vocab_kge['entity'].keys() or v not in self.agent_vocab_kge['entity'].keys():
                break

            u_idx = self.agent_vocab_kge['entity'][u]
            v_idx = self.agent_vocab_kge['entity'][v]
            self.agent_adj_matrix[u_idx][v_idx] = 1

            ret.append(u)
            ret.append(v)

        return list(set(ret))

    def get_action_rep(self, action, env):
        if action is None:
            return self.vocab_act['']
        else:
            return self.vocab_act[env.Actions(action).name]

    def step(self, obs, prev_action, env, cache=None):
        add_rules, remove_rules, agent_rules = self.update_state(obs, prev_action, env, cache)
        self.graph_state_rep = self.get_state_rep_kge(), self.adj_matrix
        self.agent_graph_state_rep = self.get_agent_state_rep_kge(), self.agent_adj_matrix
        return add_rules, remove_rules, agent_rules


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


