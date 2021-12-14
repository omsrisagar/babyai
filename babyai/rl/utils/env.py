import collections
import numpy as np
from babyai.rl.utils.representations import StateAction
import redis
import random
import gym
from gym_minigrid.wrappers import RGBImgPartialObsWrapper


GraphInfo = collections.namedtuple('GraphInfo', 'prev_act_rep, graph_state, graph_state_rep, '
                                                'agent_graph_state, agent_graph_state_rep')

def gen_vocab_kge(vocab_kge_file):
    colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
    objs = ['ball', 'box', 'key']
    door = ['door']
    wall = ['wall']
    rels = ['left', 'front', 'right', 'far_left', 'front_left', 'far_front', 'front_right', 'far_right']
    # locs = ['top_left', 'top_center', 'top_right', 'center_left', 'center', 'center_right', 'bottom_left',
    #         'bottom_center', 'bottom_right']
    locs = ['right', 'behind', 'left', 'front']
    door_status = ['locked_closed', 'unlocked_closed', 'unlocked_open']
    vocab_kge = []
    # only obj desc
    for c in colors:
        for o in objs + door:
            obj_desc = c + '_' + o
            vocab_kge.append(obj_desc)
            if o is 'door':
                for ds in door_status:
                    full_obj_desc = ds + '_' + obj_desc
                    vocab_kge.append(full_obj_desc)
    # obj desc + agent relation
    for o in objs + door + wall:
        for r in rels:
            if o is 'wall':
                vocab_kge.append('wall' + '_to_the_' + r)
            else:
                for c in colors:
                    obj_desc = c + '_' + o
                    if o is 'door':
                        for ds in door_status:
                            full_obj_desc = ds + '_' + obj_desc
                            vocab_kge.append(full_obj_desc + '_to_the_' + r)
                    else:
                        vocab_kge.append(obj_desc + '_to_the_' + r)
    # roomm locations
    # for loc in locs:
    #     vocab_kge.append(loc + '_' + 'room')
    for i in range(1, 10):
        vocab_kge.append('Room' + str(i))

    for loc in locs:
        for i in range(1, 10):
            vocab_kge.append('Room' + str(i) + '_to_the_' + loc)

    for i in range(1, 10):
        for c in colors:
            vocab_kge.append(c + '_door_to_Room' + str(i))

    vocab_kge.extend(['You'])
    textfile = open(vocab_kge_file, "w")
    for ent in vocab_kge:
        textfile.write(ent + "\n")
    textfile.close()

def load_vocab_kge(vocab_kge_file):
    ent = {}
    id = 0
    with open(vocab_kge_file, 'r') as f:
        for line in f:
            ent[line.strip()] = id
            id += 1
    return {'entity': ent}

def load_vocab(vocab_file):
    vocab = {}
    id = 0
    with open(vocab_file, 'r') as f:
        for line in f:
            vocab[line.strip()] = id
            id += 1
    vocab_rev = {v: i for i, v in vocab.items()}
    return vocab, vocab_rev

class KGEnv:
    '''

    KG environment performs additional graph-based processing.

    '''
    def __init__(self, gym_env, use_pixel, seed, vocab_file, vocab_kge_file, debug_mode):
        random.seed(seed)
        np.random.seed(seed)
        self.gym_env         = gym_env
        self.use_pixel       = use_pixel
        self.seed            = seed
        self.episode_steps   = 0
        gen_vocab_kge(vocab_kge_file)
        self.vocab_kge = load_vocab_kge(vocab_kge_file)
        self.vocab, self.vocab_rev = load_vocab(vocab_file)
        self.env             = None
        self.state_rep       = None
        self.room_idx = {}
        self.ball_idx = {}
        self.box_idx = {}
        self.key_idx = {}
        self.door_idx = {}
        self.create() # create the gym environment, so above self.env is instantiated
        self.debug_mode = debug_mode

    def room_to_idx(self, id):
        if not id in self.room_idx.keys():
            self.room_idx[id] = len(self.room_idx) + 1
        return self.room_idx[id]

    def ball_to_idx(self, id):
        if not id in self.ball_idx.keys():
            self.ball_idx[id] = len(self.ball_idx) + 1
        return self.ball_idx[id]

    def box_to_idx(self, id):
        if not id in self.box_idx.keys():
            self.box_idx[id] = len(self.box_idx) + 1
        return self.box_idx[id]

    def key_to_idx(self, id):
        if not id in self.key_idx.keys():
            self.key_idx[id] = len(self.key_idx) + 1
        return self.key_idx[id]

    def door_to_idx(self, id):
        if not id in self.door_idx.keys():
            self.door_idx[id] = len(self.door_idx) + 1
        return self.door_idx[id]

    def create(self):
        ''' Create the Jericho environment and connect to redis. '''
        env = gym.make(self.gym_env)
        if self.use_pixel:
            env = RGBImgPartialObsWrapper(env)
        env.seed(self.seed)
        self.env = env
        # self.conn_agent_graph = redis.Redis(host='localhost', port=6379, db=0) # not using at the moment

    def set_seed(self, seed):
        self.env.seed(seed)

    def _build_graph_rep(self, obs, action):
        ''' Returns various graph-based representations of the current state. '''
        add_rules, remove_rules, agent_rules = self.state_rep.step(obs, action, self)
        graph_state = self.state_rep.graph_state
        graph_state_rep = self.state_rep.graph_state_rep
        agent_graph_state = self.state_rep.agent_graph_state
        agent_graph_state_rep = self.state_rep.agent_graph_state_rep
        action_rep = self.state_rep.get_action_rep(action, self.env)
        return GraphInfo(action_rep, graph_state, graph_state_rep, agent_graph_state, agent_graph_state_rep)


    def step(self, action):
        self.episode_steps += 1
        obs, reward, done, info = self.env.step(action)
        if done:
            graph_info = GraphInfo(prev_act_rep=self.state_rep.get_action_rep(action, self.env),
                                   graph_state=self.state_rep.graph_state,
                                   graph_state_rep=self.state_rep.graph_state_rep,
                                   agent_graph_state=self.state_rep.agent_graph_state,
                                   agent_graph_state_rep=self.state_rep.agent_graph_state_rep)
        else:
            graph_info = self._build_graph_rep(obs['raw_obs'], action)
        obs.pop('raw_obs')
        return obs, reward, done, info, graph_info


    def reset(self):
        self.state_rep = StateAction(self.vocab_kge, self.vocab, self.vocab_rev, self.debug_mode)
        self.room_idx = {}
        self.ball_idx = {}
        self.box_idx = {}
        self.key_idx = {}
        self.door_idx = {}
        self.episode_steps = 0
        obs = self.env.reset()
        graph_info = self._build_graph_rep(obs['raw_obs'], None)
        obs.pop('raw_obs')
        return obs, graph_info


    def close(self):
        self.env.close()
