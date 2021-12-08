import collections
import numpy as np
from representations import StateAction
import redis
import random
import gym
from gym_minigrid.wrappers import RGBImgPartialObsWrapper


GraphInfo = collections.namedtuple('GraphInfo', 'act_rep, graph_state, graph_state_rep, '
                                                'agent_graph_state, agent_graph_state_rep')

def load_vocab_kge(vocab_file):
    ent = {}
    id = 0
    with open(vocab_file, 'r') as f:
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
    def __init__(self, gym_env, use_pixel, seed, spm_model, vocab_file):
        random.seed(seed)
        np.random.seed(seed)
        self.gym_env         = gym_env
        self.use_pixel       = use_pixel
        self.seed            = seed
        self.episode_steps   = 0
        self.spm_model       = spm_model
        self.vocab_file        = vocab_file
        self.vocab_kge = load_vocab_kge(vocab_file)
        self.vocab, self.vocab_rev = load_vocab(vocab_file)
        self.env             = None
        self.state_rep       = None
        self.room_idx = {}
        self.ball_idx = {}
        self.box_idx = {}
        self.key_idx = {}
        self.door_idx = {}

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
        self.conn_agent_graph = redis.Redis(host='localhost', port=6379, db=0) # not using at the moment


    def _build_graph_rep(self, obs, action):
        ''' Returns various graph-based representations of the current state. '''
        rules, agent_rules = self.state_rep.step(obs, action, self)
        graph_state = self.state_rep.world_graph_state
        graph_state_rep = self.state_rep.world_graph_state_rep
        agent_graph_state = self.state_rep.agent_graph_state
        agent_graph_state_rep = self.state_rep.agent_graph_state_rep
        action_rep = self.state_rep.get_action_rep_drqa(action) if action != None else None
        return GraphInfo(action_rep, graph_state, graph_state_rep, agent_graph_state, agent_graph_state_rep)


    def step(self, action):
        self.episode_steps += 1
        obs, reward, done, info = self.env.step(action)
        if done:
            graph_info = GraphInfo(act_rep=self.state_rep.get_action_rep_drqa(action),
                                   graph_state=self.state_rep.graph_state,
                                   graph_state_rep=self.state_rep.graph_state_rep,
                                   agent_graph_state=self.state_rep.agent_graph_state,
                                   agent_graph_state_rep=self.state_rep.agent_graph_state_rep)
        else:
            graph_info = self._build_graph_rep(obs['raw_obs'], action)
        return obs, reward, done, info, graph_info


    def reset(self):
        self.state_rep = StateAction(self.spm_model, self.vocab_kge, self.vocab, self.vocab_rev)
        self.episode_steps = 0
        obs = self.env.reset()
        graph_info = self._build_graph_rep(obs['raw_obs'], None)
        return obs, graph_info


    def close(self):
        self.env.close()
