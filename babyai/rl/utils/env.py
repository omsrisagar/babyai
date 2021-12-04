import collections
import numpy as np
from representations import StateAction
import random
import gym
from gym_minigrid.wrappers import RGBImgPartialObsWrapper


GraphInfo = collections.namedtuple('GraphInfo', 'act_rep, graph_state, graph_state_rep')

def load_vocab(env):
    vocab = {i+2: str(v) for i, v in enumerate(env.get_dictionary())}
    vocab[0] = ' '
    vocab[1] = '<s>'
    vocab_rev = {v: i for i, v in vocab.items()}
    return vocab, vocab_rev

def clean_obs(s):
    garbage_chars = ['*', '-', '!', '[', ']']
    for c in garbage_chars:
        s = s.replace(c, ' ')
    return s.strip()


class KGEnv:
    '''

    KG environment performs additional graph-based processing.

    '''
    def __init__(self, gym_env, use_pixel, seed, spm_model, step_limit=None, stuck_steps=10, gat=True):
        random.seed(seed)
        np.random.seed(seed)
        self.gym_env         = gym_env
        self.use_pixel       = use_pixel
        self.seed            = seed
        self.episode_steps   = 0
        self.stuck_steps     = 0
        self.valid_steps     = 0
        self.spm_model       = spm_model
        self.gat             = gat
        self.env             = None
        self.vocab           = None
        self.vocab_rev       = None
        self.state_rep       = None


    def create(self):
        ''' Create the Jericho environment and connect to redis. '''
        env = gym.make(self.gym_env)
        if self.use_pixel:
            env = RGBImgPartialObsWrapper(env)
        env.seed(self.seed)
        self.vocab, self.vocab_rev = load_vocab(self.env)


    def _build_graph_rep(self, action):
        ''' Returns various graph-based representations of the current state. '''
        graph_state = self.state_rep.graph_state
        graph_state_rep = self.state_rep.graph_state_rep
        action_rep = self.state_rep.get_action_rep_drqa(action)
        return GraphInfo(action_rep, graph_state, graph_state_rep)


    def step(self, action):
        self.episode_steps += 1
        obs, reward, done, info = self.env.step(action)
        info['steps'] = self.episode_steps
        if done:
            graph_info = GraphInfo(act_rep=self.state_rep.get_action_rep_drqa(action),
                                   graph_state=self.state_rep.graph_state,
                                   graph_state_rep=self.state_rep.graph_state_rep)
        else:
            graph_info = self._build_graph_rep(action)
        return obs, reward, done, info, graph_info


    def reset(self):
        self.state_rep = StateAction(self.spm_model, self.vocab, self.vocab_rev,
                                     self.tsv_file, self.max_word_len)
        self.stuck_steps = 0
        self.valid_steps = 0
        self.episode_steps = 0
        obs, info = self.env.reset()
        info['valid'] = False
        info['steps'] = 0
        graph_info = self._build_graph_rep('look', obs)
        return obs, info, graph_info


    def close(self):
        self.env.close()
