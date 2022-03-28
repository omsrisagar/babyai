"""
Script to train the agent through reinforcment learning.
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import logging
import csv
import json
import gym
import time
import datetime
import torch
import numpy as np
import subprocess
import sentencepiece as spm
import json
import itertools

import babyai
import babyai.utils as utils
import babyai.rl
from babyai.arguments import ArgumentParser
from babyai.model import ACModel
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import ModelAgent
from babyai.rl.utils.env import KGEnv
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--algo", default='ppo',
                        help="algorithm to use (default: ppo)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--reward-scale", type=float, default=20.,
                        help="Reward scale multiplier")
    parser.add_argument("--gae-lambda", type=float, default=0.99,
                        help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="number of updates between two saves (default: 50, 0 means no saving)")
    parser.add_argument('--vocab_file', default='../babyai/data/vocab.txt')
    parser.add_argument('--vocab_kge_file', default='../babyai/data/vocab_kge.txt')
    parser.add_argument('--gat_emb_size', default=128, type=int)
    parser.add_argument('--agent_gat_emb_size', default=128, type=int)
    parser.add_argument('--dropout_ratio', default=0.2, type=float)
    parser.add_argument('--no-world-graph', dest='use_world_graph', action='store_false')
    parser.add_argument('--no-agent-graph', dest='use_agent_graph', action='store_false')
    parser.add_argument('--no-obs-image', dest='use_obs_image', action='store_false')
    parser.add_argument('--no-film', dest='no_film', action='store_true', help='applies only to kg representations')
    parser.add_argument('--debug', dest='debug_mode', action='store_true')
    # parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
    #                     help='number of physical computers/nodes/EC2 instances')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus you plan to use in this node')
    parser.add_argument('-sgr', '--sgr', default=0, type=int,
                        help='starting gpu rank: starting global rank of gpus in this node (sum gpus in prev nodes)')
    parser.add_argument('-spr', '--spr', default=0, type=int,
                        help='starting proc rank: starting global rank of procs in this node (sum procs in prev nodes)')
    parser.add_argument('-ws', '--ws', default=1, type=int,
                        help='world_size: total number of GPUs/processes you are running across all nodes')
    parser.add_argument('--gpu_ids', default='0', type=str, help='id(s) from nvidia-smi for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--master_addr', default='localhost', type=str, help='Address/name of Rank 0')
    parser.add_argument('--master_port', default='8888', type=str, help='Port to be used on Rank 0')
    parser.set_defaults(use_world_graph=True)
    parser.set_defaults(use_agent_graph=True)
    parser.set_defaults(use_obs_image=True)
    parser.set_defaults(no_film=False)
    parser.set_defaults(debug_mode=False)
    args = parser.parse_args()

    # args.world_size = args.gpus * args.nodes
    # os.environ['MASTER_ADDR'] = '130.107.72.224' #cuda0003
    # os.environ['MASTER_ADDR'] = '130.107.72.207'  # tulsi
    os.environ['MASTER_ADDR'] = args.master_addr
    # os.environ['MASTER_ADDR'] = '130.107.72.220'  # cuda0001
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    utils.seed(args.seed)

    mp.spawn(train, nprocs=args.gpus, args=(args,))
    # train(0, args)

def train(gpu, args):
    rank = args.sgr + gpu  # global rank of this gpu/process.Dont get confused between gpu process and episode process
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.ws,
        rank=rank
    )
    # Generate environments
    envs = []
    use_pixel = 'pixel' in args.arch
    proc_per_gpu = args.procs // args.gpus # for now assum'g equal proc per gpu; no. of env proc to launch per gpu on
    # this node
    for i in range(proc_per_gpu):
        env = KGEnv(args.env, use_pixel, 100 * args.seed + i + args.spr + gpu * proc_per_gpu, args.vocab_file,
                    args.vocab_kge_file, args.debug_mode, i==0)
        if not env.vocab_kge['entity']:
            print('Empty vocab kge!')
        envs.append(env)

    # Define model name
    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    instr = args.instr_arch if args.instr_arch else "noinstr"
    mem = "mem" if not args.no_mem else "nomem"
    model_name_parts = {
        'env': args.env,
        'algo': args.algo,
        'arch': args.arch,
        'instr': instr,
        'mem': mem,
        'seed': args.seed,
        'info': '',
        'coef': '',
        'suffix': suffix}
    default_model_name = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(**model_name_parts)
    if args.pretrained_model:
        default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
    args.model = args.model.format(**model_name_parts) if args.model else default_model_name

    if rank == args.sgr:  # writing to disk can be done only once in each node (i.e., first gpu in this node)
        utils.configure_logging(args.model)
    logger = logging.getLogger(__name__)

    # Save KGEnv vocab into json file.
    model_dir = utils.get_model_dir(args.model)
    if rank == args.sgr:  # writing to disk can be done only once in each node (i.e., first gpu in this node)
        os.makedirs(model_dir, exist_ok=True)
        with open(model_dir + '/vocab.json', "w") as output:
            json.dump(envs[0].vocab, output)

    # Define obss preprocessor
    if 'emb' in args.arch:
        obss_preprocessor = utils.IntObssPreprocessor(args.model, envs[0].env.observation_space, args.pretrained_model)
    else:
        obss_preprocessor = utils.ObssPreprocessor(args.model, envs[0].env.observation_space, args.pretrained_model)

    if torch.cuda.is_available():
        # device = torch.device('cuda')
        # acmodel.cuda()
        # acmodel.to(device)
        torch.cuda.set_device(gpu)
        device = torch.device(gpu)
    else:
        device=torch.device('cpu')

    # Define actor-critic model
    # acmodel = utils.load_model(args.model, raise_not_found=False)
    # if acmodel is None:
    if args.pretrained_model:
        acmodel = utils.load_model(args.pretrained_model, device, raise_not_found=True)  # need to copy pretrained
        # model to
        # all nodes, in case you plan to use this in DDP
    else:
        acmodel = ACModel(obss_preprocessor.obs_space, envs[0].env.action_space,
                          args.image_dim, args.memory_dim, args.instr_dim, args.gat_emb_size, args.agent_gat_emb_size,
                          args.dropout_ratio, envs[0].vocab, args.vocab_kge_file, args.use_obs_image,
                          args.use_agent_graph, args.use_world_graph, args.no_film, not args.no_instr, args.instr_arch,
                          not args.no_mem, args.arch)

    if rank == args.sgr:  # writing to disk can be done only once in each node (i.e., first gpu in this node)
        obss_preprocessor.vocab.save()
        utils.save_model(acmodel, args.model)

    if torch.cuda.is_available():
        acmodel.cuda(gpu)

    # Define actor-critic algo

    reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
    if args.algo == "ppo":
        algo = babyai.rl.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                                 args.gae_lambda,
                                 args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                 args.optim_eps, args.clip_eps, args.ppo_epochs, args.batch_size, obss_preprocessor,
                                 reshape_reward, gpu, rank)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    # When using extra binary information, more tensors (model params) are initialized compared to when we don't use that.
    # Thus, there starts to be a difference in the random state. If we want to avoid it, in order to make sure that
    # the results of supervised-loss-coef=0. and extra-binary-info=0 match, we need to reseed here.

    utils.seed(args.seed)

    # Restore training status

    status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
    if os.path.exists(status_path):
        with open(status_path, 'r') as src:
            status = json.load(src)
    else:
        status = {'i': 0,
                  'num_episodes': 0,
                  'num_frames': 0}

    # Define logger and Tensorboard writer and CSV writer

    if rank == args.sgr:
        header = (["update", "episodes", "frames", "FPS", "duration"]
                  + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
                  + ["success_rate"]
                  + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
                  + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])
        if args.tb:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(utils.get_log_dir(args.model))

        csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
        first_created = not os.path.exists(csv_path)
        # we don't buffer data going in the csv log, cause we assume
        # that one update will take much longer that one write to the log
        csv_writer = csv.writer(open(csv_path, 'a', 1))  # 1 here indicates 1 line of buffering (before writing to file)
        if first_created:
            csv_writer.writerow(header)

        # Log code state, command, availability of CUDA and model

        babyai_code = list(babyai.__path__)[0]
        try:
            last_commit = subprocess.check_output(
                'cd {}; git log -n1'.format(babyai_code), shell=True).decode('utf-8')
            logger.info('LAST COMMIT INFO:')
            logger.info(last_commit)
        except subprocess.CalledProcessError:
            logger.info('Could not figure out the last commit')
        try:
            diff = subprocess.check_output(
                'cd {}; git diff'.format(babyai_code), shell=True).decode('utf-8')
            if diff:
                logger.info('GIT DIFF:')
                logger.info(diff)
        except subprocess.CalledProcessError:
            logger.info('Could not figure out the last commit')
        logger.info('COMMAND LINE ARGS:')
        logger.info(args)
        logger.info("CUDA available: {}".format(torch.cuda.is_available()))
        logger.info(acmodel)

    # Train model

    total_start_time = time.time()
    best_success_rate = 0
    best_mean_return = 0
    test_env_name = args.env
    while status['num_frames'] < args.frames:
        # Update parameters

        update_start_time = time.time()
        logs = algo.update_parameters()
        all_logs = {}
        for k, v in logs.items():
            # print(f'{k}: GPU {gpu}')
            # print(logs[k])
            if 'per_episode' in k:
                all_logs[k] = all_gather(logs[k], args.ws, device)
            else:
                all_logs[k] = [torch.zeros_like(logs[k]) for _ in range(args.ws)]
                dist.all_gather(all_logs[k], logs[k])
                # print('Done rest')
            # if gpu == 0:
            #     print(all_logs[k])
        update_end_time = time.time()

        # reduce the data from all ranks
        # func_list =  itertools.chain.from_iterable([[torch.cat]*3, [torch.sum]*2, [torch.mean]*6])
        for k, v in all_logs.items():
            if 'per_episode' in k:
                all_logs[k] = torch.cat(all_logs[k]).cpu().numpy()
            elif 'num_frames' in k or 'episodes_done' in k:
                all_logs[k] = int(torch.stack(all_logs[k]).sum().cpu().numpy())
            else:
                all_logs[k] = torch.stack(all_logs[k]).mean().cpu().numpy()

        # print('Updated all_logs')
        logs = all_logs
        status['num_frames'] += logs["num_frames"]
        status['num_episodes'] += logs['episodes_done']
        status['i'] += 1

        # Print logs

        if rank == args.sgr:
            if status['i'] % args.log_interval == 0:
                # print('entered print stage')
                total_ellapsed_time = int(time.time() - total_start_time)
                fps = logs["num_frames"] / (update_end_time - update_start_time)
                duration = datetime.timedelta(seconds=total_ellapsed_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                success_per_episode = utils.synthesize(
                    [1 if r > 0 else 0 for r in logs["return_per_episode"]])
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                data = [status['i'], status['num_episodes'], status['num_frames'],
                        fps, total_ellapsed_time,
                        *return_per_episode.values(),
                        success_per_episode['mean'],
                        *num_frames_per_episode.values(),
                        logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                        logs["loss"], logs["grad_norm"]]

                format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                              "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                              "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")

                logger.info(format_str.format(*data))
                if args.tb:
                    assert len(header) == len(data)
                    for key, value in zip(header, data):
                        writer.add_scalar(key, float(value), status['num_frames'])

                csv_writer.writerow(data)

        # Save obss preprocessor vocabulary and model

        if args.save_interval > 0 and status['i'] % args.save_interval == 0:
            if rank == args.sgr:
                obss_preprocessor.vocab.save()
                with open(status_path, 'w') as dst:
                    json.dump(status, dst)
                    # utils.save_model(acmodel, args.model)
                    utils.save_model(algo.acmodel, args.model)

            # Testing the model before saving
            # agent = ModelAgent(args.model, obss_preprocessor, device, argmax=False)
            agent = ModelAgent(algo.acmodel, obss_preprocessor, device, argmax=False)
            # agent.model = acmodel
            agent.model.eval()
            logs = batch_evaluate(agent, test_env_name, args.val_seed, rank, args.gpus, gpu, args.val_episodes,
                                  pixel=use_pixel,
                                  vocab_file=args.vocab_file, vocab_kge_file=args.vocab_kge_file,
                                  debug_mode=args.debug_mode)
            agent.model.train()
            mean_return = np.mean(logs["return_per_episode"])
            success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
            mean_return = torch.tensor(mean_return, device=device)
            success_rate = torch.tensor(success_rate, device=device)
            mean_mean_return = [torch.zeros_like(mean_return) for _ in range(args.ws)]
            dist.all_gather(mean_mean_return, mean_return)
            mean_return = torch.stack(mean_mean_return).mean().cpu().numpy()
            mean_success_rate = [torch.zeros_like(success_rate) for _ in range(args.ws)]
            dist.all_gather(mean_success_rate, success_rate)
            success_rate = torch.stack(mean_success_rate).mean().cpu().numpy()
            save_model = False
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                save_model = True
            elif (success_rate == best_success_rate) and (mean_return > best_mean_return):
                best_mean_return = mean_return
                save_model = True
            if rank == args.sgr:
                if save_model:
                    utils.save_model(algo.acmodel, args.model + '_best')
                    obss_preprocessor.vocab.save(utils.get_vocab_path(args.model + '_best'))
                    logger.info("Return {: .2f}; best model is saved".format(mean_return))
                else:
                    logger.info("Return {: .2f}; not the best model; not saved".format(mean_return))

def all_gather(q, ws, device):
    """
    Gathers tensor arrays of different lengths across multiple gpus

    Parameters
    ----------
        q : tensor array
        ws : world size
        device : current gpu device

    Returns
    -------
        all_q : list of gathered tensor arrays from all the gpus

    """
    local_size = torch.tensor(q.size(), device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(all_sizes)

    size_diff = (max_size - local_size).item()
    if size_diff:
        padding = torch.zeros(size_diff, device=device, dtype=q.dtype)
        q = torch.cat((q, padding))

    all_qs_padded = [torch.zeros_like(q) for _ in range(ws)]
    dist.all_gather(all_qs_padded, q)
    all_qs = []
    for q, size in zip(all_qs_padded, all_sizes):
        all_qs.append(q[:size])
    return all_qs


if __name__ == '__main__':
    main()
