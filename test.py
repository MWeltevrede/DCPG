import numpy as np
import torch
import imageio
import wandb

from dcpg.envs import make_envs

def evaluate(config, actor_critic, device, test_envs=True, norm_infos=None, rendering=False, render_filename=""):
    # Set actor-critic to evaluation mode
    actor_critic.eval()

    # Create environments
    if test_envs:
        envs = make_envs(
            num_envs=config["num_eval_episodes"],
            env_name=config["env_name"],
            num_levels=0,
            start_level=0,
            distribution_mode=config["distribution_mode"],
            normalize_reward=False,
            device=device,
        )
    else:
        envs = make_envs(
            num_envs=config["num_eval_episodes"],
            env_name=config["env_name"],
            num_levels=config["num_levels"],
            start_level=config["start_level"],
            distribution_mode=config["distribution_mode"],
            normalize_reward=False,
            device=device,
        )

    # Initialize environments
    obs = envs.reset()

    if rendering:
        images = []


    # Sample episodes
    episode_rewards = []
    episode_steps = []
    finished = torch.zeros(obs.size(0))

    step = 0

    # Initialize value and return variables
    if norm_infos is not None:
        var, eps, cliprew = norm_infos.values()
        var = torch.tensor(var)
        gamma = 0.999
        disc = 1

        # Initial value estimates
        with torch.no_grad():
            init_value_ests = actor_critic.forward_critic(obs)["value"]
        init_value_ests = torch.squeeze(init_value_ests)

        # Returns
        ret_un_nodisc = torch.zeros(config["num_eval_episodes"])
        ret_n_nodisc = torch.zeros(config["num_eval_episodes"])
        ret_un_disc = torch.zeros(config["num_eval_episodes"])
        ret_n_disc = torch.zeros(config["num_eval_episodes"])

    while not torch.all(finished):
        # Sample action
        with torch.no_grad():
            action, *_ = actor_critic.act(obs)

        if rendering and step < 200:
            # render the first 200 steps
            img = render_obs((obs*255).to(torch.uint8), config["num_eval_episodes"])
            images.append(img)

        # Interact with environments
        obs, reward, *_, infos = envs.step(action)
        step += 1

        # Update returns
        if norm_infos is not None:
            # Calculate Normalized rewards
            reward_n = torch.clip(reward / torch.sqrt(var + eps), -cliprew, cliprew)

            # Update coefficient for discounting
            disc *= gamma

            # Add rewards to returns
            ret_un_nodisc += torch.squeeze(reward) * (1 - finished)
            ret_un_disc += disc * torch.squeeze(reward) * (1 - finished)
            ret_n_nodisc += torch.squeeze(reward_n) * (1 - finished)
            ret_n_disc += disc * torch.squeeze(reward_n) * (1 - finished)

        # Track episode info
        for i, info in enumerate(infos):
            if "episode" in info and not finished[i]:
                episode_rewards.append(info["episode"]["r"])
                episode_steps.append(step)
                finished[i] = 1

    # Close environment
    envs.close()

    if rendering:
        imageio.mimsave(render_filename, [np.array(img) for i, img in enumerate(images) if i%1 == 0], duration=200)
        wandb.log({"pure_eval": wandb.Video(np.array(images).transpose((0,3,1,2)), fps=8)})

    # Statistics
    eval_statistics = {
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
    }

    value_statistics = {}
    if norm_infos is not None:
        value_statistics = {
            "init_value_ests": init_value_ests.mean().item(),
            "ret_un_nodisc": ret_un_nodisc.mean().item(),
            "ret_un_disc": ret_un_disc.mean().item(),
            "ret_n_nodisc": ret_n_nodisc.mean().item(),
            "ret_n_disc": ret_n_disc.mean().item(),
        }

    return eval_statistics, value_statistics


def resize_image(image_matrix, nh, nw):
    image_size = image_matrix.shape
    oh = image_size[0]
    ow = image_size[1]   

    re_image_matrix = np.array([
        np.array([image_matrix[(oh*h // nh)][(ow*w // nw)] for w in range(nw)]) 
        for h in range(nh)
    ])

    return re_image_matrix

def concat_images(image_set, how):
    # dimension of each matrix in image_set
    shape_vals = [imat.shape for imat in image_set]

    # length of dimension of each matrix in image_set
    shape_lens = [len(ishp) for ishp in shape_vals]

    # if all the images in image_set are read in same mode
    channel_flag = True if len(set(shape_lens)) == 1 else False

    if channel_flag:
        ideal_shape = max(shape_vals)
        images_resized = [
            # function call to resize the image
            resize_image(image_matrix=imat, nh=ideal_shape[0], nw=ideal_shape[1]) 
            if imat.shape != ideal_shape else imat for imat in image_set
        ]
    else:
        return False

    images_resized = tuple(images_resized)

    if (how == 'vertical') or (how == 0):
        axis_val = 0
    elif (how == 'horizontal') or (how == 1):
        axis_val = 1
    else:
        axis_val = 1

    # numpy code to concatenate the image matrices
    # concatenation is done based on axis value
    concats = np.concatenate(images_resized, axis=axis_val)
    return concats

def render_obs(obs, n_envs):
    obs = obs.cpu().numpy().transpose((0,2,3,1))
    image_size = int(np.floor(np.sqrt(n_envs)))
    rows = []
    for i in range(image_size):
     row_start_ind = i*image_size
     row = obs[row_start_ind:row_start_ind + image_size]

     rows.append(concat_images(row, how='horizontal'))

    img = concat_images(rows, how='vertical')

    return img
