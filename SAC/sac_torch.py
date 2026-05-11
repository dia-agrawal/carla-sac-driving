import os
import torch as T
import torch.nn.functional as F
import numpy as np

from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(
        self,
        input_dims,
        env,
        n_actions,
        alpha=9e-4,
        beta=3e-4,
        gamma=0.99,
        tau=0.005,
        max_size=int(1e6),
        batch_size=128,
        reward_scale=1.0,
        chkpt_dir="tmp/sac",
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.scale = reward_scale
        self.chkpt_dir = chkpt_dir
        self.max_grad_norm = 1.0

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        max_action = env.action_space.high

        self.actor = ActorNetwork(
            alpha=alpha,
            input_dims=input_dims,
            max_action=max_action,
            n_actions=n_actions,
            name="actor",
            chkpt_dir=chkpt_dir,
        )

        self.critic_1 = CriticNetwork(
            beta, input_dims, n_actions, name="critic_1", chkpt_dir=chkpt_dir
        )
        self.critic_2 = CriticNetwork(
            beta, input_dims, n_actions, name="critic_2", chkpt_dir=chkpt_dir
        )

        self.target_critic_1 = CriticNetwork(
            beta, input_dims, n_actions, name="target_critic_1", chkpt_dir=chkpt_dir
        )
        self.target_critic_2 = CriticNetwork(
            beta, input_dims, n_actions, name="target_critic_2", chkpt_dir=chkpt_dir
        )
        self._hard_update_targets()

        self.target_entropy = -float(n_actions)
        self.log_alpha = T.tensor(
            -2.0, dtype=T.float32, requires_grad=True, device=self.actor.device
        )
        self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=3e-4)

    def _hard_update_targets(self):
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        with T.no_grad():
            for target_param, param in zip(
                self.target_critic_1.parameters(), self.critic_1.parameters()
            ):
                target_param.data.mul_(1.0 - tau).add_(tau * param.data)

            for target_param, param in zip(
                self.target_critic_2.parameters(), self.critic_2.parameters()
            ):
                target_param.data.mul_(1.0 - tau).add_(tau * param.data)

    def set_chkpt_dir(self, new_dir: str):
        os.makedirs(new_dir, exist_ok=True)

        nets = [
            self.actor,
            self.critic_1,
            self.critic_2,
            self.target_critic_1,
            self.target_critic_2,
        ]
        for net in nets:
            basename = os.path.basename(net.checkpoint_file)
            net.checkpoint_file = os.path.join(new_dir, basename)

        self.chkpt_dir = new_dir

    def save_models(self):
        print("....saving models...")
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

        try:
            T.save(
                {"log_alpha": self.log_alpha.detach().cpu()},
                os.path.join(self.chkpt_dir, "alpha_sac.pt"),
            )
        except Exception as e:
            print(f"Warning: failed to save alpha: {e}")

        try:
            buf_path = os.path.join(self.chkpt_dir, "replay_buffer.npz")
            self.memory.save(buf_path)
        except Exception as e:
            print(f"Warning: failed to save replay buffer: {e}")

    def load_models(self, load_alpha = True, load_buffer=True):
        print("....loading models...")
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

        try:
            self.target_critic_1.load_checkpoint()
            self.target_critic_2.load_checkpoint()
        except Exception:
            self._hard_update_targets()

        if load_alpha:
            alpha_path = os.path.join(self.chkpt_dir, "alpha_sac.pt")
            if os.path.exists(alpha_path):
                ck = T.load(alpha_path, map_location=self.actor.device)
                self.log_alpha.data.copy_(ck["log_alpha"].to(self.actor.device))
                print(f"Loaded alpha from {alpha_path}")
        
        if load_buffer:
            buf_path = os.path.join(self.chkpt_dir, "replay_buffer.npz")
            if os.path.exists(buf_path):
                self.memory.load(buf_path)
                print(f"Loaded replay buffer from {buf_path}")

    def load_models_from(
        self,
        load_dir: str,
        load_actor: bool = True,
        load_critics: bool = True,
        load_alpha: bool = True,
        load_buffer: bool = True,
    ):
        print(f"....loading models from {load_dir}...")

        if load_actor:
            actor_path = os.path.join(load_dir, os.path.basename(self.actor.checkpoint_file))
            if os.path.exists(actor_path):
                self.actor.load_state_dict(T.load(actor_path, map_location=self.actor.device))
                print(f"Loaded {os.path.basename(actor_path)} from {load_dir}")

        if load_critics:
            critic_nets = [
                self.critic_1,
                self.critic_2,
                self.target_critic_1,
                self.target_critic_2,
            ]
            for net in critic_nets:
                basename = os.path.basename(net.checkpoint_file)
                path = os.path.join(load_dir, basename)
                if os.path.exists(path):
                    net.load_state_dict(T.load(path, map_location=net.device))
                    print(f"Loaded {basename} from {load_dir}")
        else:
            print("Critics not loaded; using fresh critics and targets.")
            self._hard_update_targets()

        if load_alpha:
            alpha_path = os.path.join(load_dir, "alpha_sac.pt")
            if os.path.exists(alpha_path):
                ck = T.load(alpha_path, map_location=self.actor.device)
                self.log_alpha.data.copy_(ck["log_alpha"].to(self.actor.device))
                print(f"Loaded alpha_sac.pt from {load_dir}")
        else:
            self.log_alpha.data.fill_(-2.0)
            print("Alpha reset to default.")

        if load_buffer:
            buf_path = os.path.join(load_dir, "replay_buffer.npz")
            if os.path.exists(buf_path):
                self.memory.load(buf_path)
                print(f"Loaded replay buffer from {buf_path}")
        else:
            print("Replay buffer not loaded; starting with fresh buffer.")

    def choose_action(self, observation, evaluate=False):
        obs_np = np.asarray(observation, dtype=np.float32)
        state = T.from_numpy(obs_np).unsqueeze(0).to(self.actor.device)

        with T.no_grad():
            if evaluate:
                action = self.actor.deterministic(state)
            else:
                action, _ = self.actor.sample(state, reparameterize=False)

        return action.cpu().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return None

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(states, dtype=T.float32).to(self.actor.device)
        action = T.tensor(actions, dtype=T.float32).to(self.actor.device)
        reward = T.tensor(rewards, dtype=T.float32).to(self.actor.device)
        done = T.tensor(dones, dtype=T.float32).to(self.actor.device)
        state_ = T.tensor(states_, dtype=T.float32).to(self.actor.device)

        self.log_alpha.data.clamp_(-6.0, 6.0)
        alpha = self.log_alpha.exp()

        with T.no_grad():
            next_action, next_logp = self.actor.sample(state_, reparameterize=False)
            q1_next = self.target_critic_1(state_, next_action).view(-1)
            q2_next = self.target_critic_2(state_, next_action).view(-1)
            min_q_next = T.min(q1_next, q2_next)

            y = self.scale * reward + self.gamma * (1.0 - done) * (
                min_q_next - alpha * next_logp.view(-1)
            )

        q1 = self.critic_1(state, action).view(-1)
        q2 = self.critic_2(state, action).view(-1)

        critic_1_loss = F.mse_loss(q1, y)
        critic_2_loss = F.mse_loss(q2, y)
        critic_loss = critic_1_loss + critic_2_loss

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.max_grad_norm)
        T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.max_grad_norm)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        new_action, logp = self.actor.sample(state, reparameterize=True)
        q1_pi = self.critic_1(state, new_action).view(-1)
        q2_pi = self.critic_2(state, new_action).view(-1)
        min_q_pi = T.min(q1_pi, q2_pi)

        actor_loss = (alpha.detach() * logp.view(-1) - min_q_pi).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor.optimizer.step()

        alpha_loss = -(self.log_alpha * (logp.view(-1) + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.log_alpha.data.clamp_(-6.0, 6.0)

        try:
            if float(critic_loss.item()) > 1e4:
                print(
                    f"[SAC WARNING] actor_loss={actor_loss.item():.3e}, "
                    f"critic_loss={critic_loss.item():.3e}, "
                    f"alpha={alpha.item():.3e}, "
                    f"logp_mean={float(logp.mean().item()):.3e}, "
                    f"target_mean={float(y.mean().item()):.3e}, "
                    f"target_max={float(y.abs().max().item()):.3e}"
                )
        except Exception:
            pass

        self.update_network_parameters()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
            "alpha_loss": float(alpha_loss.item()),
        }