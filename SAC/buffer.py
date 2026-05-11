import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((max_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((max_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((max_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def save(self, filepath: str):
        """Save the replay buffer to a compressed .npz file at filepath."""
        # only save up to max_mem (valid entries)
        max_mem = min(self.mem_cntr, self.mem_size)
        np.savez_compressed(
            filepath,
            state_memory=self.state_memory[:max_mem],
            new_state_memory=self.new_state_memory[:max_mem],
            action_memory=self.action_memory[:max_mem],
            reward_memory=self.reward_memory[:max_mem],
            terminal_memory=self.terminal_memory[:max_mem],
            mem_cntr=np.int64(self.mem_cntr),
        )

    def load(self, filepath: str):
        """Load the replay buffer from a .npz file. Existing buffer contents are replaced."""
        data = np.load(filepath, allow_pickle=False)
        state_memory = data['state_memory']
        new_state_memory = data['new_state_memory']
        action_memory = data['action_memory']
        reward_memory = data['reward_memory']
        terminal_memory = data['terminal_memory']
        mem_cntr = int(data['mem_cntr'].tolist() if hasattr(data['mem_cntr'], 'tolist') else data['mem_cntr'])

        # determine how many entries to restore (can't exceed buffer capacity)
        n = min(state_memory.shape[0], self.mem_size)
        self.state_memory[:n] = state_memory[:n]
        self.new_state_memory[:n] = new_state_memory[:n]
        self.action_memory[:n] = action_memory[:n]
        self.reward_memory[:n] = reward_memory[:n]
        self.terminal_memory[:n] = terminal_memory[:n]
        # set mem_cntr to the loaded count (capped to buffer size)
        self.mem_cntr = min(mem_cntr, self.mem_size)
