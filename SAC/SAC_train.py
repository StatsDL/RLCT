from SAC_helper import *
import copy


# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SACagent:  
    def __init__(self, state_dim, action_dim, params, max_memory_size=20000, min_action=0, max_action=1):
        # Params
        self.num_states = state_dim
        self.num_actions = action_dim
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.alpha = params['alpha']
        self.min_action = min_action
        self.max_action = max_action
        self.batch_size = int(params['batch_size'])

        # Networks
        self.actor = Actor(self.num_states, self.num_actions, int(params['Ahidden_dim']), int(params['Anum_layers']), params['Aact_fn'], params['Adr']).to(device)
        self.critic1 = Critic(self.num_states, self.num_actions, int(params['C1hidden_dim']), int(params['C1num_layers']), params['Cact_fn'], params['Cdr']).to(device)
        self.critic2 = Critic(self.num_states, self.num_actions, int(params['C2hidden_dim']), int(params['C2num_layers']), params['Cact_fn'], params['Cdr']).to(device)
        self.value = Value_Network(self.num_states, int(params['Vhidden_dim']), int(params['Vnum_layers']), params['Vact_fn'], params['Vdr']).to(device)
        self.value_target = copy.deepcopy(self.value).to(device)
        # concat critic parameters to use one optim
        critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())

        # Training
        self.memory = Memory(max_memory_size)
        self.critic_criterion  = nn.MSELoss()
        self.value_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=params['alr'])
        self.critic_optimizer = optim.Adam(self.critic1.parameters(), lr=params['clr'])
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=params['vlr'])
        
    def agent_mode(self, mode):
        if mode == 'train':
            self.actor.train()
            self.critic1.train()
            self.critic2.train()
            self.value.train()
            self.value_target.train()
        elif mode == 'eval':
            self.actor.eval()
        else:
            print('undefined agent mode')
    
    def get_action(self, state, mode = None):
      state_tensor = torch.FloatTensor(state).to(device)
      norm_dists = self.actor(state_tensor)
      if mode is not None:
          x = norm_dists.mean
      else:
          x = norm_dists.sample()
      logs_probs = norm_dists.log_prob(x)
      action = torch.sigmoid(x)

      return action, logs_probs
    
    
    def update(self):
        states, actions, log_probs, rewards, next_states, _ = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        # actions = actions.reshape((actions.shape[0],1))
        actions = torch.FloatTensor(actions).to(device)
        log_probs = torch.FloatTensor(log_probs).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

        # states = torch.squeeze(states)
        # actions = torch.squeeze(actions)
        # log_probs = torch.squeeze(log_probs)
        # next_states = torch.squeeze(next_states)
        # rewards = torch.squeeze(rewards)
        # rewards = torch.squeeze(rewards, dim=2)

        # value loss
        current_Q1 = self.critic1(states, actions)
        target_v1 = current_Q1 - (self.alpha * log_probs)
        # target_v1 = current_Q1 - (self.alpha * log_probs).unsqueeze(dim=1)
        current_Q2 = self.critic2(states, actions)
        target_v2 = current_Q2 - (self.alpha * log_probs)
        # target_v2 = current_Q2 - (self.alpha * log_probs).unsqueeze(dim=1)
        value_target = torch.min(target_v1, target_v2)

        # Update value network
        value_loss = self.value_criterion(value_target, self.value.forward(states))
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # target_Q = rewards.unsqueeze(dim=1) + self.gamma * self.value_target(next_states)
        target_Q = rewards + self.gamma * self.value_target(next_states)

        # critic loss
        current_Q1 = self.critic1.forward(states, actions)
        current_Q2 = self.critic2.forward(states, actions)
        critic1_loss = self.critic_criterion(current_Q1, target_Q)
        critic2_loss = self.critic_criterion(current_Q2, target_Q)
        critic_loss = critic1_loss + critic2_loss

        # Update Critic Networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -(self.critic1.forward(states, actions) - self.alpha * log_probs).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        for target_param, param in zip(self.value_target.parameters(), self.value.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
            
    
    def buffer_fill(self, states_train, train_y, action_low, action_high):
        index = min(len(states_train)-500, np.random.choice(range(len(states_train))))
        for step in range(index, states_train.shape[0]-1):       
            state = states_train[step]
            action, logs_probs = self.get_action(state)
            action = action.cpu()
            action = action.detach().numpy()
            logs_probs = logs_probs.cpu()
            logs_probs = logs_probs.detach().numpy()
            
            reward = -abs(action-train_y[step])                
            next_state = states_train[step+1]
            self.memory.push(state, action, logs_probs, reward, next_state, False)  
            
          
            
    def regression_prediction(self, states_test):
        Actions = []
        for i in range(states_test.shape[0]):
            # state = torch.FloatTensor(states_test[i]).reshape(1,-1).to(device)
            state = states_test[i]
            test_action, logs_probs = self.get_action(state, 'flag')
            Actions.append(test_action)
            
        return Actions
      
      

