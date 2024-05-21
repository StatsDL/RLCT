from PPO_helper import *


# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOagent:
    def __init__(self, state_dim, action_dim, params, eps_clip=0.2):
        # Params
        self.num_states = state_dim
        self.num_actions = action_dim
        self.gamma = params['gamma']
        self.PPO_epochs = int(params['PPO_epochs'])
        self.value_coeff = params['val_coeff']
        self.ent_coeff = params['ent_coeff']
        self.eps_clip = eps_clip
    
    
        # Networks
        self.policy = Actor(self.num_states, self.num_actions, int(params['Ahidden_dim']), int(params['Anum_layers']), params['Aact_fn'], params['Adr']).to(device)
        self.critic = Critic(self.num_states, int(params['Chidden_dim']), int(params['Cnum_layers']), params['Cact_fn'], params['Cdr']).to(device)
    
        # Training
        self.optimizer = torch.optim.Adam([
                            {'params': self.policy.parameters(), 'lr': params['alr']},
                            {'params': self.critic.parameters(), 'lr': params['clr']}  ])
    
        self.MseLoss = nn.MSELoss().to(device)
    
    def agent_mode(self, mode):
        if mode == 'train':
            self.policy.train()
            self.critic.train()
        elif mode == 'eval':
            self.policy.eval()
        else:
            print('undefined agent mode')
              
    def get_action(self, state, mode = None):
        # state_tensor = torch.FloatTensor(state).to(device)
        norm_dists = self.policy(state)
        if mode is not None:
            x = norm_dists.mean
        else:
            x = norm_dists.sample()
        logs_probs = norm_dists.log_prob(x)
        entropy = norm_dists.entropy()
        action = torch.sigmoid(x)
        return action, logs_probs, entropy
        
    
    def calculate_returns(self, rewards, discount_factor):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + R * discount_factor
            returns.insert(0, R)
    
        # return torch.tensor(np.array(returns))
        return torch.tensor(returns)
    
    
    def update_policy(self, states, actions, log_prob_actions, advantages, returns):
        old_states = states.detach().to(device)
        old_actions = actions.detach().to(device)
        old_log_probs = log_prob_actions.detach().to(device)
        advantages = advantages.detach().to(device)
        returns = returns.detach().to(device)
    
        for _ in range(self.PPO_epochs):
          #get new log prob of old actions for all input states
          action, log_probs_new, entropy = self.get_action(old_states)
          value_pred = self.critic(old_states)
    
          policy_ratio = (log_probs_new - old_log_probs).exp()
          policy_loss_1 = policy_ratio * advantages
          policy_loss_2 = torch.clamp(policy_ratio, min=1.0 - self.eps_clip, max=1.0 + self.eps_clip) * advantages
    
          policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
    
          value_loss = self.MseLoss(self.critic(old_states), returns)
    
          loss = policy_loss + self.value_coeff * value_loss - self.ent_coeff * entropy.mean()
    
          # take gradient step
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          
          
    
    def regression_prediction(self, states_test):
        Actions = []
        for i in range(states_test.shape[0]):
            state = torch.FloatTensor(states_test[i]).reshape(1,-1).to(device)
            test_action, logs_probs, entropy = self.get_action(state, 'flag')
            Actions.append(test_action)
            
        return Actions
      
      

