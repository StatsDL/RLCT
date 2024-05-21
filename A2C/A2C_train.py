from A2C.A2C_helper import Actor, Critic
import torch
import torch.nn as nn
import torch.optim as optim


# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  
class A2Cagent:
    def __init__(self, state_dim, action_dim, params):
      # Params
      self.num_states = state_dim
      self.num_actions = action_dim
      self.gamma = params['gamma']
    
      # Networks
      self.actor = Actor(self.num_states, self.num_actions, int(params['Ahidden_dim']), int(params['Anum_layers']), params['Aact_fn'], params['Adr']).to(device)
      self.critic = Critic(self.num_states, int(params['Chidden_dim']), int(params['Cnum_layers']), params['Cact_fn'], params['Cdr']).to(device)
    
      # Training
      self.critic_criterion  = nn.MSELoss()
      self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=params['alr'])
      self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params['clr'])
    
    
    def agent_mode(self, mode):
        if mode == 'train':
            self.actor.train()
            self.critic.train()
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
      # action = x
      # if x.dim() == 1:
      #     action = torch.softmax(x, dim=0)
      # else:
      #     action = torch.softmax(x, dim=1)
      return action, logs_probs
    

    def update(self, state, action, reward, logs_probs, next_state):
      state_tensor = torch.FloatTensor(state).to(device)
      #action = action.to(device)
      value = self.critic(state_tensor)
      next_state_tensor = torch.FloatTensor(next_state).to(device)
      next_value = self.critic(next_state_tensor)
      
      td_target = torch.from_numpy(reward).to(device) + self.gamma * next_value
      advantage = td_target - value
      
      # critic loss
      critic_loss = nn.functional.mse_loss(td_target, value)
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # actor loss
      actor_loss = (-logs_probs*(torch.from_numpy(reward).to(device) + self.gamma * self.critic(next_state_tensor) - self.critic(state_tensor))).mean()
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()
    
    
    def regression_prediction(self, states_test):
        Actions = []
        for i in range(states_test.shape[0]):
            # state = torch.FloatTensor(states_test[i]).reshape(1,-1).to(device)
            state = states_test[i]
            test_action, logs_probs = self.get_action(state, 'flag')
            Actions.append(test_action)
            
        return Actions
          
      

