from copy import deepcopy
import torch.optim as optim

'''
Each Sample is a policy which contains the actor_critic, agent status and running mean std info.
The algorithm can pick any sample to resume its training process or train with another optimization direction
through those information.
Each Sample is indexed by a unique optgraph_id
'''
class Sample:
    def __init__(self, env_params, actor_critic, agent, objs = None, optgraph_id = None):
        self.env_params = env_params
        self.actor_critic = actor_critic
        self.agent = agent
        self.link_policy_agent()
        self.objs = objs
        self.optgraph_id = optgraph_id

    @classmethod
    def copy_from(cls, sample):
        env_params = deepcopy(sample.env_params)
        actor_critic = deepcopy(sample.actor_critic)
        agent = deepcopy(sample.agent)
        objs = deepcopy(sample.objs)
        optgraph_id = sample.optgraph_id
        return cls(env_params, actor_critic, agent, objs, optgraph_id)

    def link_policy_agent(self):
        self.agent.actor_critic = self.actor_critic
        optim_state_dict = deepcopy(self.agent.optimizer.state_dict())
        self.agent.optimizer = optim.Adam(self.actor_critic.parameters(), lr = 3e-4, eps = 1e-5)
        self.agent.optimizer.load_state_dict(optim_state_dict)