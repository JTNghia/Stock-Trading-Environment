#================================================================
#
#   File name   : RL-Bitcoin-trading-bot_4.py
#   Author      : PyLessons
#   Created date: 2021-01-13
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/RL-Bitcoin-trading-bot
#   Description : Trading Crypto with Reinforcement Learning #4
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import copy
import pandas as pd
import numpy as np
import random
from collections import deque
from keras.models import Model, load_model
from tensorboardX import SummaryWriter
from tensorflow.keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'..')

#in colab
from env.model import Actor_Model, Critic_Model, Shared_Model, DQNModel
from env.utils import TradingGraph, Write_to_file

# from model import Actor_Model, Critic_Model, Shared_Model
# from utils import TradingGraph, Write_to_file
import matplotlib.pyplot as plt
from datetime import datetime
import cv2 

class CustomAgent:
    # A custom Bitcoin trading agent
    def __init__(self, lookback_window_size=50, lr=0.00005, epochs=1, optimizer=Adam, batch_size=32, model=""):
        self.lookback_window_size = lookback_window_size
        self.model = model
        
        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        # folder to save models
        self.log_name = datetime.now().strftime("%Y_%m_%d_%H_%M")+"Stock_trader"
        
        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (lookback_window_size, 18)

        # Neural Networks part bellow
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size

        # Create shared Actor-Critic network model
        self.Actor = self.Critic = Shared_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer, model=self.model)
        # Create Actor-Critic network model
        #self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
        #self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
        
    # create tensorboard writer
    def create_writer(self, initial_balance, normalize_value, train_episodes):
        self.replay_count = 0
        self.writer = SummaryWriter('runs/'+self.log_name)

        # Create folder to save models
        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.start_training_log(initial_balance, normalize_value, train_episodes)
            
    def start_training_log(self, initial_balance, normalize_value, train_episodes):      
        # save training parameters to Parameters.txt file for future
        with open(self.log_name+"/Parameters.txt", "w") as params:
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            params.write(f"training start: {current_date}\n")
            params.write(f"initial_balance: {initial_balance}\n")
            params.write(f"training episodes: {train_episodes}\n")
            params.write(f"lookback_window_size: {self.lookback_window_size}\n")
            params.write(f"lr: {self.lr}\n")
            params.write(f"epochs: {self.epochs}\n")
            params.write(f"batch size: {self.batch_size}\n")
            params.write(f"normalize_value: {normalize_value}\n")
            params.write(f"model: {self.model}\n")
            
    def end_training_log(self):
        with open(self.log_name+"/Parameters.txt", "a+") as params:
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            params.write(f"training end: {current_date}\n")

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions 
        values = self.Critic.critic_predict(states)
        next_values = self.Critic.critic_predict(next_states)
        
        # Compute advantages
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        '''
        plt.plot(target,'-')
        plt.plot(advantages,'.')
        ax=plt.gca()
        ax.grid(True)
        plt.show()
        '''
        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])
        
        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)
        c_loss = self.Critic.Critic.fit(states, target, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1

        return np.sum(a_loss.history['loss']), np.sum(c_loss.history['loss'])

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.actor_predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction
        
    def save(self, name="Crypto_trader", score="", args=[]):
        # save keras model weights
        self.Actor.Actor.save_weights(f"{self.log_name}/{score}_{name}_Actor.h5")
        self.Critic.Critic.save_weights(f"{self.log_name}/{score}_{name}_Critic.h5")

        # log saved model arguments to file
        if len(args) > 0:
            with open(f"{self.log_name}/log.txt", "a+") as log:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log.write(f"{current_time}, {args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}\n")

    def load(self, folder, name):
        # load keras model weights
        self.Actor.Actor.load_weights(os.path.join(folder, f"{name}_Actor.h5"))
        self.Critic.Critic.load_weights(os.path.join(folder, f"{name}_Critic.h5"))


class DQNAgent:
    def __init__(self, env, lookback_window_size=50, lr=0.00005, epochs=1, batch_size=200, EPISODES = 10000, gamma = 0.95):
        # df = pd.read_csv('excel_hvn.csv')
        # # df['Date'] = pd.to_datetime(df['Date'])
        # # df = df.sort_values('Date')
        # lookback_window_size = 10
        # num = 10
        # train_df = df[:-num-lookback_window_size]
        # train_df.info()
        # test_df = df[-num-lookback_window_size:] # 30 days


        self.env = env #CustomEnv(df, lookback_window_size=lookback_window_size)
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = 18*lookback_window_size
        self.action_size = len(self.env.action_space)
        self.EPISODES = EPISODES
        self.memory = deque(maxlen=2000)
        
        self.gamma = gamma    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = batch_size
        self.train_start = len(df)

        # create main model
        self.model = DQNModel(input_shape=(self.state_size,), action_space = self.action_size)
        print(f'env: {self.env}')
        print(f'state_size: {self.state_size}')
        print(f'action_size: {self.action_size}')
        print(f'memory: {self.memory}')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)


    def load(self, name):
        self.model = load_model(name)

    def save(self, path = '../weightsDQN', filename = ''):
        self.model.save(f'{path}/{filename}')
        w = open(f'{path}/DQN_AverageProfits_Log.txt','a+')
        w.write(f'{filename}: {np.mean(self.env._profits)}\n')


            
    def train(self):
        prev_avg_profits = -99999999
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            print('episode: ',e)
            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                if done:     
                    avg_profit = np.mean(self.env._profits)
                    print("episode: {}/{}, avg_profit: {}, e: {:.2}".format(e, self.EPISODES, avg_profit, self.epsilon))
                    if prev_avg_profits < avg_profit and e > 0:
                        prev_avg_profits = avg_profit
                        print(f"Saving trained model as DQN_Episode({e}).h5")
                        self.save(filename=f'DQN_Episode({e}).h5')
                    # self.env.render_all(f"SaveModel/DQN_Episode({e})")
                    self.env._profits.clear()
                    self.env._networths.clear()
                    break
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1

                self.replay()

    def retrain(self):
        num = 368
        self.load(f"SaveModel/DQN_Episode({num}).h5")
        for e in range(num+1, self.EPISODES):
            state = self.env.reset()
            # print(f'state {state}, shape {state.shape}\n\n\n\n\n\n\n\n\n\n\n')
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            print('episode: ',e)
            while not done:
                # self.env.render()

                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                if done:                   
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    print("Saving trained model as DQN_HVN_Episode({e}).h5")
                    self.save(f"SaveModel/DQN_Episode({e}).h5")
                    self.env.render_all(f"SaveModel/DQN_Episode({e})")
                    self.env._profits.clear()
                    self.env._networths.clear()
                    break
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1


    def test(self, folder = '../weightsDQN', filename='DQN_Episode(2).h5', episode = 1):
        self.load(f"{folder}/{filename}")
        for e in range(episode):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done= self.env.step(action)
                if done:
                    print("episode: {}/{}, avg_profit: {}".format(e, self.EPISODES, np.mean(self.env._profits)))
                    # self.env.render_all(f"TestImage/DQN_HVN_Episode(16)_Episode({len(self.env.df)-self.env.lookback_window_size}).png")
                    break
                state = np.reshape(next_state, [1, self.state_size])
                i += 1

class CustomEnv:
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=50, Render_range=100, Show_reward=False, normalize_value=40000):
        # Define action space and state size and other custom parameters
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df)-1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.Render_range = Render_range # render range in visualization
        self.Show_reward = Show_reward # show order reward in rendered visualization
        self.action_space = np.array([0, 1, 2])

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)
        
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.normalize_value = normalize_value
                # For visualization
        self._profits = []
        self._networths = []
        self._dates = []

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size = 0):
        self.visualization = TradingGraph(Render_range=self.Render_range, Show_reward=self.Show_reward) # init visualization
        self.trades = deque(maxlen=self.Render_range) # limited orders memory for visualization
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0 # track episode orders count
        self.prev_episode_orders = 0 # track previous episode orders count
        self.rewards = deque(maxlen=self.Render_range)
        self.env_steps_size = env_steps_size
        self.punish_value = 0
        if env_steps_size > 0: # used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else: # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
            
        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            self.market_history.append([self.df.loc[current_step, 'Open'],
                                        self.df.loc[current_step, 'High'],
                                        self.df.loc[current_step, 'Low'],
                                        self.df.loc[current_step, 'Close'],
                                        self.df.loc[current_step, 'Volume'],
                                        self.df.loc[current_step, 'SMA'],
                                        self.df.loc[current_step, 'TENKAN'],
                                        self.df.loc[current_step, 'KIJUN'],
                                        self.df.loc[current_step, 'senkou_span_a'],
                                        self.df.loc[current_step, 'SENKOU'],
                                        self.df.loc[current_step, 'CHIKOU'],
                                        self.df.loc[current_step, 'RSI'],
                                        self.df.loc[current_step, 'EMA']
                                        ])

        state = np.concatenate((self.market_history, self.orders_history), axis=1)
        return state

    # Get the data points for the given current_step
    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'Open'],
                                    self.df.loc[self.current_step, 'High'],
                                    self.df.loc[self.current_step, 'Low'],
                                    self.df.loc[self.current_step, 'Close'],
                                    self.df.loc[self.current_step, 'Volume'],
                                    self.df.loc[self.current_step, 'SMA'],
                                    self.df.loc[self.current_step, 'TENKAN'],
                                    self.df.loc[self.current_step, 'KIJUN'],
                                    self.df.loc[self.current_step, 'senkou_span_a'],
                                    self.df.loc[self.current_step, 'SENKOU'],
                                    self.df.loc[self.current_step, 'CHIKOU'],
                                    self.df.loc[self.current_step, 'RSI'],
                                    self.df.loc[self.current_step, 'EMA']
                                    ])
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
        return obs

    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to a random price between open and close
        current_price = random.uniform(
           self.df.loc[self.current_step, 'Open'],
           self.df.loc[self.current_step, 'Close'])
        # current_price = self.df.loc[self.current_step, 'Open']
        Date = self.df.loc[self.current_step, 'Date'] # for visualization
        High = self.df.loc[self.current_step, 'High'] # for visualization
        Low = self.df.loc[self.current_step, 'Low'] # for visualization

        if action == 0: # Hold
            pass

        elif action == 1 and self.balance > self.initial_balance/100:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.crypto_bought, 'type': "buy", 'current_price': current_price})
            self.episode_orders += 1

        elif action == 2 and self.crypto_held>0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.crypto_sold, 'type': "sell", 'current_price': current_price})
            self.episode_orders += 1

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

        # Receive calculated reward
        # reward = self.get_reward()
        reward = self.net_worth - self.prev_net_worth

        self._profits.append(self.net_worth - self.initial_balance)
        self._networths.append(self.net_worth)

        if self.net_worth <= self.initial_balance/2 or self.current_step == self.end_step:
            done = True
        else:
            done = False

        obs = self._next_observation() / self.normalize_value
        
        return obs, reward, done

    # Calculate reward
    def get_reward(self):
        self.punish_value += self.net_worth * 0.00001
        if self.episode_orders > 1 and self.episode_orders > self.prev_episode_orders:
            self.prev_episode_orders = self.episode_orders
            if self.trades[-1]['type'] == "buy" and self.trades[-2]['type'] == "sell":
                reward = self.trades[-2]['total']*self.trades[-2]['current_price'] - self.trades[-2]['total']*self.trades[-1]['current_price']
                reward -= self.punish_value
                self.punish_value = 0
                self.trades[-1]["Reward"] = reward
                return reward
            elif self.trades[-1]['type'] == "sell" and self.trades[-2]['type'] == "buy":
                reward = self.trades[-1]['total']*self.trades[-1]['current_price'] - self.trades[-2]['total']*self.trades[-2]['current_price']
                reward -= self.punish_value
                self.punish_value = 0
                self.trades[-1]["Reward"] = reward
                return reward
        else:
            return 0 - self.punish_value

    # render environment
    def render(self, visualize = False):
        #print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
        if visualize:
            Date = self.df.loc[self.current_step, 'Date']
            Open = self.df.loc[self.current_step, 'Open']
            Close = self.df.loc[self.current_step, 'Close']
            High = self.df.loc[self.current_step, 'High']
            Low = self.df.loc[self.current_step, 'Low']
            Volume = self.df.loc[self.current_step, 'Volume']

            # Render the environment to the screen
            return self.visualization.render(Date, Open, High, Low, Close, Volume, self.net_worth, self.trades)
    def render_all(self,path_name):
        plt.figure(figsize=(30,15))
        plt.plot(self._profits)
        plt.title(f'Average profit: {np.mean(self._profits)}',fontsize=20)
        plt.xlabel('Day',fontsize=20)
        plt.ylabel('Profit',fontsize=20)
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.savefig(f'{path_name}_Average_profit.png')

        plt.figure(figsize=(30,15))
        plt.plot(self._networths)
        plt.title(f'Average networths: {np.mean(self._networths)}',fontsize=20)
        plt.xlabel('Day',fontsize=20)
        plt.ylabel('Profit',fontsize=20)
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.savefig(f'{path_name}_Average_networths.png')

        w = open('DQN_HVN_AverageProfits_Log.txt','a+')
        w.write(f'{path_name}: {np.mean(self._profits)}\n')
        wn = open('DQN_HVN_AverageNetworths_Log.txt','a+')
        wn.write(f'{path_name}: {np.mean(self._networths)}\n')

        
def Random_games(env, visualize, test_episodes = 50, comment=""):
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action = np.random.randint(3, size=1)[0]
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                average_orders += env.episode_orders
                if env.net_worth < env.initial_balance: no_profit_episodes += 1 # calculate episode count where we had negative profit through episode
                print("episode: {}, net_worth: {}, average_net_worth: {}, orders: {}".format(episode, env.net_worth, average_net_worth/(episode+1), env.episode_orders))
                break

    print("average {} episodes random net_worth: {}, orders: {}".format(test_episodes, average_net_worth/test_episodes, average_orders/test_episodes))
    # save test results to test_results.txt file
    with open("test_results.txt", "a+") as results:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'{current_date}, {"Random games"}, test episodes:{test_episodes}')
        results.write(f', net worth:{average_net_worth/(episode+1)}, orders per episode:{average_orders/test_episodes}')
        results.write(f', no profit episodes:{no_profit_episodes}, comment: {comment}\n')

def train_agent(env, agent, visualize=False, train_episodes = 50, training_batch_size=500):
    agent.create_writer(env.initial_balance, env.normalize_value, train_episodes) # create TensorBoard writer
    total_average = deque(maxlen=100) # save recent 100 episodes net worth
    best_average = 0 # used to track best average net worth
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            env.render(visualize)
            action, prediction = agent.act(state)
            next_state, reward, done = env.step(action)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(3)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state

        a_loss, c_loss = agent.replay(states, actions, rewards, predictions, dones, next_states)
        total_average.append(env.net_worth)
        average = np.average(total_average)
        
        agent.writer.add_scalar('Data/average net_worth', average, episode)
        agent.writer.add_scalar('Data/episode_orders', env.episode_orders, episode)
        
        print("episode: {:<5} net worth {:<7.2f} average: {:<7.2f} orders: {}".format(episode, env.net_worth, average, env.episode_orders))
        if episode > len(total_average):
            if best_average < average:
                best_average = average
                print("Saving model")
                agent.save(score="{:.2f}".format(best_average), args=[episode, average, env.episode_orders, a_loss, c_loss])
            agent.save()
            
    agent.end_training_log()

def test_agent(env, agent, visualize=True, test_episodes=10, folder="", name="Crypto_trader", comment=""):
    agent.load(folder, name)
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            img = env.render(visualize)
            action, prediction = agent.act(state)
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                average_orders += env.episode_orders
                if env.net_worth < env.initial_balance: no_profit_episodes += 1 # calculate episode count where we had negative profit through episode
                print("episode: {:<5}, net_worth: {:<7.2f}, average_net_worth: {:<7.2f}, orders: {}".format(episode, env.net_worth, average_net_worth/(episode+1), env.episode_orders))
                break
            
    print("average {} episodes agent net_worth: {}, orders: {}".format(test_episodes, average_net_worth/test_episodes, average_orders/test_episodes))
    print("No profit episodes: {}".format(no_profit_episodes))
    # save test results to test_results.txt file
    with open("test_results.txt", "a+") as results:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'{current_date}, {name}, test episodes:{test_episodes}')
        results.write(f', net worth:{average_net_worth/(episode+1)}, orders per episode:{average_orders/test_episodes}')
        results.write(f', no profit episodes:{no_profit_episodes}, model: {agent.model}, comment: {comment}\n')
    return img



if __name__ == "__main__":            
    df = pd.read_csv('../data/vic_indicators.csv')
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d'))
    df = df.sort_values('Date')

    lookback_window_size = 10
    test_window = 60 # 60 sessions 
    train_df = df[:-test_window-lookback_window_size]
    test_df = df[-test_window-lookback_window_size:]


    # agent = CustomAgent(lookback_window_size=lookback_window_size, lr=0.0001, epochs=1, optimizer=Adam, batch_size = 32, model="CNN")
    # train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size, Show_reward=False)
    # dqn_agent = DQNAgent(train_env, lookback_window_size = lookback_window_size)
    # dqn_agent.train()

    test_env = CustomEnv(test_df, lookback_window_size=lookback_window_size, Show_reward=False)
    dqn_test_agent = DQNAgent(test_env, lookback_window_size = lookback_window_size)
    dqn_test_agent.test(episode=10)
    # img = test_agent(test_env, agent, visualize=True, test_episodes=1, folder="..\weights", name="fpt_Crypto_trader", comment="2021_07_09_CNN_vic")
    # cv2.imshow('res', img)
    # cv2.waitKey(0);
    
    
