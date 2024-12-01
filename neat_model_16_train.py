# Import necessary libraries
import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import ta
from scipy.stats import linregress
import neat
import time
import pickle
from datetime import datetime, timedelta


def sync_to_1_day():
    now = datetime.now()
    # Calculate the next midnight (next day at 00:00)
    next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    # Calculate the seconds until the next midnight
    seconds_left = (next_midnight - now).total_seconds()
    # Sleep until the next midnight
    if seconds_left > 0:
        time.sleep(seconds_left)


while True:
    # Initialize tvDatafeed
    tv = TvDatafeed()


    # Fetch data with retry logic
    def fetch_data_with_retry(tv, symbol, exchange, interval, n_bars, retries=3):
        """Fetch historical data with retry logic."""
        for attempt in range(retries):
            try:
                df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
                if df is not None and not df.empty:
                    return df
                else:
                    print(f"No data returned for {symbol}. Retrying... (Attempt {attempt + 1}/{retries})")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}. Retrying... (Attempt {attempt + 1}/{retries})")
            time.sleep(2)  # Wait before retrying
        return None  # Return None if all retries fail


    # Specify parameters
    symbol_used = 'SOLUSDT.P'
    platform = 'OKX'
    n_bars = 6400

    # Fetch historical data for symbol_used
    df_symbolused = fetch_data_with_retry(tv, symbol_used, platform, Interval.in_5_minute, n_bars)

    if df_symbolused is None:
        raise ValueError("Failed to fetch data for symbol_used.")

    # Rename columns for consistency
    df_symbolused.columns = ['symbol', 'open_symbolused', 'high_symbolused', 'low_symbolused', 'close_symbolused',
                             'volume_symbolused']

    # Symbols for additional data
    sol_symbol = 'SOL'
    btc_symbol = 'BTC'
    total_symbol = 'TOTAL'
    platform_cryptocap = 'CRYPTOCAP'


    # Function to calculate slope over a rolling window
    def calculate_slope(series, window=7):
        slopes = [np.nan] * (window - 1)  # Fill initial rows with NaN
        for i in range(window - 1, len(series)):
            y = series[i - window + 1:i + 1]
            x = np.arange(window)
            # Perform linear regression and extract the slope
            slope, _, _, _, _ = linregress(x, y)
            slopes.append(slope)
        return pd.Series(slopes, index=series.index)


    def calculate_rolling_std_high_low(high_series, low_series, window=7):
        """
        Calculate the rolling standard deviation using the average of high and low prices over a specified window.

        Parameters:
            high_series (pd.Series): The high prices series.
            low_series (pd.Series): The low prices series.
            window (int): The rolling window size.

        Returns:
            pd.Series: A series containing the rolling standard deviation of the average of high and low prices.
        """
        # Calculate the average of high and low prices
        avg_high_low = (high_series + low_series) / 2

        # Initialize a list with NaN for the initial values where the rolling window can't be applied
        rolling_std = [np.nan] * (window - 1)

        # Calculate the rolling standard deviation
        for i in range(window - 1, len(avg_high_low)):
            window_data = avg_high_low[i - window + 1:i + 1]
            std_dev = np.std(window_data)
            rolling_std.append(std_dev)
        return pd.Series(rolling_std, index=high_series.index)


    def calculate_rolling_correlation(series1, series2, window=7):
        """
        Calculate the rolling correlation between two price series over a specified window.

        Parameters:
            series1 (pd.Series): The first price series (e.g., BTC close prices).
            series2 (pd.Series): The second price series (e.g., ETH close prices).
            window (int): The rolling window size (default is 7).

        Returns:
            pd.Series: A series containing the rolling correlation values.
        """
        # Calculate the rolling correlation
        rolling_corr = series1.rolling(window).corr(series2)
        return rolling_corr


    # Fetch OHLC data for SOL ,BTC and TOTAL
    df_sol = fetch_data_with_retry(tv, sol_symbol, platform_cryptocap, Interval.in_5_minute, n_bars)
    df_total = fetch_data_with_retry(tv, total_symbol, platform_cryptocap, Interval.in_5_minute, n_bars)
    df_btc = fetch_data_with_retry(tv, btc_symbol, platform_cryptocap, Interval.in_5_minute, n_bars)

    # Check if the data is fetched successfully
    if df_sol is None or df_total is None or df_btc is None:
        raise ValueError("Failed to fetch data for BTC or TOTAL.")

    # Rename columns for consistency
    df_sol.columns = ['symbol', 'open_sol', 'high_sol', 'low_sol', 'close_sol', 'volume_sol']
    df_total.columns = ['symbol', 'open_total', 'high_total', 'low_total', 'close_total', 'volume_total']
    df_btc.columns = ['symbol', 'open_btc', 'high_btc', 'low_btc', 'close_btc', 'volume_btc']

    # Merge the datasets on index (align the timestamps)
    df_combined = pd.concat([df_symbolused, df_sol, df_total, df_btc], axis=1)
    df_combined.dropna(inplace=True)  # Drop NaN values due to potential mismatched timestamps

    # Calculate Stochastic Slow and EMAs for the symbolused
    data = df_combined
    data['ema5'] = ta.trend.EMAIndicator(close=data['close_symbolused'], window=5).ema_indicator()
    data['ema8'] = ta.trend.EMAIndicator(close=data['close_symbolused'], window=8).ema_indicator()
    data['ema100'] = ta.trend.EMAIndicator(close=data['close_symbolused'], window=50).ema_indicator()

    #Calculate Sol Dominance
    data['open_sol_dom'] = data['open_sol']*100/data['open_total']
    data['close_sol_dom'] = data['close_sol'] * 100 / data['close_total']

    #Calculate BTC Dominance
    data['open_btc_dom'] = data['open_btc']*100/data['open_total']
    data['close_btc_dom'] = data['close_btc'] * 100 / data['close_total']

    data.dropna(inplace=True)


    def signal_symbolused(text, num_past,r):
        # Initialize the column with zeros for the entire length of the DataFrame
        data[text] = [0] * len(data)
        for i in range(r,len(data)-1):
            if data['close_symbolused'].iloc[i] > data['open_symbolused'].iloc[i-num_past]:
                data[text].iloc[i] = 1
            elif data['close_symbolused'].iloc[i] < data['open_symbolused'].iloc[i-num_past]:
                data[text].iloc[i] = -1
            else:
                data[text].iloc[i] = 0

    def signal_btc(text, num_past,r):
        data[text] = [0] * len(data)
        for i in range(r, len(data) - 1):
            if data['close_btc'].iloc[i] > data['open_btc'].iloc[i - num_past]:
                data[text].iloc[i] = 1
            elif data['close_btc'].iloc[i] < data['open_btc'].iloc[i - num_past]:
                data[text].iloc[i] = -1
            else:
                data[text].iloc[i] = 0


    def signal_sol(text, num_past,r):
        data[text] = [0] * len(data)
        for i in range(r, len(data) - 1):
            if data['close_sol_dom'].iloc[i] > data['open_sol_dom'].iloc[i - num_past]:
                data[text].iloc[i] = 1
            elif data['close_sol_dom'].iloc[i] < data['open_sol_dom'].iloc[i - num_past]:
                data[text].iloc[i] = -1
            else:
                data[text].iloc[i] = 0

    #Add new datas
    signal_symbolused('symbolused_sig_0',0,0)
    signal_symbolused('symbolused_sig_1',1,1)
    signal_symbolused('symbolused_sig_2',2,2)
    signal_symbolused('symbolused_sig_3', 3, 3)
    signal_symbolused('symbolused_sig_4', 4, 4)
    signal_symbolused('symbolused_sig_5', 5, 5)
    signal_symbolused('symbolused_sig_6', 6, 6)
    signal_symbolused('symbolused_sig_7', 7, 7)
    signal_symbolused('symbolused_sig_8', 8, 8)
    signal_symbolused('symbolused_sig_9', 9, 9)
    signal_symbolused('symbolused_sig_10', 10, 10)
    signal_symbolused('symbolused_sig_11', 11, 11)
    signal_symbolused('symbolused_sig_12', 12, 12)
    signal_symbolused('symbolused_sig_13', 13, 13)
    signal_symbolused('symbolused_sig_14', 14, 14)

    data.dropna(inplace=True)


    data_inputs = pd.DataFrame({
        'symbolused_sig_0':data['symbolused_sig_0'],
        'symbolused_sig_1': data['symbolused_sig_1'],
        'symbolused_sig_2': data['symbolused_sig_2'],
        'symbolused_sig_3': data['symbolused_sig_3'],
        'symbolused_sig_4': data['symbolused_sig_4'],
        'symbolused_sig_5': data['symbolused_sig_5'],
        'symbolused_sig_6': data['symbolused_sig_6'],
        'symbolused_sig_7': data['symbolused_sig_7'],
        'symbolused_sig_8': data['symbolused_sig_8'],
        'symbolused_sig_9': data['symbolused_sig_9'],
        'symbolused_sig_10': data['symbolused_sig_10'],
        'symbolused_sig_11': data['symbolused_sig_11'],
        'symbolused_sig_12': data['symbolused_sig_12'],
        'symbolused_sig_13': data['symbolused_sig_13'],
        'symbolused_sig_14': data['symbolused_sig_14'],
        'close_symbolused': data['close_symbolused'],
        'high_symbolused': data['high_symbolused'],
        'low_symbolused': data['low_symbolused'],
        'open_symbolused': data['open_symbolused'],
        'close_total': data['close_total'],
        'high_total': data['high_total'],
        'low_total': data['low_total'],
        'open_total': data['open_total'],
        'close_btc': data['close_btc'],
        'high_btc': data['high_btc'],
        'low_btc': data['low_btc'],
        'open_btc': data['open_btc'],
        'open_sol_dom':data['open_sol_dom'],
        'close_sol_dom':data['close_sol_dom'],
        'open_btc_dom': data['open_btc_dom'],
        'close_btc_dom': data['close_btc_dom'],
        'ema5': data['ema5'],
        'ema8': data['ema8'],
        'ema100': data['ema100'],
        })

    # Split data into two halves for training and testing
    first_half = data_inputs[:len(data_inputs) - 2000]
    second_half = data_inputs[len(data_inputs) - 2000:]


    # Define a trading environment with a feedback loop
    class TradingEnvironment:
        def __init__(self, data, starting_balance=20):
            self.data = data
            self.balance = starting_balance
            self.current_step = 15
            self.open_trade = []
            self.close_trade = []
            self.action_list = []
            self.equity_history = [starting_balance]
            self.profits = [float(starting_balance)]
            self.entry = []
            self.stop_loss = []
            self.take_profit = []
            self.leverage = []
            self.amount = []
            self.side = []
            self.fit = [float(0)]
            self.close_price = data['close_symbolused'].values
            self.open_price = data['open_symbolused'].values
            self.high_price = data['high_symbolused'].values
            self.low_price = data['low_symbolused'].values
            self.ema5 = data['ema5'].values
            self.ema8 = data['ema8'].values
            self.ema100 = data['ema100'].values
            self.open_sol_dom = data['open_sol_dom']
            self.close_sol_dom = data['close_sol_dom']
            self.open_btc_dom = data['open_btc_dom']
            self.close_btc_dom = data['close_btc_dom']
            self.close_btc = data['close_btc']
            self.open_btc = data['open_btc']
            self.high_btc = data['high_btc']
            self.low_btc = data['low_btc']
            self.close_total = data['close_total']
            self.open_total = data['open_total']
            self.high_total = data['high_total']
            self.low_total = data['low_total']



        def reset(self):
            self.balance = 20
            self.current_step = 15
            self.open_trade = []
            self.close_trade = []
            self.action_list = []
            self.equity_history = [self.balance]
            self.profits = [float(self.balance)]
            self.entry = []
            self.stop_loss = []
            self.take_profit = []
            self.leverage = []
            self.amount = []
            self.side = []
            self.fit = [float(0)]

        def step(self, action):
            # Actions: 0 = hold, 1 = buy, -1 = sell
            price = self.close_price[self.current_step]

            ema5_current, ema8_current, ema100_current = self.ema5[self.current_step], self.ema8[self.current_step], \
                self.ema100[self.current_step]
            ema5_prev, ema8_prev, ema100_prev = self.ema5[self.current_step - 1], self.ema8[self.current_step - 1], \
                self.ema100[self.current_step - 1]

            step = 1

            if action == 1 and ema5_prev < ema8_prev and ema5_current > ema8_current and self.close_price[
                self.current_step] > self.open_price[self.current_step] and self.close_price[
                self.current_step] > ema100_current and self.close_price[
                self.current_step - 1] < ema100_prev and len(self.entry) == 0:
                loss_pct = 0.03
                gain_pct = 0.15
                stop_loss = self.low_price[self.current_step - 1]
                leverage = abs(loss_pct / ((stop_loss - price) / price))
                tp_limit = ((gain_pct * price) / abs(-loss_pct / ((stop_loss - price) / price))) + price
                self.open_trade.append(float(1))
                self.action_list.append(action)
                self.entry.append(float(price))
                self.stop_loss.append(float(stop_loss))
                self.take_profit.append(float(tp_limit))
                self.leverage.append(leverage)
                self.side.append(1)
                self.amount.append(sum(self.profits) - (sum(self.profits) * self.leverage[0] * 0.0005))

            elif action == 1 and ema5_prev > ema8_prev and ema5_current < ema8_current and self.close_price[
                self.current_step] < self.open_price[self.current_step] and self.close_price[
                self.current_step] < ema100_current and self.close_price[
                self.current_step - 1] > ema100_prev and len(self.entry) == 0:
                loss_pct = 0.03
                gain_pct = 0.15
                stop_loss = self.high_price[self.current_step - 1]
                leverage = abs(loss_pct / ((stop_loss - price) / price))
                tp_limit = ((-gain_pct * price) / abs(loss_pct / ((stop_loss - price) / price))) + price
                self.open_trade.append(float(1))
                self.leverage.append(leverage)
                self.action_list.append(action)
                self.entry.append(float(price))
                self.stop_loss.append(float(stop_loss))
                self.take_profit.append(float(tp_limit))
                self.side.append(-1)
                self.amount.append(sum(self.profits) - (sum(self.profits) * self.leverage[0] * 0.0005))

            elif action == -1 and len(self.entry) > 0:
                self.action_list.append(-1)
                if price > self.entry[0] and self.side[0] == 1 and self.low_price[self.current_step] > self.stop_loss[0] and self.high_price[self.current_step] < self.take_profit[0]:
                    pct = abs((price - self.entry[0]) / self.entry[0])
                    self.profits.append((self.amount[0] * self.leverage[0] * pct) - (self.amount[0] * self.leverage[0] * 0.0005))
                    self.fit.append(self.leverage[0]*pct/0.03)
                    self.close_trade.append(float(1))
                    self.entry.clear()
                    self.take_profit.clear()
                    self.stop_loss.clear()
                    self.amount.clear()
                    self.leverage.clear()
                    self.side.clear()

                elif price < self.entry[0] and self.side[0] == 1 and self.low_price[self.current_step] > self.stop_loss[0] and self.high_price[self.current_step] < self.take_profit[0]:
                    pct = abs((price - self.entry[0]) / self.entry[0])
                    self.profits.append(-(self.amount[0] * self.leverage[0] * pct) - (self.amount[0] * self.leverage[0] * 0.0005))
                    self.fit.append(-self.leverage[0] * pct / 0.03)
                    self.close_trade.append(float(1))
                    self.entry.clear()
                    self.take_profit.clear()
                    self.stop_loss.clear()
                    self.amount.clear()
                    self.leverage.clear()
                    self.side.clear()

                elif price > self.entry[0] and self.side[0] == -1 and self.high_price[self.current_step] < self.stop_loss[0] and self.low_price[self.current_step] > self.take_profit[0]:
                    pct = abs((price - self.entry[0]) / self.entry[0])
                    self.profits.append(-(self.amount[0] * self.leverage[0] * pct) - (self.amount[0] * self.leverage[0] * 0.0005))
                    self.fit.append(-self.leverage[0] * pct / 0.03)
                    self.close_trade.append(float(1))
                    self.entry.clear()
                    self.take_profit.clear()
                    self.stop_loss.clear()
                    self.amount.clear()
                    self.leverage.clear()
                    self.side.clear()

                elif price < self.entry[0] and self.side[0] == -1 and self.high_price[self.current_step] < self.stop_loss[0] and self.low_price[self.current_step] > self.take_profit[0]:
                    pct = abs((price - self.entry[0]) / self.entry[0])
                    self.profits.append((self.amount[0] * self.leverage[0] * pct) - (self.amount[0] * self.leverage[0] * 0.0005))
                    self.fit.append(self.leverage[0] * pct / 0.03)
                    self.close_trade.append(float(1))
                    self.entry.clear()
                    self.take_profit.clear()
                    self.stop_loss.clear()
                    self.amount.clear()
                    self.leverage.clear()
                    self.side.clear()

            elif len(self.side) > 0:
                self.action_list.append(0)

                if self.side[0] == 1 and self.low_price[self.current_step] <= self.stop_loss[0]:
                    self.balance -= self.balance * 0.03
                    self.profits.append(-(self.amount[0] * 0.03) - (self.amount[0] * self.leverage[0] * 0.0005))
                    self.fit.append(float(-1))
                    self.entry.clear()
                    self.take_profit.clear()
                    self.stop_loss.clear()
                    self.amount.clear()
                    self.leverage.clear()
                    self.side.clear()

                elif len(self.side)>0 and len(self.take_profit)>0 and self.high_price[self.current_step]>0:
                    if self.side[0] == 1 and self.high_price[self.current_step] >= self.take_profit[0]:
                        self.balance += self.balance * 0.15
                        self.profits.append((self.amount[0] * 0.15) - (self.amount[0] * self.leverage[0] * 0.0005))
                        self.fit.append(float(0))
                        self.entry.clear()
                        self.take_profit.clear()
                        self.stop_loss.clear()
                        self.amount.clear()
                        self.leverage.clear()
                        self.side.clear()
                elif len(self.side)>0 and len(self.stop_loss)>0 and self.high_price[self.current_step]>0:
                    if self.side[0] == -1 and self.high_price[self.current_step] >= self.stop_loss[0]:
                        self.balance -= self.balance * 0.03
                        self.profits.append(-(self.amount[0] * 0.03) - (self.amount[0] * self.leverage[0] * 0.0005))
                        self.fit.append(float(-1))
                        self.entry.clear()
                        self.take_profit.clear()
                        self.stop_loss.clear()
                        self.amount.clear()
                        self.leverage.clear()
                        self.side.clear()

                elif len(self.side)>0 and len(self.take_profit)>0 and self.low_price[self.current_step]>0:
                    if self.side[0] == -1 and self.low_price[self.current_step] <= self.take_profit[0]:
                        self.balance += self.balance * 0.15
                        self.profits.append((self.amount[0] * 0.15) - (self.amount[0] * self.leverage[0] * 0.0005))
                        self.fit.append(float(0))
                        self.entry.clear()
                        self.take_profit.clear()
                        self.stop_loss.clear()
                        self.amount.clear()
                        self.leverage.clear()
                        self.side.clear()

            else:
                self.action_list.append(0)

            self.current_step += step
            self.equity_history.append(self.balance)
            done = self.current_step >= len(self.data) - 1
            return self.balance, done, sum(self.open_trade), sum(self.close_trade), sum(self.profits), self.action_list, sum(self.fit)


    def run_neat(config_path, save_path="best_genome3.pkl", generations=999999999999):
        """
        Runs the NEAT algorithm with a fitness function and saves the best genome.

        Args:
            config_path (str): Path to the NEAT configuration file.
            save_path (str): Path to save the best genome and its network.
            generations (int): Number of generations to run.
            winrate_weight (float): Weight of winrate in the fitness score.
            pnl_weight (float): Weight of profit and loss (PNL) in the fitness score.
        """
        # Load NEAT configuration
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        # Create a population
        p = neat.Population(config)

        # Add reporters to monitor progress
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        # Function to prune unconnected nodes
        def prune_unconnected_nodes(genome):
            """
            Removes nodes from the genome that are not part of any enabled connection.
            Parameters:
                genome: The genome to prune nodes from.
            """
            # Gather all nodes involved in enabled connections
            connected_nodes = set()
            for conn_key, conn in genome.connections.items():
                if conn.enabled:  # Only consider enabled connections
                    connected_nodes.add(conn_key[0])  # Add input node
                    connected_nodes.add(conn_key[1])  # Add output node

            # Remove nodes not connected to any enabled connection
            for node_key in list(genome.nodes.keys()):
                if node_key not in connected_nodes:
                    del genome.nodes[node_key]  # Remove the node

        profits_2 = 20
        generation = 0
        # Define fitness function for NEAT
        def evaluate_genomes(genomes, config, prune_every=10):
            nonlocal best_genome, best_net, profits_2, generation  # Use outer-scope variables
            best_fitness = float('-inf')
            best_test = float('-inf')

            saved_best_fitness = 0
            saved_test_fitness = 0

            best_genome = None
            best_net = None


            for genome_id, genome in genomes:
                genome.fitness = 0.0  # Initialize fitness

                # try:
                # Create the neural network for the genome
                net = neat.nn.RecurrentNetwork.create(genome, config)
                env = TradingEnvironment(first_half)  # Assumes `data_inputs` is defined globally or passed in
                env.reset()

                env2 = TradingEnvironment(second_half)
                env2.reset()


                total_profit = 0
                while True:
                    def signal():
                        if env.ema5[env.current_step -1] < env.ema8[env.current_step -1] and env.ema5[env.current_step] > env.ema8[env.current_step] and env.close_price[env.current_step] > env.open_price[env.current_step] and env.close_price[env.current_step] > env.ema100[env.current_step] and env.close_price[env.current_step - 1] < env.ema100[env.current_step-1]:
                            signal = 1
                        elif env.ema5[env.current_step -1] > env.ema8[env.current_step -1] and env.ema5[env.current_step] < env.ema8[env.current_step] and env.close_price[env.current_step] < env.open_price[env.current_step] and env.close_price[env.current_step] < env.ema100[env.current_step] and env.close_price[env.current_step - 1] > env.ema100[env.current_step-1]:
                            signal = -1
                        else:
                            signal = 0
                        return signal

                    def tp_distance():
                        global comp
                        if len(env.side)>0:
                            if env.side[0] == 1 and env.close_price[env.current_step]>env.entry[0] and env.close_price[env.current_step]<=env.take_profit[0]:
                                comp = (abs(env.close_price[env.current_step]-env.entry[0])*env.leverage[0]/env.entry[0])/0.15
                            elif env.side[0] == -1 and env.close_price[env.current_step]<env.entry[0] and env.close_price[env.current_step]>=env.take_profit[0]:
                                comp = (abs(env.close_price[env.current_step] - env.entry[0]) * env.leverage[0] /env.entry[0])/0.15
                            else:
                                comp = 0
                        else:
                            comp = 0
                        return comp

                    def sl_distance():
                        global comp
                        if len(env.side)>0:
                            if env.side[0] == 1 and env.close_price[env.current_step]<env.entry[0] and env.close_price[env.current_step]>=env.stop_loss[0]:
                                comp = (abs(env.close_price[env.current_step]-env.entry[0])*env.leverage[0]/env.entry[0])/0.03
                            elif env.side[0] == -1 and env.close_price[env.current_step]>env.entry[0] and env.close_price[env.current_step]<=env.stop_loss[0]:
                                comp = (abs(env.close_price[env.current_step] - env.entry[0]) * env.leverage[0] /env.entry[0])/0.03
                            else:
                                comp = 0
                        else:
                            comp = 0
                        return comp


                    # Ensure state is compatible with neural network input size
                    state = np.concatenate([[signal()], env.data.iloc[env.current_step, :15].values, [tp_distance()], [sl_distance()]])
                    action = np.argmax(net.activate(state)) - 1  # Map to -1, 0, 1

                    # Execute the action in the environment
                    balance, done, open_trade, close_trade, total_profits, action_list, fit = env.step(action)
                    if done:
                        break


                while True:
                    def signal2():
                        if env2.ema5[env2.current_step -1] < env2.ema8[env2.current_step -1] and env2.ema5[env2.current_step] > env2.ema8[env2.current_step] and env2.close_price[env2.current_step] > env2.open_price[env2.current_step] and env2.close_price[env2.current_step] > env2.ema100[env2.current_step] and env2.close_price[env2.current_step - 1] < env2.ema100[env2.current_step-1]:
                            signal = 1
                        elif env2.ema5[env2.current_step -1] > env2.ema8[env2.current_step -1] and env2.ema5[env2.current_step] < env2.ema8[env2.current_step] and env2.close_price[env2.current_step] < env2.open_price[env2.current_step] and env2.close_price[env2.current_step] < env2.ema100[env2.current_step] and env2.close_price[env2.current_step - 1] > env2.ema100[env2.current_step-1]:
                            signal = -1
                        else:
                            signal = 0
                        return signal

                    def tp_distance2():
                        global comp
                        if len(env2.side)>0:
                            if env2.side[0] == 1 and env2.close_price[env2.current_step]>env2.entry[0] and env2.close_price[env2.current_step]<=env2.take_profit[0]:
                                comp = (abs(env2.close_price[env2.current_step]-env2.entry[0])*env2.leverage[0]/env2.entry[0])/0.15
                            elif env2.side[0] == -1 and env2.close_price[env2.current_step]<env2.entry[0] and env2.close_price[env2.current_step]>=env2.take_profit[0]:
                                comp = (abs(env2.close_price[env2.current_step] - env2.entry[0]) * env2.leverage[0] /env2.entry[0]) / 0.15
                            else:
                                comp = 0
                        else:
                            comp = 0
                        return comp

                    def sl_distance2():
                        global comp
                        if len(env2.side)>0:
                            if env2.side[0] == 1 and env2.close_price[env2.current_step]<env2.entry[0] and env2.close_price[env2.current_step]>=env2.stop_loss[0]:
                                comp = (abs(env2.close_price[env2.current_step]-env2.entry[0])*env2.leverage[0]/env2.entry[0])/0.03
                            elif env2.side[0] == -1 and env2.close_price[env2.current_step]>env2.entry[0] and env2.close_price[env2.current_step]<=env2.stop_loss[0]:
                                comp = (abs(env2.close_price[env2.current_step] - env2.entry[0]) * env2.leverage[0] /env2.entry[0])/0.03
                            else:
                                comp = 0
                        else:
                            comp = 0
                        return comp

                    # Ensure state is compatible with neural network input size
                    state2 = np.concatenate([[signal2()], env2.data.iloc[env2.current_step, :15].values, [tp_distance2()], [sl_distance2()]])
                    action2 = np.argmax(net.activate(state2)) - 1  # Map to -1, 0, 1

                    # Execute the action in the environment
                    balance2, done2, open_trade2, close_trade2, total_profits2, action_list2, fit2 = env2.step(action2)
                    if done2:
                        break

                total_profit = total_profits
                PNL = (total_profit - 20) * 100 / 20

                total_profit2 = total_profits2
                PNL2 = (total_profit2 - 20) * 100 / 20

                genome.fitness = max(fit, 0)


                # Print genome performance
                print(f'Trader: {genome_id}, PNL_Train%: {round(PNL, 2)}% - (Open/Close: {round(open_trade,0)}/{round(close_trade,0)}), PNL2_Test%: {round(PNL2,2)}% - (Open/Close: {round(open_trade2,0)}/{round(close_trade2,0)}), Fitness: {round(fit,2)}')

                # Track the best genome
                if genome.fitness > best_fitness:
                    best_fitness = genome.fitness
                    best_genome = genome
                    best_net = net
                    profits_2 = total_profit2


            generation +=1

            # Save the best genome and network after evaluations
            if best_genome:
                saved_best_fitness = best_fitness
                saved_test_fitness = best_test
                with open(save_path, 'wb') as f:
                    pickle.dump({'genome': best_genome, 'network': best_net}, f)
                print(f"Saved the best genome and network to {save_path}")
            print(f'Saved Best Fitness: {saved_best_fitness}, Saved Test Fitness: {saved_test_fitness}')

        # Initialize variables for the best genome and network
        best_genome = None
        best_net = None

        prune_every = 10
        generation = 0
        # Run the NEAT algorithm
        p.run(lambda genomes,config: evaluate_genomes(genomes, config, prune_every), generations)
        print('\nBest genome:\n{!s}'.format(best_genome))


    def create_neat_config():
        config_content = """
        [NEAT]
        pop_size = 100
        fitness_criterion = max
        fitness_threshold = 999999999999999999999
        reset_on_extinction = True

        [DefaultGenome]
        feed_forward = False
        
        # Node activation functions
        activation_default = tanh
        activation_mutate_rate = 0.1
        activation_options = tanh sigmoid
        
        # Node aggregation functions
        aggregation_default = sum
        aggregation_mutate_rate = 0.0
        aggregation_options = sum mean
        
        # Structural mutation rates
        single_structural_mutation = True
        structural_mutation_surer = 0
        conn_add_prob = 0.6
        conn_delete_prob = 0.1
        node_add_prob = 0.2
        node_delete_prob = 0.2
        
        # Connection parameters
        initial_connection = partial 0.5
        bias_init_mean = 0.0
        bias_init_stdev = 1.0
        bias_max_value = 10.0
        bias_min_value = -10.0
        bias_mutate_power = 0.3
        bias_mutate_rate = 0.1
        bias_replace_rate = 0.1
        
        # Response parameters
        response_init_mean = 0.0
        response_init_stdev = 1.0
        response_replace_rate = 0.1
        response_mutate_rate = 0.1
        response_mutate_power = 0.3
        response_max_value = 10.0
        response_min_value = -10.0
        
        # Default enabled state
        enabled_default = True
        
        # Enable mutation rate
        enabled_mutate_rate = 0.1
        
        # Node parameters
        num_hidden = 0
        num_inputs = 18
        num_outputs = 3
        
        # Connection mutation
        weight_init_mean = 0.0
        weight_init_stdev = 1.0
        weight_max_value = 10.0
        weight_min_value = -10.0
        weight_mutate_power = 0.3
        weight_mutate_rate = 0.5
        weight_replace_rate = 0.1
        
        # Compatibility parameters
        compatibility_disjoint_coefficient = 1.0
        compatibility_weight_coefficient = 0.5

        [DefaultSpeciesSet]
        compatibility_threshold = 3.0

        [DefaultStagnation]
        species_fitness_func = max
        max_stagnation = 15
        species_elitism = 2

        [DefaultReproduction]
        elitism = 2
        survival_threshold = 0.2
        """
        with open('neat_config6.txt', 'w') as f:
            f.write(config_content)


    # Function to test the trained NEAT model on the test data
    def test_model(net, test_data, config_path):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        # net = neat.nn.FeedForwardNetwork.create(genome, config)  # Create the neural network from the best genome
        env2 = TradingEnvironment(test_data)  # Initialize the environment with test data
        env2.reset()  # Reset the environment

        def signal_symbolused_price(field, num_past):
            if field.close_price[field.current_step] > field.open_price[field.current_step - num_past]:
                signal = 1
            elif field.close_price[field.current_step] < field.open_price[field.current_step - num_past]:
                signal = -1
            else:
                signal = 0
            return signal

        def signal_btc_price(field, num_past):
            if field.close_btc[field.current_step] > field.open_btc[field.current_step - num_past]:
                signal = 1
            elif field.close_btc[field.current_step] < field.open_btc[field.current_step - num_past]:
                signal = -1
            else:
                signal = 0
            return signal

        def signal_sol_dom(field, num_past):
            if field.close_sol_dom[field.current_step] > field.open_sol_dom[field.current_step - num_past]:
                signal = 1
            elif field.close_sol_dom[field.current_step] < field.open_sol_dom[
                field.current_step - num_past]:
                signal = -1
            else:
                signal = 0
            return signal

        total_profit = 0  # Variable to accumulate profit during testing
        while True:
            def signal2():
                if env2.ema5[env2.current_step - 1] < env2.ema8[env2.current_step - 1] and env2.ema5[env2.current_step] > env2.ema8[env2.current_step] and env2.close_price[env2.current_step] > env2.open_price[env2.current_step] and env2.close_price[env2.current_step] > env2.ema100[env2.current_step] and env2.close_price[env2.current_step - 1] < env2.ema100[env2.current_step - 1]:
                    signal = 1
                elif env2.ema5[env2.current_step - 1] > env2.ema8[env2.current_step - 1] and env2.ema5[env2.current_step] < env2.ema8[env2.current_step] and env2.close_price[env2.current_step] < env2.open_price[env2.current_step] and env2.close_price[env2.current_step] < env2.ema100[env2.current_step] and env2.close_price[env2.current_step - 1] > env2.ema100[env2.current_step - 1]:
                    signal = -1
                else:
                    signal = 0
                return signal

            # Ensure state is compatible with neural network input size
            state2 = np.concatenate(
                [[signal2()], [signal_symbolused_price(env2, 0)], [signal_symbolused_price(env2, 1)],
                 [signal_symbolused_price(env2, 2)], [signal_symbolused_price(env2, 3)],
                 [signal_symbolused_price(env2, 4)], [signal_symbolused_price(env2, 5)],
                 [signal_symbolused_price(env2, 6)], [signal_symbolused_price(env2, 7)],
                 [signal_symbolused_price(env2, 8)], [signal_symbolused_price(env2, 9)],
                 [signal_symbolused_price(env2, 10)], [signal_symbolused_price(env2, 11)],
                 [signal_symbolused_price(env2, 12)], [signal_symbolused_price(env2, 13)],
                 [signal_symbolused_price(env2, 14)], [signal_btc_price(env2, 0)], [signal_btc_price(env2, 1)],
                 [signal_btc_price(env2, 2)], [signal_btc_price(env2, 3)], [signal_btc_price(env2, 4)],
                 [signal_btc_price(env2, 5)], [signal_btc_price(env2, 6)], [signal_btc_price(env2, 7)],
                 [signal_btc_price(env2, 8)], [signal_btc_price(env2, 9)], [signal_btc_price(env2, 10)],
                 [signal_btc_price(env2, 11)], [signal_btc_price(env2, 12)], [signal_btc_price(env2, 13)],
                 [signal_btc_price(env2, 14)], [signal_sol_dom(env2, 0)], [signal_sol_dom(env2, 1)],
                 [signal_sol_dom(env2, 2)], [signal_sol_dom(env2, 3)], [signal_sol_dom(env2, 4)],
                 [signal_sol_dom(env2, 5)], [signal_sol_dom(env2, 6)], [signal_sol_dom(env2, 7)],
                 [signal_sol_dom(env2, 8)], [signal_sol_dom(env2, 9)], [signal_sol_dom(env2, 10)],
                 [signal_sol_dom(env2, 11)], [signal_sol_dom(env2, 12)], [signal_sol_dom(env2, 13)],
                 [signal_sol_dom(env2, 14)]])
            action2 = np.argmax(net.activate(state2)) - 1  # Map to -1, 0, 1

            # Execute the action in the environment
            balance, done, open, close, total_profits2, action_list, fit = env2.step(action2)
            if done:
                break

        total_profits = total_profit  # Calculate profit (final balance - initial balance)
        return total_profits


    def load_best_genome(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            winner_genome = data['genome']
            net = data['network']  # This is the loaded network, fully instantiated
        return winner_genome, net


    if __name__ == "__main__":
        # Provide path to your NEAT config file
        # Create configuration file
        create_neat_config()
        config_path = "neat_config6.txt"
        run_neat(config_path)
        best_genome, network = load_best_genome('best_genome3.pkl')
        test = test_model(network, second_half, config_path)
        print(f'total balance: {test}')
        sync_to_1_day()





