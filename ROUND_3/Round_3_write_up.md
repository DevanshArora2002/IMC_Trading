There are 2 ‘asset classes’ in the three products you trade. The HYDROGEL_PACK and VELVETFRUIT_EXTRACT are “delta 1” products, similar to the products in the tutorial and rounds 1 and 2. 
The 10 VELVETFRUIT_EXTRACT_VOUCHER products (each with a different strike price) are options, and thus follow different dynamics. All products are traded independently, even though the price of VELVETFRUIT_EXTRACT_VOUCHER might be related to that of VELVETFRUIT_EXTRACT due to the nature of options.

The vouchers are labeled `VEV_4000`, `VEV_4500`, `VEV_5000`, `VEV_5100`, `VEV_5200`, `VEV_5300`, `VEV_5400`, `VEV_5500`, `VEV_6000`, `VEV_6500`, where VEV stands for **V**elvetfruit **E**xtract **V**oucher, and the number represents the strike price. They all have a 7-day expiration deadline starting from round 1, where each round represents 1 day. Thus, the ‘time till expiry’ (TTE) is 7 days at the start of round 1 (TTE=7d), 6 days at the start of round 2, 5 days at the start of round 3, and so on.

The position limits ([see the Position Limits page for extra context and troubleshooting]

- `HYDROGEL_PACK`: 200
- `VELVETFRUIT_EXTRACT`: 200
- `VELVETFRUIT_EXTRACT_VOUCHER`: 300 for each of the 10 vouchers.


**Example**: `VEV_5000` is an option on the underlying VEV with a strike price of 5000 and a position limit of 300. At the start of the final simulation of Round 3, its time to expiry (TTE) is 5 days. In the historical data, the corresponding TTE values are:

- TTE=8d at the start of historical day 0 (coinciding with the tutorial round),
- TTE=7d at the start of historical day 1 (coinciding with Round 1),
- TTE=6d at the start of historical day 2 (coinciding with Round 2).


Vouchers cannot be exercised before their expiry, and inventory does not carry over into the next round. Like in previous rounds, any open positions are automatically liquidated against a hidden fair value at the end of the round.

**HINT**
Hello everyone, hope you're enjoying the VELVETFRUIT_EXTRACT vouchers and a variety of trading strategies these new products introduce. While digging for the rock, Archipelago residents found some ancient mathematics sharing insights into VELVETFRUIT_EXTRACT voucher trading. Here's what the message with obscure and advanced mathematics read,

Message begins,

I have discovered a strategy which will make ArchiCapital the biggest trading company ever. Here's how my thesis goes,

t: Timestamp
St: Voucher Underlying Price at t
K: Strike
TTE: Remaining Time till expiry at t
Vt: Voucher price of strike K at t

Compute,

m_t = log(K/St)/ sqrt(TTE)
v_t = BlackScholes ImpliedVol(St, Vt, K, TTE)

for each t, plot v_t vs m_t and fit a parabolic curve to filter random noise.

This fitted v_t(m_t) allows me to evaluate opportunities between different strikes. I also call fitted v_t(m_t=0) the base IV and I have identified interesting patterns in timeseries of base IV.

Pro Trader's Viewpoint:
Major insight came from following hints dropped in the competition wiki, suggesting the construction of a volatility smile: plotting implied volatility (IV) against moneyness. By fitting a parabola to the observed IVs across strikes and then detrending (subtracting the fitted curve from observed values), we could isolate IV deviations that were no longer dependent on moneyness.

The expected value from gamma scalping was consistently positive, as the gains from underlying price movements outweighed the losses from time decay. This made buying options and rehedging the resulting deltas from gamma exposure a relatively low-risk way to generate profit. However, while the approach was stable and mostly safe, the absolute returns were limited. It was a reliable source of small gains, but ultimately, we had a higher risk appetite and wanted better returns.


## Sample Code :
```python
OPTION_SYMBOLS = [
    'VEV_4000',
    'VEV_4500',
    'VEV_5000',
    'VEV_5500',
    'VEV_6000',
    'VEV_6500',
]

OPTION_UNDERLYING_SYMBOL = 'VELVETFRUIT_EXTRACT'

####### OPTIONS ####### OPTIONS ####### OPTIONS ####### OPTIONS ####### OPTIONS ####### OPTIONS ####### OPTIONS ####### OPTIONS  

DAY = 5

DAYS_PER_YEAR = 365

THR_OPEN, THR_CLOSE = 0.5, 0
LOW_VEGA_THR_ADJ = 0.5

THEO_NORM_WINDOW = 20

IV_SCALPING_THR = 0.7
IV_SCALPING_WINDOW = 100

# UNDERLYING
underlying_mean_reversion_thr = 15
underlying_mean_reversion_window = 10

# OPTIONS
options_mean_reversion_thr = 5
options_mean_reversion_window = 30



class ProductTrader:

    def __init__(self, name, state, prints, new_trader_data, product_group=None):

        self.orders = []

        self.name = name
        self.state = state
        self.prints = prints
        self.new_trader_data = new_trader_data
        self.product_group = name if product_group is None else product_group

        self.last_traderData = self.get_last_traderData()

        self.position_limit = POS_LIMITS.get(self.name, 0)
        self.initial_position = self.state.position.get(self.name, 0) # position at beginning of round

        self.expected_position = self.initial_position # update this if you expect a certain change in position e.g. to already hedge


        self.mkt_buy_orders, self.mkt_sell_orders = self.get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self.get_walls()
        self.best_bid, self.best_ask = self.get_best_bid_ask()

        self.max_allowed_buy_volume, self.max_allowed_sell_volume = self.get_max_allowed_volume() # gets updated when order created
        self.total_mkt_buy_volume, self.total_mkt_sell_volume = self.get_total_market_buy_sell_volume()

    def get_last_traderData(self):
                        
        last_traderData = {}
        try:
            if self.state.traderData != '':
                last_traderData = json.loads(self.state.traderData)
        except: self.log("ERROR", 'td')

        return last_traderData


    def get_best_bid_ask(self):

        best_bid = best_ask = None

        try:
            if len(self.mkt_buy_orders) > 0:
                best_bid = max(self.mkt_buy_orders.keys())
            if len(self.mkt_sell_orders) > 0:
                best_ask = min(self.mkt_sell_orders.keys())
        except: pass

        return best_bid, best_ask


    def get_walls(self):

        bid_wall = wall_mid = ask_wall = None

        try: bid_wall = min([x for x,_ in self.mkt_buy_orders.items()])
        except: pass
        
        try: ask_wall = max([x for x,_ in self.mkt_sell_orders.items()])
        except: pass

        try: wall_mid = (bid_wall + ask_wall) / 2
        except: pass

        return bid_wall, wall_mid, ask_wall
    
    def get_total_market_buy_sell_volume(self):

        market_bid_volume = market_ask_volume = 0

        try:
            market_bid_volume = sum([v for p, v in self.mkt_buy_orders.items()])
            market_ask_volume = sum([v for p, v in self.mkt_sell_orders.items()])
        except: pass

        return market_bid_volume, market_ask_volume
    

    def get_max_allowed_volume(self):
        max_allowed_buy_volume = self.position_limit - self.initial_position
        max_allowed_sell_volume = self.position_limit + self.initial_position
        return max_allowed_buy_volume, max_allowed_sell_volume

    def get_order_depth(self):

        order_depth, buy_orders, sell_orders = {}, {}, {}

        try: order_depth: OrderDepth = self.state.order_depths[self.name]
        except: pass
        try: buy_orders = {bp: abs(bv) for bp, bv in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)}
        except: pass
        try: sell_orders = {sp: abs(sv) for sp, sv in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])}
        except: pass

        return buy_orders, sell_orders
    

    def bid(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
        order = Order(self.name, int(price), abs_volume)
        if logging: self.log("BUYO", {"p":price, "s":self.name, "v":int(volume)}, product_group='ORDERS')
        self.max_allowed_buy_volume -= abs_volume
        self.orders.append(order)

    def ask(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
        order = Order(self.name, int(price), -abs_volume)
        if logging: self.log("SELLO", {"p":price, "s":self.name, "v":int(volume)}, product_group='ORDERS')
        self.max_allowed_sell_volume -= abs_volume
        self.orders.append(order)

    def log(self, kind, message, product_group=None):
        if product_group is None: product_group = self.product_group

        if product_group == 'ORDERS':
            group = self.prints.get(product_group, [])
            group.append({kind: message})
        else:
            group = self.prints.get(product_group, {})
            group[kind] = message

        self.prints[product_group] = group

    def check_for_informed(self):

        informed_direction, informed_bought_ts, informed_sold_ts = NEUTRAL, None, None
        

        informed_bought_ts, informed_sold_ts = self.last_traderData.get(self.name, [None, None])

        trades = self.state.market_trades.get(self.name, []) + self.state.own_trades.get(self.name, [])

        for trade in trades:
            if trade.buyer == INFORMED_TRADER_ID:
                informed_bought_ts = trade.timestamp
            if trade.seller == INFORMED_TRADER_ID: 
                informed_sold_ts = trade.timestamp
            
        self.new_trader_data[self.name] = [informed_bought_ts, informed_sold_ts]

        informed_sold = informed_sold_ts is not None
        informed_bought = informed_bought_ts is not None

        if not informed_bought and not informed_sold:
            informed_direction = NEUTRAL

        elif not informed_bought and informed_sold:
            informed_direction = SHORT

        elif informed_bought and not informed_sold:
            informed_direction = LONG

        elif informed_bought and informed_sold:
            if informed_sold_ts > informed_bought_ts:
                informed_direction = SHORT
            elif informed_sold_ts < informed_bought_ts:
                informed_direction = LONG
            else:
                informed_direction = NEUTRAL

        self.log('TD', self.new_trader_data[self.name])
        self.log('ID', informed_direction)

        return informed_direction, informed_bought_ts, informed_sold_ts


    def get_orders(self):
        # overwrite this in each trader
        return {}


class OptionTrader:
    def __init__(self, state, prints, new_trader_data):

        self.options = [ProductTrader(os, state, prints, new_trader_data, product_group='OPTION') for os in OPTION_SYMBOLS]
        self.underlying = ProductTrader(OPTION_UNDERLYING_SYMBOL, state, prints, new_trader_data, product_group='OPTION')

        self.state = state
        self.last_traderData = self.underlying.last_traderData
        self.new_trader_data = new_trader_data

        self.indicators = self.calculate_indicators()


    def get_option_values(self, S, K, TTE):

        def bs_call(S, K, TTE, s, r=0):        
            d1 = (math.log(S/K) + (r + 0.5 * s**2) * TTE) / (s * TTE**0.5)
            d2 = d1 - s * TTE**0.5
            return S * _N.cdf(d1) - K * math.exp(-r * TTE) * _N.cdf(d2), _N.cdf(d1)

        def bs_vega(S, K, TTE, s, r=0):
            d1 = d1 = (math.log(S/K) + (r + 0.5*s**2) * TTE) / (s * TTE**0.5)
            return S * _N.pdf(d1) * TTE**0.5

        def get_iv(St, K, TTE):
            m_t_k = np.log(K/St) / TTE**0.5
            coeffs = [0.27362531, 0.01007566, 0.14876677] # from the fitted vol smile
            iv = np.poly1d(coeffs)(m_t_k)
            return iv

        iv = get_iv(S, K, TTE)
        bs_call_value, delta = bs_call(S, K, TTE, iv)
        vega = bs_vega(S, K, TTE, iv)
        return bs_call_value, delta, vega
    

    def calculate_ema(self, td_key, window, value):
        old_mean = self.last_traderData.get(td_key, 0)
        alpha = 2/(window+1)
        new_mean = alpha * value + (1 - alpha) * old_mean
        self.new_trader_data[td_key] = new_mean

        return new_mean



    def calculate_indicators(self):

        indicators = {
            'ema_u_dev': None,
            'ema_o_dev': None,
            'mean_theo_diffs': {},
            'current_theo_diffs': {},
            'switch_means': {},
            'deltas': {},
            'vegas': {},
        }


        if self.underlying.wall_mid is not None:

            new_mean_price = self.calculate_ema('ema_u', underlying_mean_reversion_window, self.underlying.wall_mid)
            indicators['ema_u_dev'] = self.underlying.wall_mid - new_mean_price

            new_mean_price = self.calculate_ema('ema_o', options_mean_reversion_window, self.underlying.wall_mid)
            indicators['ema_o_dev'] = self.underlying.wall_mid - new_mean_price


            for option in self.options:

                k = int(option.name.split('_')[-1])

                if option.wall_mid is None:
                    if option.ask_wall is not None:
                        option.wall_mid = option.ask_wall - 0.5
                        option.bid_wall = option.ask_wall - 1
                        option.best_bid = option.ask_wall - 1
                    elif option.bid_wall is not None:
                        option.wall_mid = option.bid_wall + 0.5
                        option.ask_wall = option.bid_wall + 1
                        option.best_ask = option.bid_wall + 1


                if option.wall_mid is not None:

                    tte = 1 - (DAYS_PER_YEAR - 8 + DAY + self.state.timestamp // 100 / 10_000) / DAYS_PER_YEAR
                    underlying = self.underlying.best_bid * 0.5 + self.underlying.best_ask * 0.5
                    option_theo, option_delta, option_vega = self.get_option_values(underlying, k, tte)
                    option_theo_diff = option.wall_mid - option_theo

                    indicators['current_theo_diffs'][option.name] = option_theo_diff
                    indicators['deltas'][option.name] = option_delta
                    indicators['vegas'][option.name] = option_vega


                    new_mean_diff = self.calculate_ema(f'{option.name}_theo_diff', THEO_NORM_WINDOW, option_theo_diff)
                    indicators['mean_theo_diffs'][option.name] = new_mean_diff


                    new_mean_avg_dev = self.calculate_ema(f'{option.name}_avg_devs', IV_SCALPING_WINDOW, abs(option_theo_diff - new_mean_diff))
                    indicators['switch_means'][option.name] = new_mean_avg_dev

        return indicators
    

    def get_iv_scalping_orders(self, options):

        out = {}

        for option in options:

            if option.name in self.indicators['mean_theo_diffs'] and option.name in self.indicators['current_theo_diffs'] and option.name in self.new_switch_mean:

                if self.new_switch_mean[option.name] >= IV_SCALPING_THR:

                    current_theo_diff = self.indicators['current_theo_diffs'][option.name]
                    mean_theo_diff = self.indicators['mean_theo_diffs'][option.name]

                    low_vega_adj = 0
                    if self.vegas.get(option.name, 0) <= 1:
                        low_vega_adj = LOW_VEGA_THR_ADJ


                    if current_theo_diff - option.wall_mid + option.best_bid - mean_theo_diff >= (THR_OPEN + low_vega_adj) and option.max_allowed_sell_volume > 0:
                        option.ask(option.best_bid, option.max_allowed_sell_volume)

                    if current_theo_diff - option.wall_mid + option.best_bid - mean_theo_diff >= THR_CLOSE and option.initial_position > 0:
                        option.ask(option.best_bid, option.initial_position)

                    elif current_theo_diff - option.wall_mid + option.best_ask - mean_theo_diff <= -(THR_OPEN + low_vega_adj) and option.max_allowed_buy_volume > 0:
                        option.bid(option.best_ask, option.max_allowed_buy_volume)
                        
                    if current_theo_diff - option.wall_mid + option.best_ask - mean_theo_diff <= -THR_CLOSE and option.initial_position < 0:
                        option.bid(option.best_ask, -option.initial_position)

                else:

                    if option.initial_position > 0:
                        option.ask(option.best_bid, option.initial_position)
                    elif option.initial_position < 0:
                        option.bid(option.best_ask, -option.initial_position)


            out[option.name] = option.orders

        return out
    
    def get_mr_orders(self, options):

        out = {}

        for option in options:

            if option.name in self.indicators['current_theo_diffs'] and option.name in self.indicators['mean_theo_diffs'] and self.indicators.get('ema_o_dev') is not None:

                current_deviation = self.indicators['ema_o_dev']

                iv_deviation = self.indicators['current_theo_diffs'][option.name] - self.indicators['mean_theo_diffs'][option.name]
                current_deviation += iv_deviation

                if current_deviation > options_mean_reversion_thr and option.max_allowed_sell_volume > 0:
                    option.ask(option.best_bid, option.max_allowed_sell_volume)

                elif current_deviation < -options_mean_reversion_thr and option.max_allowed_buy_volume > 0:
                    option.bid(option.best_ask, option.max_allowed_buy_volume)

                out[option.name] = option.orders

        return out


    def get_option_orders(self):

        if self.state.timestamp / 100 < min([THEO_NORM_WINDOW, underlying_mean_reversion_window, options_mean_reversion_window]): return {}

        iv_scalping_options = [o for o in self.options if int(o.name.split('_')[-1]) >= 9750]
        mr_options = [o for o in self.options if o.name.endswith('9500')]


        out = {
            **self.get_iv_scalping_orders(iv_scalping_options),
            **self.get_mr_orders(mr_options)
        }

        return out
    
    
    def get_underlying_orders(self):

        if self.state.timestamp / 100 < underlying_mean_reversion_window: return {}

        if self.indicators.get('ema_u_dev') is not None:

            current_deviation = self.indicators['ema_o_dev']

            if current_deviation > underlying_mean_reversion_thr and self.underlying.max_allowed_sell_volume > 0:
                self.underlying.ask(self.underlying.bid_wall + 1, self.underlying.max_allowed_sell_volume)

            elif current_deviation < -underlying_mean_reversion_thr and self.underlying.max_allowed_buy_volume > 0:
                self.underlying.bid(self.underlying.ask_wall - 1, self.underlying.max_allowed_buy_volume)


        return {self.underlying.name: self.underlying.orders}


    def get_orders(self):

        orders = {
            **self.get_option_orders(), # order important, first option, then hedge
            **self.get_underlying_orders()
        }

        return orders

class Trader:

    def run(self, state: TradingState):
        result:dict[str,list[Order]] = {}
        new_trader_data = {}
        prints = {
            "GENERAL": {
                "TIMESTAMP": state.timestamp,
                "POSITIONS": state.position
            },
        }

        def export(prints):
            try: print(json.dumps(prints))
            except: pass


        product_traders = {
            STATIC_SYMBOL: StaticTrader,
            DYNAMIC_SYMBOL: DynamicTrader,
            INK_SYMBOL: InkTrader,
            ETF_BASKET_SYMBOLS[0]: EtfTrader,
            OPTION_UNDERLYING_SYMBOL: OptionTrader,
            COMMODITY_SYMBOL: CommodityTrader,
        }

        result, conversions = {}, 0
        for symbol, product_trader in product_traders.items():
            if symbol in state.order_depths:

                try:
                    trader = product_trader(state, prints, new_trader_data)
                    result.update(trader.get_orders())

                    if symbol == COMMODITY_SYMBOL:
                        conversions = trader.get_conversions()
                except: pass


        try: final_trader_data = json.dumps(new_trader_data)
        except: final_trader_data = ''


        export(prints)
        return result, conversions, final_trader_data