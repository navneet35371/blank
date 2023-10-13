from blankly.exchanges.interfaces.exchange_interface import ExchangeInterface
import warnings
from upstox_client.rest import ApiException
import pandas as pd
import numpy as np

from blankly.exchanges.orders.limit_order import LimitOrder
from blankly.exchanges.orders.market_order import MarketOrder
from blankly.exchanges.orders.stop_loss import StopLossOrder
from blankly.exchanges.orders.take_profit import TakeProfitOrder
from blankly.utils import utils
from datetime import datetime as dt, timezone

from blankly.utils.exceptions import APIException
from blankly.utils.time_builder import build_minute, time_interval_to_seconds, number_interval_to_string
import time
from alpaca_trade_api.rest import TimeFrame


class UpstoxInterface(ExchangeInterface):
    def __init__(self, exchange_name, authenticated_api):
        self.__unique_assets = None
        super().__init__(exchange_name, authenticated_api, valid_resolutions=[60, 60 * 5, 60 * 15,
                                                                              60 * 60 * 24])
        self._user = self.calls['user']
        self._order = self.calls['order']
        self._market = self.calls['market']
        self._charge = self.calls['charge']
        self._history = self.calls['history']
        self._login = self.calls['login']
        self._portfolio = self.calls['portfolio']
        self._tradepl = self.calls['tradepl']
        self._websocket = self.calls['websocket']

    def init_exchange(self):
        try:
            account_info = self.calls['user'].get_profile('2.0')
        except ApiException as e:
            raise ApiException(e.__str__() + ". Are you trying to use your normal exchange keys "
                                             "while in sandbox mode? \nTry toggling the \'sandbox\' setting "
                                             "in your keys.json or check if the keys were input correctly into your "
                                             "keys.json.")
        try:
            if not account_info.data.is_active:
                warnings.warn('Your alpaca account is indicated as blocked for trading....')
        except KeyError:
            raise LookupError("alpaca API call failed")

        filtered_assets = ['NSE_EQ|INE040A01034']
        # products = self.calls.list_assets(status=None, asset_class=None)
        products = []
        for i in products:
            if i['symbol'] not in filtered_assets and i['status'] != 'inactive':
                filtered_assets.append(i['symbol'])
            else:
                # TODO handle duplicate symbols
                pass
        self.__unique_assets = filtered_assets

    @utils.enforce_base_asset
    def get_account(self, symbol=None):

        symbol = super().get_account(symbol)

        positions = self.calls['portfolio'].get_positions('2.0').data
        positions_dict = utils.AttributeDict({})
        print(positions)
        for position in positions:
            curr_symbol = position.instrument_token
            positions_dict[curr_symbol] = utils.AttributeDict({
                'available': float(position.quantity),
                'hold': 0.0
            })

        symbols = list(positions_dict.keys())
        # Catch an edge case bug that if there are no positions it won't try to snapshot
        if len(symbols) != 0:
            open_orders = self.calls['order'].get_order_book('2.0').data
            # snapshot_price = self.calls.get_snapshots(symbols=symbols)
        else:
            open_orders = []
            # snapshot_price = {}

        # now grab the available cash in the account
        account = self.calls['user'].get_user_fund_margin('2.0', segment='COM').data
        print(account)
        positions_dict['USD'] = utils.AttributeDict({
            'available': float(account['commodity'].available_margin),
            'hold': 0.0
        })
        print(open_orders)
        for order in open_orders:
            curr_symbol = order.instrument_token
            if order.transaction_type == 'BUY':  # buy orders only affect USD holds
                if order.quantity:  # this case handles qty market buy and limit buy
                    if order.order_type == 'LIMIT':
                        dollar_amt = float(order.quantity) * float(order.price)
                    elif order.order_type == 'MARKET':
                        dollar_amt = float(order.quantity) * positions_dict[curr_symbol]['last_price']
                    else:  # we don't have support for stop_order, stop_limit_order
                        dollar_amt = 0.0
                else:  # this is the case for notional market buy
                    dollar_amt = 0.0

                # In this case we don't have to subtract because the buying power is the available money already
                # we just need to add to figure out how much is actually on limits
                # positions_dict['USD']['available'] -= dollar_amt

                # So just add to our hold
                positions_dict['USD']['hold'] += dollar_amt

            else:
                if order.quantity:  # this case handles qty market sell and limit sell
                    qty = float(order.quantity)
                else:  # this is the case for notional market sell, calculate the qty with cash/price
                    qty = float(order['notional']) / positions_dict[curr_symbol]['last_price']

                positions_dict[curr_symbol]['available'] -= qty
                positions_dict[curr_symbol]['hold'] += qty

        # Note that now __unique assets could be uninitialized:
        if self.__unique_assets is None:
            self.init_exchange()

        for i in self.__unique_assets:
            if i not in positions_dict:
                positions_dict[i] = utils.AttributeDict({
                    'available': 0.0,
                    'hold': 0.0
                })

        if symbol is not None:
            if symbol in positions_dict:
                return utils.AttributeDict({
                    'available': float(positions_dict[symbol]['available']),
                    'hold': float(positions_dict[symbol]['hold'])
                })
            else:
                raise KeyError('Symbol not found.')

        if symbol == 'USD':
            return utils.AttributeDict({
                'available': positions_dict['USD']['available'],
                'hold': positions_dict['USD']['hold']
            })

        return positions_dict

    @utils.order_protection
    def stop_loss_order(self, symbol: str, price: float, size: float) -> StopLossOrder:
        side = 'sell'
        needed = self.needed['stop_loss']
        order = utils.build_order_info(price, side, size, symbol, 'stop_loss')

        response = self.calls.submit_order(symbol,
                                           side=side,
                                           type='stop',
                                           time_in_force='gtc',
                                           qty=size,
                                           stop_price=price)

        response = self._fix_response(needed, response)
        return StopLossOrder(order, response, self)

    def _fix_response(self, needed, response):
        response = self.__parse_iso(response)
        response = utils.rename_to([
            ['limit_price', 'price'],
            ['qty', 'size']
        ], response)
        response = utils.isolate_specific(needed, response)
        if 'time_in_force' in response:
            response['time_in_force'] = response['time_in_force'].upper()
        return response

    @utils.order_protection
    def take_profit_order(self, symbol: str, price: float, size: float) -> TakeProfitOrder:
        side = 'sell'
        needed = self.needed['take_profit']
        order = utils.build_order_info(price, side, size, symbol, 'take_profit')

        response = self.calls.submit_order(symbol,
                                           side=side,
                                           type='limit',
                                           time_in_force='gtc',
                                           qty=size,
                                           limit_price=price)

        response = self._fix_response(needed, response)
        return TakeProfitOrder(order, response, self)

    @utils.order_protection
    def market_order(self, symbol, side, size) -> MarketOrder:

        needed = self.needed['market_order']
        order = utils.build_order_info(0, side, size, symbol, 'market')

        response = self.calls.submit_order(symbol, side=side, type='market', time_in_force='day', qty=size)

        response = self._fix_response(needed, response)
        return MarketOrder(order, response, self)

    @utils.order_protection
    def limit_order(self, symbol: str, side: str, price: float, size: float) -> LimitOrder:
        needed = self.needed['limit_order']
        order = utils.build_order_info(price, side, size, symbol, 'limit')

        response = self.calls.submit_order(symbol,
                                           side=side,
                                           type='limit',
                                           time_in_force='gtc',
                                           qty=size,
                                           limit_price=price)

        response = self._fix_response(needed, response)
        return LimitOrder(order, response, self)

    def get_products(self) -> dict:
        """
        [
            {
              "id": "904837e3-3b76-47ec-b432-046db621571b",
              "class": "us_equity",
              "exchange": "NASDAQ",
              "symbol": "AAPL",
              "status": "active",
              "tradable": true,
              "marginable": true,
              "shortable": true,
              "easy_to_borrow": true,
              "fractionable": true
            },
            ...
        ]
        """
        needed = self.needed['get_products']
        assets = self.calls['assets'].get_assets_list()

        for asset in assets:
            base_asset = asset['instrument_key']
            asset['symbol'] = base_asset
            asset['base_asset'] = base_asset
            asset['quote_asset'] = 'USD'
            asset['base_min_size'] = 1
            asset['base_increment'] = 1
            asset['base_max_size'] = 10000000000

        for i in range(len(assets)):
            assets[i] = utils.isolate_specific(needed, assets[i])

        return assets

    def get_order_filter(self, symbol: str):
        current_price = self.get_price(symbol)

        products = self.get_products()

        product = None
        for i in products:
            if i['symbol'] == symbol:
                product = i
                break
        if product is None:
            raise APIException("Symbol not found.")

        exchange_specific = product['exchange_specific']
        fractionable = False

        if fractionable:
            quote_increment = 1e-9
            min_funds_buy = 1
            min_funds_sell = 1e-9 * current_price

            # base_min_size = product['base_min_size']
            base_max_size = product['base_max_size']
            # base_increment = product['base_increment']
            min_price = 0.0001
            max_price = 10000000000

            # Guaranteed nano share if fractionable
            base_min_size = 1e-9
            base_increment = 1e-9
        else:
            quote_increment = current_price
            min_funds_buy = current_price
            min_funds_sell = current_price

            # base_min_size = product['base_min_size']
            base_max_size = product['base_max_size']
            # base_increment = product['base_increment']
            min_price = 0.0001
            max_price = 10000000000

            # Always 1 if not fractionable
            base_min_size = 1
            base_increment = 1

        max_funds = current_price * 10000000000

        return {
            "symbol": symbol,
            "base_asset": symbol,
            "quote_asset": 'USD',
            "max_orders": 500,  # More than this and we can't calculate account value (alpaca is very bad)
            "limit_order": {
                "base_min_size": 1,  # Minimum size to buy
                "base_max_size": base_max_size,  # Maximum size to buy
                "base_increment": 1,  # Specifies the minimum increment for the base_asset.

                "price_increment": min_price,  # TODO test this at market open

                "min_price": min_price,
                "max_price": max_price,
            },
            'market_order': {
                "fractionable": fractionable,

                "base_min_size": base_min_size,  # Minimum size to buy
                "base_max_size": base_max_size,  # Maximum size to buy
                "base_increment": base_increment,  # Specifies the minimum increment for the base_asset.

                "quote_increment": quote_increment,  # Specifies the min order price as well as the price increment.
                "buy": {
                    "min_funds": min_funds_buy,
                    "max_funds": max_funds,
                },
                "sell": {
                    "min_funds": min_funds_sell,
                    "max_funds": max_funds,
                },
            },
            "exchange_specific": {
                "id": exchange_specific['instrument_key'],
                "class": exchange_specific['instrument_type'],
                "exchange": exchange_specific['exchange'],
                "status": "active",
                "tradable": True,
                "marginable": False,
                "shortable": False,
                "easy_to_borrow": False,
                "price": current_price
            }
        }

    @staticmethod
    def __parse_iso(response):
        from dateutil import parser
        try:
            response['created_at'] = parser.isoparse(response['created_at']).timestamp()
        except ValueError as e:
            if str(e) == 'Unused components in ISO string':
                response['created_at'] = parser.parse(response['created_at']).timestamp()
            else:
                raise e

        return response

    def cancel_order(self, symbol, order_id) -> dict:
        self.calls.cancel_order(order_id)

        # TODO: handle the different response codes
        return {'order_id': order_id}

    def get_open_orders(self, symbol=None):
        if symbol is None:
            orders = self.calls.list_orders(status='open')
        else:
            orders = self.calls.list_orders(status='open', symbols=[symbol])

        for i in range(len(orders)):
            # orders[i] = utils.rename_to(renames, orders[i])
            # if orders[i]['type'] == "limit":
            #     orders[i]['price'] = orders[i]['limit_price']
            # if orders[i]['type'] == "market":
            #     if orders[i]['notional']:
            #         orders[i]['funds'] = orders[i]['notional']
            #     else:
            #         orders[i]['funds'] = orders[i]['notional']
            # orders[i]['created_at'] = parser.isoparse(orders[i]['created_at']).timestamp()
            orders[i] = self.homogenize_order(orders[i])

        return orders

    def get_order(self, symbol, order_id) -> dict:
        order = self.calls.get_order(order_id)
        order = self.homogenize_order(order)
        return order

    # TODO: fix this function
    def homogenize_order(self, order):
        if order['type'] == "limit":
            renames = [
                ["qty", "size"],
                ["limit_price", "price"]
            ]
            order = utils.rename_to(renames, order)
        elif order['type'] == "stop_loss":
            renames = [
                ["qty", "size"],
                ["limit_price", "price"]
            ]
            order = utils.rename_to(renames, order)
        elif order['type'] == "take_profit":
            renames = [
                ["qty", "size"],
                ["limit_price", "price"]
            ]
            order = utils.rename_to(renames, order)
        elif order['type'] == "market":
            if order['notional']:
                renames = [
                    ["notional", "funds"]
                ]
                order = utils.rename_to(renames, order)

            else:  # market order of number of shares
                order['size'] = order.pop('qty')

        # TODO: test stop_limit orders
        elif order['type'] == "stop_limit":
            renames = [
                ["qty", "size"],
            ]
            order = utils.rename_to(renames, order)

        order = self.__parse_iso(order)
        if 'time_in_force' in order:
            order['time_in_force'] = order['time_in_force'].upper()

        needed = self.choose_order_specificity(order['type'])

        order = utils.isolate_specific(needed, order)
        return order

    def get_fees(self, symbol):
        return {
            'maker_fee_rate': 0.0,
            'taker_fee_rate': 0.0
        }

    @staticmethod
    def parse_yfinance(symbol: str, epoch_start: [int, float], epoch_stop: [int, float], resolution: int):
        try:
            import yfinance

            start_date = dt.fromtimestamp(epoch_start, tz=timezone.utc)
            stop_date = dt.fromtimestamp(epoch_stop, tz=timezone.utc)
            ticker = yfinance.Ticker(symbol)
            result = ticker.history(start=start_date, end=stop_date, interval=number_interval_to_string(resolution))

            result['time'] = result.index.astype(int) // 10 ** 9
            result = result[['Open', 'High', 'Low', 'Close', 'Volume', 'time']].reset_index()

            result = result.rename(columns={
                'time': 'time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            result = result[['time', 'open', 'high', 'low', 'close', 'volume']]

            return result
        except ImportError:
            raise ImportError("To use yfinance to download data please pip install yfinance")

    def get_product_history(self, symbol: str, epoch_start: float, epoch_stop: float, resolution: int):
        if not self.user_preferences['settings']['alpaca']['use_yfinance']:

            resolution = time_interval_to_seconds(resolution)

            supported_multiples = [60, 3600, 86400]
            if resolution not in supported_multiples:
                utils.info_print("Granularity is not an accepted granularity...rounding to nearest valid value.")
                resolution = supported_multiples[min(range(len(supported_multiples)),
                                                     key=lambda i: abs(supported_multiples[i] - resolution))]

            found_multiple, row_divisor = super().evaluate_multiples(supported_multiples, resolution)

            if found_multiple == 60:
                time_interval = TimeFrame.Minute
            elif found_multiple == 3600:
                time_interval = TimeFrame.Hour
            else:
                time_interval = TimeFrame.Day

            epoch_start_str = dt.fromtimestamp(epoch_start, tz=timezone.utc).date()
            epoch_stop_str = dt.fromtimestamp(epoch_stop, tz=timezone.utc).date()

            try:
                try:
                    data = self.calls['history'].get_historical_candle_data1(symbol, 'day', epoch_stop_str, epoch_start_str,
                                               '2.0').data
                    bars = pd.DataFrame(data.candles, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'valid'])
                except Exception as e:
                    # If you query a timeframe with no data the API throws a Nonetype issue so just return something
                    #  empty if that happens
                    return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            except Exception as e:
                if e == 42210000:
                    warning_string = "Your alpaca subscription does not permit querying data from the last 15 " \
                                     "minutes. Blankly is adjusting your query."
                    utils.info_print(warning_string)
                    epoch_stop = time.time() - (build_minute() * 15)
                    if epoch_stop >= epoch_start:
                        try:
                            return self.get_product_history(symbol, epoch_stop, epoch_start, resolution)
                        except TypeError:
                            # If you query a timeframe with no data the API throws a Nonetype issue so just
                            #  return something
                            #  empty if that happens
                            return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                    else:
                        warning_string = "No data range queried after time adjustment."
                        utils.info_print(warning_string)
                        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                else:
                    raise e
            bars.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"},
                        inplace=True)

            del bars['valid']
            bars['time'] = pd.to_datetime(bars['time']).values.astype(np.int64) // 10 ** 6
            print(bars)

            if bars.empty:
                return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            return utils.get_ohlcv(bars, row_divisor, from_zero=False)
        else:
            # This runs yfinance on the symbol
            return self.parse_yfinance(symbol, epoch_start, epoch_stop, resolution)

    def get_price(self, symbol) -> float:
        # response = self.calls['market'].ltp(symbol, '2.0')
        # keys = list(response.data.keys())
        # return float(response.data[keys[0]].last_price)
        return float(1560.0)
