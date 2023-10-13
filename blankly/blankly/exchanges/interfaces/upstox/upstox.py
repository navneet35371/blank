from blankly.exchanges.auth.auth_constructor import AuthConstructor
from blankly.exchanges.exchange import Exchange
import upstox_client

from blankly.exchanges.interfaces.upstox.upstox_api import UpstoxAPI


class Upstox(Exchange):
    def __init__(self, portfolio_name=None, keys_path="keys.json", settings_path=None):
        Exchange.__init__(self, "upstox", portfolio_name, settings_path)

        # Load the auth from the keys file
        auth = AuthConstructor(keys_path, portfolio_name, 'alpaca', ['API_KEY', 'API_SECRET', 'sandbox'])

        sandbox = super().evaluate_sandbox(auth)
        configuration = upstox_client.Configuration()
        configuration.access_token = 'eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIxMDkxMjAiLCJqdGkiOiI2NTI4ZTU4NzNiMzNiNzAyNTY1NTEyNTAiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNBY3RpdmUiOnRydWUsInNjb3BlIjpbImludGVyYWN0aXZlIiwiaGlzdG9yaWNhbCJdLCJpYXQiOjE2OTcxNzkwMTUsImlzcyI6InVkYXBpLWdhdGV3YXktc2VydmljZSIsImV4cCI6MTY5NzIzNDQwMH0.EOt0csvRj5y_0CIEHrGc_aOdRhXwBzNWBZn_Pz7Y4L8'

        calls = {
            'user': upstox_client.UserApi(upstox_client.ApiClient(configuration)),
            'order': upstox_client.OrderApi(upstox_client.ApiClient(configuration)),
            'market': upstox_client.MarketQuoteApi(upstox_client.ApiClient(configuration)),
            'charge': upstox_client.ChargeApi(upstox_client.ApiClient(configuration)),
            'history': upstox_client.HistoryApi(upstox_client.ApiClient(configuration)),
            'login': upstox_client.LoginApi(upstox_client.ApiClient(configuration)),
            'portfolio': upstox_client.PortfolioApi(upstox_client.ApiClient(configuration)),
            'tradepl': upstox_client.TradeProfitAndLossApi(upstox_client.ApiClient(configuration)),
            'websocket': upstox_client.WebsocketApi(upstox_client.ApiClient(configuration)),
            'assets': UpstoxAPI()
        }

        # Always finish the method with this function
        super().construct_interface_and_cache(calls)

    def get_exchange_state(self):
        return self.interface.get_products()

    def get_asset_state(self, symbol):
        return self.interface.get_account(symbol)

    def get_direct_calls(self):
        return self.calls

    def get_market_clock(self):
        return self.calls.get_clock()
