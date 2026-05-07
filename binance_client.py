import os
import time
import hmac
import hashlib
import json
import requests
import logging
from urllib.parse import urlencode
from config import Config

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self, config: Config):
        self.config = config
        self.api_key = config.BINANCE_API_KEY
        self.api_secret = config.BINANCE_API_SECRET
        self.use_testnet = config.USE_TESTNET
        
        if self.use_testnet:
            self.base_url = 'https://testnet.binance.vision'  # spot testnet
            self.futures_base = 'https://testnet.binancefuture.com'  # futures testnet
        else:
            self.base_url = 'https://api.binance.com'
            self.futures_base = 'https://fapi.binance.com'
        
        self.session = requests.Session()
        self.session.headers.update({'X-MBX-APIKEY': self.api_key})
        
        # For futures, set leverage on init if needed
        if config.LEVERAGE > 1:
            self._set_leverage(config.SYMBOL, config.LEVERAGE)
    
    def _sign(self, params: dict) -> str:
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False, futures: bool = False):
        base = self.futures_base if futures else self.base_url
        url = f"{base}{endpoint}"
        params = params or {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._sign(params)
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, params=params, timeout=10)
            elif method == 'DELETE':
                response = self.session.delete(url, params=params, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e} - URL: {url}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time: int = None, end_time: int = None, futures: bool = True):
        """Fetch historical candles. futures=True for USDT-M futures"""
        endpoint = '/fapi/v1/klines' if futures else '/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000)
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        data = self._request('GET', endpoint, params, signed=False, futures=futures)
        
        # Parse to DataFrame friendly
        columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                   'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                   'taker_buy_quote_asset_volume', 'ignore']
        df_data = []
        for row in data:
            df_data.append({
                'timestamp': int(row[0]),
                'open': float(row[1]),
                'high': float(row[2]),
                'low': float(row[3]),
                'close': float(row[4]),
                'volume': float(row[5])
            })
        return df_data
    
    def get_account_balance(self, futures: bool = True):
        """Get USDT balance"""
        if futures:
            endpoint = '/fapi/v2/account'
            data = self._request('GET', endpoint, signed=True, futures=True)
            for asset in data.get('assets', []):
                if asset['asset'] == 'USDT':
                    return float(asset['availableBalance'])
            return 0.0
        else:
            endpoint = '/api/v3/account'
            data = self._request('GET', endpoint, signed=True, futures=False)
            for bal in data.get('balances', []):
                if bal['asset'] == 'USDT':
                    return float(bal['free'])
            return 0.0
    
    def get_position(self, symbol: str, futures: bool = True):
        """Get current position for symbol"""
        if not futures:
            return {'positionAmt': 0, 'entryPrice': 0, 'unRealizedProfit': 0}
        
        endpoint = '/fapi/v2/positionRisk'
        data = self._request('GET', endpoint, {'symbol': symbol}, signed=True, futures=True)
        for pos in data:
            if pos['symbol'] == symbol:
                return {
                    'positionAmt': float(pos['positionAmt']),
                    'entryPrice': float(pos['entryPrice']),
                    'unRealizedProfit': float(pos['unRealizedProfit']),
                    'leverage': float(pos['leverage'])
                }
        return {'positionAmt': 0, 'entryPrice': 0, 'unRealizedProfit': 0}
    
    def place_order(self, symbol: str, side: str, order_type: str = 'MARKET', 
                    quantity: float = None, price: float = None, 
                    stop_price: float = None, reduce_only: bool = False, futures: bool = True):
        """Place order. side: BUY/SELL, order_type: MARKET/LIMIT/STOP_MARKET"""
        endpoint = '/fapi/v1/order' if futures else '/api/v3/order'
        
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
        }
        
        if quantity:
            params['quantity'] = f"{quantity:.6f}"
        if price and order_type == 'LIMIT':
            params['price'] = f"{price:.2f}"
            params['timeInForce'] = 'GTC'
        if stop_price and 'STOP' in order_type:
            params['stopPrice'] = f"{stop_price:.2f}"
        if reduce_only:
            params['reduceOnly'] = 'true'
        
        try:
            result = self._request('POST', endpoint, params, signed=True, futures=futures)
            logger.info(f"Order placed: {result}")
            return result
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return None
    
    def close_position(self, symbol: str, futures: bool = True):
        """Close entire position"""
        pos = self.get_position(symbol, futures)
        amt = abs(pos['positionAmt'])
        if amt > 0:
            side = 'SELL' if pos['positionAmt'] > 0 else 'BUY'
            return self.place_order(symbol, side, 'MARKET', quantity=amt, reduce_only=True, futures=futures)
        return None
    
    def _set_leverage(self, symbol: str, leverage: int):
        try:
            endpoint = '/fapi/v1/leverage'
            params = {'symbol': symbol, 'leverage': leverage}
            self._request('POST', endpoint, params, signed=True, futures=True)
            logger.info(f"Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            logger.warning(f"Could not set leverage: {e}")
