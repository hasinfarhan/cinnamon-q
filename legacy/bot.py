import os
import math
import time
import datetime as DT
import traceback
from typing import Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import random
import logging

import pandas as pd
import numpy as np

# https://python-binance.readthedocs.io/en/latest
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException

# https://apscheduler.readthedocs.io/en/3.x/userguide.html#installing-apscheduler
from apscheduler.schedulers.blocking import BlockingScheduler


class Config:
    def __init__(self):
        load_dotenv()
        self.BINANCE_API_KEY = os.getenv("API_KEY")
        self.BINANCE_API_SECRET_KEY = os.getenv("API_SECRET_KEY")
        if not self.BINANCE_API_KEY or not self.BINANCE_API_SECRET_KEY:
            raise ValueError("API_KEY and API_SECRET_KEY must be set in the .env file")
        self.TRADING_FREQUENCY_MINUTES = 15
        self.MAX_TRADE_COUNT = 3
        self.MAX_RETRIES = 2
        self.POLL_INTERVAL_SECONDS = 5
        self.LIMIT_ORDER_SPREAD = 0.002
        self.BASE_CURRENCY = 'USDT'
        self.TRADE_BUDGET_USDT = 300

@dataclass
class Trade:
    symbol: str
    side: str
    quantity: float
    in_price: float
    profit_price: float
    loss_price: float
    price_precision: int
    
@dataclass
class State:
    open_symbol: str = None
    trade_count: int = 0
    

class Bot:
    def __init__(self, config: Config, binance: Client):
        self.config = config
        self.binance = binance
        self.state = State()
        self.scheduler = BlockingScheduler()

    '''
    Trading Algo V1:

    - Pickes a ticker randomly from high RSI pairs
    - Sets exit levels
    - Makes short/SELL order
    '''
    def __get_best_trade_v1(self) -> Trade | None:
        all_symbols = self.all_symbols()
        drop_assests = self.get_potential_drop_assets(all_symbols)
        symbol = random.choice(drop_assests)
        filters = self.get_symbol_filters(symbol)
        
        side = SIDE_SELL
        price = float(self.get_symbol_price(symbol))
        price = self.adjust_price(price=price, filters=filters)
        profit_price = self.adjust_price(price = price * 0.975, filters=filters, round_up=True)
        loss_price = self.adjust_price(price = price * 1.025, filters=filters, round_up=True)
        quantity = self.config.TRADE_BUDGET_USDT / price
        quantity = self.adjust_quantity(quantity=quantity, filters=filters)
        price_precision = self.get_price_precision(filters)

        if quantity * price > self.config.TRADE_BUDGET_USDT:
            raise Exception("Budget overflows!")

        return Trade(
            symbol = symbol,
            side = side,
            quantity = quantity,
            in_price = price,
            profit_price = profit_price,
            loss_price =  loss_price,
            price_precision = price_precision
        )
    
    def get_historical_klines(self, symbol, interval):
        """Get historical klines from Binance."""
        # look back last 2h data
        timestamp = pd.Timestamp.now() - pd.Timedelta('2 hours')
        klines = self.binance.get_historical_klines(symbol=symbol, interval=interval, start_str=str(timestamp))
        data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                                            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                                            'taker_buy_quote_asset_volume', 'ignore'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        data = data.astype(float)
        return data

    def calculate_rsi(self, data, window):
        """Calculate the Relative Strength Index (RSI)."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_potential_drop_assets(self, symbols):
        """Identify assets with high potential to drop in price."""
        potential_drops = []
        for symbol in symbols:
            # calculates RSI based on last 15x1m = 15m of price changes
            data = self.get_historical_klines(symbol, interval=Client.KLINE_INTERVAL_1MINUTE)
            rsi = self.calculate_rsi(data, window = 15)
            if rsi.iloc[-1] > 70:  # RSI > 70 indicates overbought condition
                potential_drops.append(symbol)
        return potential_drops

    def get_price_precision(self, filters):
        for f in filters:
            if f['filterType'] == 'PRICE_FILTER':
                tick_size = float(f['tickSize'])
                precision = math.log10(tick_size)
                if precision < 0:
                    return -precision
        return 0

    # Adjust price to fit within the filters
    def adjust_price(self, price:float, filters, round_up:bool=False):
        for f in filters:
            if f['filterType'] == 'PRICE_FILTER':
                min_price = float(f['minPrice'])
                max_price = float(f['maxPrice'])
                tick_size = float(f['tickSize'])

                # Adjust price to be within min and max price
                price = max(min_price, min(price, max_price))

                # Round up/down to the nearest tick size
                price = price - (price % tick_size)
                if round_up:
                    price += tick_size
        return price

    # Adjust quantity to fit within the filters
    def adjust_quantity(self, quantity, filters, round_up:bool=False):
        for f in filters:
            if f['filterType'] == 'LOT_SIZE':
                min_qty = float(f['minQty'])
                max_qty = float(f['maxQty'])
                step_size = float(f['stepSize'])

                # Adjust quantity to be within min and max quantity
                quantity = max(min_qty, min(quantity, max_qty))

                # Round up/down to the nearest step size
                quantity = quantity - (quantity % step_size)
                if round_up:
                    quantity += step_size
        return quantity

    def all_symbols(self):
        symbols = self.binance.get_margin_all_pairs()
        symbols = [x for x in symbols if x['isMarginTrade'] and x['isSellAllowed'] and x['isBuyAllowed']]
        symbol_names = [x['symbol'] for x in symbols]
        symbol_names = [x for x in symbol_names if len(x) >= 4 and x[-4:] == 'USDT']
        return symbol_names

    def get_symbol_price(self, symbol:str):
        return self.binance.get_symbol_ticker(symbol=symbol).get('price')
    
    def get_price_str(self, price:float, precision:int):
        return "{:0.0{}f}".format(price, int(precision))

    def symbol_info(self, symbol:str):
        return self.binance.get_symbol_info(symbol=symbol)

    def get_symbol_filters(self, symbol:str):
        symbol_info = self.symbol_info(symbol)
        return symbol_info.get('filters')

    def repay_debt(self, symbol:str, qty: float):
        if qty > 0:
            filters = self.get_symbol_filters(symbol=symbol)
            adjusted_qty = self.adjust_quantity(quantity=qty, filters=filters, round_up=True)
            try:
                repay_order = self.binance.create_margin_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    sideEffectType=AUTO_REPAY_TYPE,
                    quantity=adjusted_qty)
                self.complete_order(symbol=symbol, order_id=repay_order.get("orderId"))
            except Exception as e:
                if 'NOTIONAL' in str(e):
                    logging.warning("Repay failed for {symbol} failed with error {error}".format(symbol=symbol, error=e))
                else:
                    raise e

    def sell_position(self, symbol:str, qty: float):
        if qty > 0:
            filters = self.get_symbol_filters(symbol=symbol)
            adjusted_qty = self.adjust_quantity(quantity=qty, filters=filters)
            try:
                close_order = self.binance.create_margin_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=adjusted_qty)
                self.complete_order(symbol=symbol, order_id=close_order.get("orderId"))
            except Exception as e:
                if 'NOTIONAL' in str(e):
                    logging.warning("Sell position for {symbol} failed with error {error}".format(symbol=symbol, error=e))
                else:
                    raise e

    # not used yet
    def try_dust_sell(self, ticker:str):
        try:
            self.binance.transfer_dust(asset=ticker, accountType='MARGIN')
        except Exception as e:
                logging.warning("Dust sell for {ticker} failed with error {error}".format(ticker=ticker, error=e))
    
    def close_positions(self):
        try:
            for attempt in range(self.config.MAX_RETRIES):
                if attempt > 0:
                    time.sleep(self.config.POLL_INTERVAL_SECONDS)

                open_symbol = self.state.open_symbol    
                if open_symbol is not None:
                    open_orders = self.binance.get_open_margin_orders(symbol=open_symbol)
                    for order in open_orders:
                        self.binance.cancel_margin_order(
                            symbol=self.state.open_symbol,
                            orderId=order.get('orderId'))

            for attempt in range(self.config.MAX_RETRIES):
                if attempt > 0:
                    time.sleep(self.config.POLL_INTERVAL_SECONDS)
                assets = self.binance.get_margin_account().get('userAssets')
                for asset in assets:
                    ticker = asset.get('asset')

                    if ticker == self.config.BASE_CURRENCY:
                        continue

                    symbol = ticker + self.config.BASE_CURRENCY
                    debt_qty = float(asset.get('borrowed')) + float(asset.get('interest'))
                    free_qty = float(asset.get('free'))
                    self.repay_debt(symbol=symbol, qty=debt_qty)
                    self.sell_position(symbol=symbol, qty=free_qty)
            
            self.state.open_symbol = None
        except Exception as e:
            logging.error("Positions closing failed, need manual action. Error: {error}".format(error=e))
            raise e
    
    def get_best_trade(self) -> Trade | None:
        return self.__get_best_trade_v1()

    def complete_order(self, symbol, order_id):
        for attempt in range(self.config.MAX_RETRIES):
            if attempt > 0:
                time.sleep(seconds=self.config.POLL_INTERVAL_SECONDS)
            order_status = self.binance.get_margin_order(
                symbol=symbol, 
                orderId=order_id).get('status')
            if order_status == ORDER_STATUS_FILLED:
                return        
        raise Exception("Order %s failed.", order_id)
        
    
    def order_long(self, trade: Trade):
        if trade.side != SIDE_BUY:
            raise Exception("Must be a BUY trade for LONG order.")
        
        '''
        There are a few caveats for BUY orders:
          - If you place order for X amount, it buys X - trading fee amout,
          so repay amount will be X - trading fee rounded to min tradeable precision
          - Anything else?
          - Will implement later
        '''
        
        raise Exception("Unimplemented.")
    
    def order_short(self, trade: Trade):
        if trade.side != SIDE_SELL:
            raise Exception("Must be a SELL trade for SHORT order.")
        
        try:
            '''
            Get into the SHORT SELL position with a limit order on ENTRY price
            '''
            order = self.binance.create_margin_order(
                symbol=trade.symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                sideEffectType=MARGIN_BUY_TYPE,
                quantity=trade.quantity,
                price=self.get_price_str(trade.in_price, trade.price_precision))
            self.complete_order(symbol=trade.symbol, order_id=order.get("orderId"))
            
            '''
            Place a REPAY BUY limit order at minimum acceptable PROFIT price
            '''
            self.binance.create_margin_order(
                symbol=trade.symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                sideEffectType=AUTO_REPAY_TYPE,
                quantity=trade.quantity,
                price=self.get_price_str(trade.profit_price, trade.price_precision))
        
            '''
            Place a REPAY BUY stop-loss limit order at maximum acceptable LOSS price
            '''
            stop_loss_price = trade.loss_price * (1 - self.config.LIMIT_ORDER_SPREAD)
            self.binance.create_margin_order(
                symbol=trade.symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_STOP_LOSS_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                sideEffectType=AUTO_REPAY_TYPE,
                quantity=trade.quantity,
                price=self.get_price_str(trade.loss_price, trade.price_precision),
                stopPrice=self.get_price_str(stop_loss_price, trade.price_precision))
        
            logging.info("Short order placed: {trade}.".format(trade=trade))
        
        except Exception as e:
            logging.error(e)
            self.close_positions()

    def trade(self, trade: Trade):
        self.state.open_symbol = trade.symbol
        self.state.trade_count += 1

        if self.state.trade_count > self.config.MAX_TRADE_COUNT:
            logging.info("Trading cycle finished, closing the bot.")
            self.close_positions()
            self.scheduler.shutdown(wait=False)
        
        logging.info(trade)
        print(trade)

        if trade.side == SIDE_BUY:
            self.order_long(trade)
        elif trade.side == SIDE_SELL:
            self.order_short(trade)
        else:
            raise Exception("Trade side should be BUY/SELL, but given " + trade.side)

    def step(self):
        self.close_positions()
        best_trade = self.get_best_trade()
        if best_trade is not None:
            self.trade(trade=best_trade)

    def run(self):
        try:
            self.scheduler.add_job(self.step, 'interval', minutes=self.config.TRADING_FREQUENCY_MINUTES)
            self.scheduler.start()
        except Exception as e:
            self.close_positions()
            self.scheduler.shutdown()

def main():
    config = Config()
    binance = Client(
        config.BINANCE_API_KEY,
        config.BINANCE_API_SECRET_KEY,
    )
    bot = Bot(config, binance)
    #bot.run()
    print(bot.all_symbols())

main()