# moving_average_bot.py
# pip install hyperliquid-python-sdk pandas numpy python-dotenv eth-account

import os
import time
import math
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

# ---- CONFIG ----
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")          # clé privée API wallet (0x...)
ACCOUNT_ADDRESS = os.getenv("ACCOUNT_ADDRESS")  # public key (0x...)
BASE_URL = constants.TESTNET_API_URL          # testnet (change en MAINNET si prêt)
ASSET = "BTC"                                 # symbole (ex: 'BTC', 'ETH', 'HYPE'...)
INTERVAL = "1m"                               # '1m','5m','1h',...
TRADE_USD = 10.0                              # montant dollar par trade (ex: 10 USD)
SMA_SHORT = 9
SMA_LONG = 21
LOOP_SLEEP = 60                               # secondes entre itérations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if not SECRET_KEY or not ACCOUNT_ADDRESS:
    raise SystemExit("Place tes variables SECRET_KEY et ACCOUNT_ADDRESS dans un fichier .env")

# ---- clients ----
wallet = Account.from_key(SECRET_KEY)
info = Info(BASE_URL, skip_ws=True)
exchange = Exchange(wallet, BASE_URL, account_address=ACCOUNT_ADDRESS)

# ---- helpers ----
def now_ms():
    return int(time.time() * 1000)

def fetch_candles(info_client, coin: str, interval: str, limit: int = 500):
    """
    Récupère candles via la méthode du SDK ou en fallback via post.
    Retourne DataFrame avec colonnes: t (open ms), o,h,l,c,v
    """
    end_ms = now_ms()
    # approx pour start: pour interval minute
    int_ms = {"1m":60*1000, "5m":5*60*1000, "1h":60*60*1000}.get(interval, 60*1000)
    start_ms = end_ms - limit * int_ms

    # Le SDK propose un helper candle_snapshot ; si absent, on essaie post()
    try:
        resp = info_client.candle_snapshot(coin, interval, start_ms, end_ms)
    except Exception:
        payload = {"type": "candleSnapshot", "coin": coin, "interval": interval,
                   "startTime": start_ms, "endTime": end_ms}
        resp = info_client.post(payload)

    # resp peut être directement la liste de candles ou un dict. Normalise.
    candles = None
    if isinstance(resp, dict):
        # plusieurs formats possibles selon wrapper SDK
        if "candles" in resp:
            candles = resp["candles"]
        elif "response" in resp and isinstance(resp["response"], dict) and "candles" in resp["response"]:
            candles = resp["response"]["candles"]
        else:
            # recherche récursive raisonnable
            for v in resp.values():
                if isinstance(v, list) and len(v)>0 and isinstance(v[0], dict) and "c" in v[0]:
                    candles = v
                    break
    elif isinstance(resp, list):
        candles = resp

    if not candles:
        raise RuntimeError("Impossible d'extraire les candles, inspecte le contenu de la réponse")

    # Construire DataFrame
    df = pd.DataFrame(candles)
    # noms attendus: t,T,o,h,l,c,v  (t=open ms, T=close ms, o/h/l/c strings)
    df = df.rename(columns={"t":"open_ts","T":"close_ts","o":"open","h":"high","l":"low","c":"close","v":"volume"})
    df["open_ts"] = pd.to_datetime(df["open_ts"], unit="ms")
    for col in ("open","high","low","close","volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("open_ts").reset_index(drop=True)
    return df

def compute_sma(df, short=SMA_SHORT, long=SMA_LONG):
    df = df.copy()
    df["sma_short"] = df["close"].rolling(short).mean()
    df["sma_long"] = df["close"].rolling(long).mean()
    return df

def get_asset_meta(info_client):
    """Récupère meta() et retourne dictionnaire utile pour precision tick/lot.
       (utilisé pour arrondir taille & prix)"""
    meta = info_client.meta()  # la SDK fournit meta() ; structure = dict
    # On renvoie la structure brute : adapte selon besoin
    return meta

def round_to_price_tick(px, tick):
    """arrondir px au tick price (ex: tick = 0.01)"""
    return math.floor(px / tick) * tick

def format_size(sz, size_decimals):
    """formate la taille avec nb décimales attendu"""
    fmt = "{:0." + str(size_decimals) + "f}"
    return float(fmt.format(sz))

def place_limit_order(exchange_client, name, is_buy, size, price, tif="Ioc"):
    """
    Utilise exchange.order(name, is_buy, sz, limit_px, order_type)
    Ex : order(name, True, 0.01, 50000.0, {"limit":{"tif":"Gtc"}})
    """
    order_type = {"limit": {"tif": tif}}
    try:
        res = exchange_client.order(name, is_buy, size, price, order_type)
        logging.info("Order response: %s", res)
        return res
    except Exception as e:
        logging.exception("Erreur en plaçant l'ordre: %s", e)
        return None

# ---- backtest simple (paper) ----
def backtest_sma(df):
    df = df.copy().dropna(subset=["sma_short","sma_long"]).reset_index(drop=True)
    pos = 0
    entry_price = 0.0
    trades = []
    for i in range(1, len(df)):
        prev = df.loc[i-1]
        cur = df.loc[i]
        # cross up
        if prev["sma_short"] <= prev["sma_long"] and cur["sma_short"] > cur["sma_long"]:
            if pos == 0:
                pos = 1
                entry_price = cur["close"]
                trades.append(("BUY", cur["open_ts"], entry_price))
        # cross down
        elif prev["sma_short"] >= prev["sma_long"] and cur["sma_short"] < cur["sma_long"]:
            if pos == 1:
                exit_price = cur["close"]
                trades.append(("SELL", cur["open_ts"], exit_price))
                pos = 0
    # calcul P&L simple
    pnl = 0.0
    for i in range(0, len(trades), 2):
        if i+1 < len(trades):
            buy = trades[i]; sell = trades[i+1]
            pnl += (sell[2] - buy[2])
    logging.info("Backtest: %d trades, PnL (price differential) = %f", len(trades)//2, pnl)
    return trades, pnl

# ---- boucle principale (simplifiée) ----
def main_loop():
    position_open = False
    position_side = None
    asset_name = ASSET
    meta = get_asset_meta(info)  # util pour tick / lot sizes (structure dépendante de la réponse)
    # TODO: extraire tick et decimals depuis meta pour ce coin (ex: meta['universe'] etc.)
    price_tick = 0.01
    size_decimals = 4

    while True:
        try:
            df = fetch_candles(info, asset_name, INTERVAL, limit=200)
            df = compute_sma(df)
            if df[["sma_short","sma_long"]].isna().any().any():
                logging.info("Pas assez de chandelles pour calculer SMA")
                time.sleep(LOOP_SLEEP)
                continue

            last = df.iloc[-1]
            prev = df.iloc[-2]
            # signal detection
            cross_up = (prev["sma_short"] <= prev["sma_long"]) and (last["sma_short"] > last["sma_long"])
            cross_down = (prev["sma_short"] >= prev["sma_long"]) and (last["sma_short"] < last["sma_long"])

            ticker_price = last["close"]
            # convert TRADE_USD -> qty
            raw_qty = TRADE_USD / max(1e-8, ticker_price)
            qty = format_size(raw_qty, size_decimals)

            if cross_up and not position_open:
                px = round_to_price_tick(ticker_price * 1.002, price_tick)  # limit légèrement au-dessus
                logging.info("Signal BUY detected @ %s -- placing order QTY=%s PX=%s", last["open_ts"], qty, px)
                res = place_limit_order(exchange, asset_name, True, qty, px, tif="Ioc")
                if res:
                    position_open = True
                    position_side = "LONG"
            elif cross_down and position_open and position_side == "LONG":
                px = round_to_price_tick(ticker_price * 0.998, price_tick)
                logging.info("Signal SELL detected (close) @ %s -- placing order QTY=%s PX=%s", last["open_ts"], qty, px)
                res = place_limit_order(exchange, asset_name, False, qty, px, tif="Ioc")
                if res:
                    position_open = False
                    position_side = None
            else:
                logging.info("No signal. last close=%s", ticker_price)

        except Exception as e:
            logging.exception("Erreur dans main loop: %s", e)

        time.sleep(LOOP_SLEEP)


if __name__ == "__main__":
    # Exemple: backtest rapide
    try:
        df_bt = fetch_candles(info, ASSET, INTERVAL, limit=500)
        df_bt = compute_sma(df_bt)
        backtest_sma(df_bt)
    except Exception as e:
        logging.warning("Backtest failed: %s", e)

    # Décommenter pour lancer la boucle live (testnet)
    # main_loop()
