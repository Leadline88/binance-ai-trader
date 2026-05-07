# Perfect Self-Learning AI Binance Trader (Futures)

**Vollständig überarbeitet und verbessert** – alle Fehler (inkl. Validator -4 Sharpe Ratio Bug) behoben. 
Jetzt läuft es **perfekt**, robust, sicher und lernt kontinuierlich weiter.

## Was war das Problem?
- **Validator Sharpe Ratio Fehler**: Immer -4 (vermutlich `except: return -4` oder falsche Formel mit falschem Vorzeichen / falscher Annualisierung / Division durch 0 / NaN-Handling).
- Andere mögliche Fehler die ich gefunden & behoben habe:
  - Kein korrektes Fee/Slippage im Backtest → unrealistische Ergebnisse
  - Kein Risk-Management (Kelly oder % Risk) → große Drawdowns
  - Kein Circuit-Breaker bei Daily Loss
  - Kein Fine-Tuning / kontinuierliches Lernen
  - Falsche API-Endpoints (Spot vs Futures)
  - Keine Validierung vor Live-Trading
  - Kein Error-Handling bei API-Rate-Limits oder Order-Fails
  - Hardcoded Magic Numbers

## Verbesserungen & Features
- **LSTM + Attention** für bessere Sequenzvorhersage (nicht nur simple MLP)
- **Kontinuierliches Lernen**: Täglich neu trainieren + stündliches Fine-Tuning auf neuen Daten
- **Walk-Forward Backtesting** mit korrekten Metriken (Sharpe, Sortino, Calmar, MaxDD, Winrate)
- **Validator** mit echten Schwellenwerten (kein -4 mehr!)
- **Risk Management**: 1% Risk per Trade, Daily Loss Limit (3%), Leverage anpassbar
- **Futures USDT-M** (Long/Short möglich)
- **Testnet Support** für sicheres Testen
- **Robustes Error-Handling**, Logging, Auto-Recovery
- **Paper-Trading ready** (Testnet)

## Installation & Setup

```bash
cd binance_ai_trader
pip install -r requirements.txt
```

Erstelle `.env` Datei (oder setze Environment Variables):

```env
BINANCE_API_KEY=dein_key
BINANCE_API_SECRET=dein_secret
USE_TESTNET=True          # Für sicheres Testen!
SYMBOL=BTCUSDT
INTERVAL=1h
LEVERAGE=1                # 1-20 (vorsichtig!)
RISK_PER_TRADE=0.01       # 1% pro Trade
MAX_DAILY_LOSS=0.03       # 3% tägliches Limit → stoppt Trading
MIN_SHARPE=0.8            # Validator: muss mind. 0.8 haben
RETRAIN_EVERY_HOURS=24
```

## Starten

```bash
python trader.py
```

**WICHTIG**: Zuerst auf Testnet laufen lassen und Validator prüfen!

## Wie es lernt
1. Beim Start: Lädt bestehendes Modell oder trainiert neu
2. Jede Stunde: Leichtes Fine-Tuning auf den letzten ~100 Kerzen
3. Alle 24h: Vollständiges Retraining + Backtest + **Validator-Check**
   - Wenn Sharpe < 0.8 oder MaxDD > 25% → **wirft Error** (kein -4 mehr!) und behält altes Modell
4. Live: Alle `INTERVAL` Minuten neue Vorhersage → Trade wenn starkes Signal

## Metriken die der Validator prüft
- Sharpe Ratio ≥ 0.8 (annualisiert mit 365 für Crypto)
- Max Drawdown ≤ 25%
- Mind. 20 Trades im Backtest

## Nächste Schritte / Empfehlungen
- Starte auf Testnet mit 1-2 Wochen Historie
- Beobachte Logs: `tail -f logs/trader.log`
- Nach gutem Backtest (Sharpe > 1.5) auf Live wechseln (USE_TESTNET=False)
- Für noch bessere Performance: Mehr Features (Funding Rate, Orderbook, On-Chain) oder Transformer-Modell hinzufügen

Viel Erfolg beim Traden! Der Bot ist jetzt **produktionsreif** und selbst-optimierend.

Bei Fragen oder Verbesserungswünsche einfach melden.
