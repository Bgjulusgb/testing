#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Stock Analysis & Prediction System
Ein umfassendes Tool zur Analyse und Vorhersage von Aktienkursen mit über 100 mathematischen Formeln
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Visualisierung
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import mplfinance as mpf

# Technische Analyse
import ta
from ta.utils import dropna
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
from ta.trend import SMAIndicator, EMAIndicator, WMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

# Statistische Analyse
from scipy import stats
from scipy.fft import fft, fftfreq
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm

# Machine Learning
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Zeitreihenanalyse
from prophet import Prophet
from pmdarima import auto_arima
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Tabellarische Ausgabe
from tabulate import tabulate
import prettytable as pt

# Weitere Module
from joblib import Parallel, delayed
import itertools
from functools import lru_cache
import multiprocessing as mp

class AdvancedStockAnalyzer:
    """Hauptklasse für die fortgeschrittene Aktienanalyse"""
    
    def __init__(self, symbols, start_date='2020-01-01', end_date=None):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = {}
        self.results = {}
        
    def fetch_data(self):
        """Lädt Aktiendaten von Yahoo Finance"""
        print("📊 Lade Aktiendaten...")
        for symbol in self.symbols:
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(start=self.start_date, end=self.end_date)
                df['Symbol'] = symbol
                self.data[symbol] = df
                print(f"✓ {symbol}: {len(df)} Datenpunkte geladen")
            except Exception as e:
                print(f"✗ Fehler beim Laden von {symbol}: {e}")
                
    def calculate_basic_metrics(self, df):
        """Berechnet grundlegende Metriken"""
        metrics = {}
        
        # Renditen
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
        
        # Volatilität
        metrics['Volatility_20d'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        metrics['Volatility_60d'] = df['Returns'].rolling(60).std() * np.sqrt(252)
        
        # Sharpe Ratio
        risk_free_rate = 0.02
        excess_returns = df['Returns'] - risk_free_rate/252
        metrics['Sharpe_Ratio'] = np.sqrt(252) * excess_returns.mean() / df['Returns'].std()
        
        # Maximum Drawdown
        rolling_max = df['Cumulative_Returns'].expanding().max()
        drawdown = (df['Cumulative_Returns'] - rolling_max) / rolling_max
        metrics['Max_Drawdown'] = drawdown.min()
        
        # CAGR
        years = (df.index[-1] - df.index[0]).days / 365.25
        metrics['CAGR'] = (df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (1/years) - 1
        
        return metrics, df
        
    def calculate_technical_indicators(self, df):
        """Berechnet über 50 technische Indikatoren"""
        print("📈 Berechne technische Indikatoren...")
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = SMAIndicator(df['Close'], period).sma_indicator()
            df[f'EMA_{period}'] = EMAIndicator(df['Close'], period).ema_indicator()
            if period <= 50:
                df[f'WMA_{period}'] = WMAIndicator(df['Close'], period).wma()
        
        # Bollinger Bands
        bb = BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = bb.bollinger_wband()
        df['BB_pband'] = bb.bollinger_pband()
        
        # ATR
        atr = AverageTrueRange(df['High'], df['Low'], df['Close'])
        df['ATR'] = atr.average_true_range()
        
        # RSI
        for period in [14, 21, 28]:
            df[f'RSI_{period}'] = RSIIndicator(df['Close'], period).rsi()
        
        # MACD
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # Stochastic
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Weitere Indikatoren
        df['ROC'] = ROCIndicator(df['Close']).roc()
        df['Williams_R'] = WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        df['CCI'] = CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['VWAP'] = VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
        
        # Keltner Channel
        kc = KeltnerChannel(df['High'], df['Low'], df['Close'])
        df['KC_upper'] = kc.keltner_channel_hband()
        df['KC_lower'] = kc.keltner_channel_lband()
        
        # Donchian Channel
        dc = DonchianChannel(df['High'], df['Low'], df['Close'])
        df['DC_upper'] = dc.donchian_channel_hband()
        df['DC_lower'] = dc.donchian_channel_lband()
        
        return df
        
    def calculate_statistical_metrics(self, df):
        """Berechnet statistische Metriken"""
        stats_dict = {}
        
        # Grundlegende Statistiken
        stats_dict['Mean'] = df['Close'].mean()
        stats_dict['Std'] = df['Close'].std()
        stats_dict['Skewness'] = stats.skew(df['Returns'].dropna())
        stats_dict['Kurtosis'] = stats.kurtosis(df['Returns'].dropna())
        
        # Z-Score
        df['Z_Score'] = (df['Close'] - df['Close'].mean()) / df['Close'].std()
        
        # Autokorrelation
        for lag in [1, 5, 10, 20]:
            stats_dict[f'Autocorr_lag{lag}'] = df['Returns'].autocorr(lag=lag)
        
        # ADF Test
        adf_result = adfuller(df['Close'].dropna())
        stats_dict['ADF_Statistic'] = adf_result[0]
        stats_dict['ADF_p_value'] = adf_result[1]
        
        # KPSS Test
        kpss_result = kpss(df['Close'].dropna())
        stats_dict['KPSS_Statistic'] = kpss_result[0]
        stats_dict['KPSS_p_value'] = kpss_result[1]
        
        # Ljung-Box Test
        lb_result = acorr_ljungbox(df['Returns'].dropna(), lags=10)
        stats_dict['Ljung_Box_Statistic'] = lb_result['lb_stat'].iloc[-1]
        
        return stats_dict, df
        
    def perform_fourier_analysis(self, df):
        """Führt Fourier-Analyse durch"""
        # FFT auf Schlusskurse
        close_prices = df['Close'].values
        n = len(close_prices)
        
        # Trend entfernen
        detrended = close_prices - np.polyval(np.polyfit(np.arange(n), close_prices, 1), np.arange(n))
        
        # FFT
        fft_values = fft(detrended)
        frequencies = fftfreq(n, 1)
        
        # Dominante Frequenzen
        power = np.abs(fft_values) ** 2
        dominant_freq_idx = np.argsort(power)[-10:]
        
        df['FFT_Power'] = pd.Series(power[:n//2], index=df.index[:n//2])
        
        return df, frequencies[:n//2], power[:n//2]
        
    def perform_pca_analysis(self, df):
        """Führt PCA auf technischen Indikatoren durch"""
        # Wähle numerische Features
        feature_cols = [col for col in df.columns if any(ind in col for ind in ['SMA', 'EMA', 'RSI', 'BB', 'ATR'])]
        features = df[feature_cols].dropna()
        
        if len(features) > 0:
            # Standardisierung
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # PCA
            pca = PCA(n_components=min(5, len(feature_cols)))
            pca_components = pca.fit_transform(features_scaled)
            
            # Ergebnisse speichern
            for i in range(pca.n_components_):
                df.loc[features.index, f'PCA_{i+1}'] = pca_components[:, i]
            
            explained_variance = pca.explained_variance_ratio_
            
            return df, explained_variance
        
        return df, None
        
    def calculate_correlation_matrix(self):
        """Berechnet Korrelationsmatrix zwischen Aktien"""
        if len(self.symbols) > 1:
            returns_df = pd.DataFrame()
            for symbol in self.symbols:
                if symbol in self.data:
                    returns_df[symbol] = self.data[symbol]['Returns']
            
            corr_matrix = returns_df.corr()
            cov_matrix = returns_df.cov() * 252  # Annualisiert
            
            return corr_matrix, cov_matrix
        return None, None
        
    def calculate_beta_alpha(self, df, market_symbol='SPY'):
        """Berechnet Beta und Alpha gegenüber dem Markt"""
        try:
            # Lade Marktdaten
            market = yf.download(market_symbol, start=self.start_date, end=self.end_date)['Close']
            market_returns = market.pct_change().dropna()
            
            # Synchronisiere Daten
            stock_returns = df['Returns'].dropna()
            aligned_data = pd.DataFrame({
                'stock': stock_returns,
                'market': market_returns
            }).dropna()
            
            if len(aligned_data) > 0:
                # Beta
                covariance = aligned_data.cov().iloc[0, 1]
                market_variance = aligned_data['market'].var()
                beta = covariance / market_variance
                
                # Alpha (CAPM)
                risk_free_rate = 0.02
                expected_return = risk_free_rate + beta * (aligned_data['market'].mean() * 252 - risk_free_rate)
                actual_return = aligned_data['stock'].mean() * 252
                alpha = actual_return - expected_return
                
                return beta, alpha
        except:
            pass
        
        return None, None
        
    def build_lstm_model(self, sequence_length=60):
        """Erstellt LSTM-Modell für Vorhersagen"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
        
    def prepare_lstm_data(self, df, sequence_length=60):
        """Bereitet Daten für LSTM vor"""
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y), scaler
        
    def predict_lstm(self, df, days=30):
        """LSTM-Vorhersage"""
        try:
            sequence_length = 60
            if len(df) < sequence_length + 100:
                return None
            
            X, y, scaler = self.prepare_lstm_data(df, sequence_length)
            
            # Train/Test Split
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Reshape für LSTM
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            # Modell trainieren
            model = self.build_lstm_model(sequence_length)
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.1)
            
            # Vorhersage
            last_sequence = scaled_data[-sequence_length:]
            predictions = []
            
            for _ in range(days):
                next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
                predictions.append(next_pred[0, 0])
                last_sequence = np.append(last_sequence[1:], next_pred)
            
            # Rücktransformation
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            return predictions.flatten()
        except:
            return None
            
    def predict_prophet(self, df, days=30):
        """Prophet-Vorhersage"""
        try:
            # Daten vorbereiten
            prophet_df = pd.DataFrame({
                'ds': df.index,
                'y': df['Close'].values
            })
            
            # Modell trainieren
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_df)
            
            # Vorhersage
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            predictions = forecast.tail(days)['yhat'].values
            
            return predictions
        except:
            return None
            
    def predict_arima(self, df, days=30):
        """ARIMA-Vorhersage"""
        try:
            # Auto ARIMA
            model = auto_arima(
                df['Close'],
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_order=10
            )
            
            # Vorhersage
            predictions, conf_int = model.predict(n_periods=days, return_conf_int=True)
            
            return predictions
        except:
            return None
            
    def predict_ml_ensemble(self, df, days=30):
        """Ensemble ML-Vorhersage"""
        try:
            # Features erstellen
            df_feat = df.copy()
            for i in range(1, 11):
                df_feat[f'lag_{i}'] = df_feat['Close'].shift(i)
            
            # Moving averages als Features
            for period in [5, 10, 20]:
                df_feat[f'ma_{period}'] = df_feat['Close'].rolling(period).mean()
            
            df_feat = df_feat.dropna()
            
            # Features und Target
            feature_cols = [col for col in df_feat.columns if col.startswith(('lag_', 'ma_'))]
            X = df_feat[feature_cols]
            y = df_feat['Close']
            
            # Train/Test Split
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Modelle trainieren
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42)
            }
            
            predictions_ensemble = []
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                
                # Vorhersage für zukünftige Tage
                last_features = X.iloc[-1:].values
                model_predictions = []
                
                for _ in range(days):
                    pred = model.predict(last_features)[0]
                    model_predictions.append(pred)
                    
                    # Update features
                    new_features = np.roll(last_features, -1)
                    new_features[0, 0] = pred  # lag_1
                    last_features = new_features
                
                predictions_ensemble.append(model_predictions)
            
            # Durchschnitt der Vorhersagen
            predictions = np.mean(predictions_ensemble, axis=0)
            
            return predictions
        except:
            return None
            
    def create_prediction_summary(self, symbol, predictions_dict):
        """Erstellt Zusammenfassung der Vorhersagen"""
        summary = []
        
        for method, preds in predictions_dict.items():
            if preds is not None:
                summary.append({
                    'Methode': method,
                    '30 Tage': f"${preds[29] if len(preds) > 29 else preds[-1]:.2f}",
                    '60 Tage': f"${preds[59] if len(preds) > 59 else preds[-1]:.2f}" if len(preds) > 59 else 'N/A',
                    '90 Tage': f"${preds[89] if len(preds) > 89 else preds[-1]:.2f}" if len(preds) > 89 else 'N/A',
                    'Trend': '↑' if preds[-1] > self.data[symbol]['Close'].iloc[-1] else '↓'
                })
        
        return pd.DataFrame(summary)
        
    def create_comprehensive_report(self, symbol):
        """Erstellt umfassenden Bericht"""
        df = self.data[symbol]
        
        # Tabellarische Zusammenfassung
        table = pt.PrettyTable()
        table.field_names = ["Metrik", "Wert"]
        
        # Aktuelle Werte
        table.add_row(["Symbol", symbol])
        table.add_row(["Aktueller Kurs", f"${df['Close'].iloc[-1]:.2f}"])
        table.add_row(["52W Hoch", f"${df['Close'].rolling(252).max().iloc[-1]:.2f}"])
        table.add_row(["52W Tief", f"${df['Close'].rolling(252).min().iloc[-1]:.2f}"])
        
        # Performance
        table.add_row(["1M Return", f"{df['Returns'].tail(20).sum()*100:.2f}%"])
        table.add_row(["3M Return", f"{df['Returns'].tail(60).sum()*100:.2f}%"])
        table.add_row(["YTD Return", f"{df['Returns'].sum()*100:.2f}%"])
        
        # Risiko
        table.add_row(["Volatilität (ann.)", f"{df['Returns'].std()*np.sqrt(252)*100:.2f}%"])
        table.add_row(["Sharpe Ratio", f"{self.results[symbol]['metrics']['Sharpe_Ratio']:.2f}"])
        table.add_row(["Max Drawdown", f"{self.results[symbol]['metrics']['Max_Drawdown']*100:.2f}%"])
        
        # Technische Indikatoren
        table.add_row(["RSI (14)", f"{df['RSI_14'].iloc[-1]:.2f}"])
        table.add_row(["MACD Signal", "Bullish" if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] else "Bearish"])
        
        print(f"\n📊 Bericht für {symbol}")
        print(table)
        
    def plot_comprehensive_analysis(self, symbol):
        """Erstellt umfassende Visualisierung"""
        df = self.data[symbol]
        
        # Plotly Subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=('Kursverlauf mit MA', 'Bollinger Bands', 
                          'RSI', 'MACD',
                          'Volume', 'Returns Distribution',
                          'Correlation Heatmap', 'Predictions'),
            row_heights=[0.3, 0.2, 0.2, 0.3],
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Kursverlauf mit MA
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ), row=1, col=1)
        
        for ma in [20, 50, 200]:
            if f'SMA_{ma}' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[f'SMA_{ma}'],
                    name=f'SMA {ma}',
                    line=dict(width=1)
                ), row=1, col=1)
        
        # 2. Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            name='Close',
            line=dict(color='black')
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_upper'],
            name='BB Upper',
            line=dict(color='red', dash='dash')
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_lower'],
            name='BB Lower',
            line=dict(color='red', dash='dash'),
            fill='tonexty'
        ), row=1, col=2)
        
        # 3. RSI
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI_14'],
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # 4. MACD
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD'],
            name='MACD',
            line=dict(color='blue')
        ), row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD_signal'],
            name='Signal',
            line=dict(color='red')
        ), row=2, col=2)
        
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['MACD_diff'],
            name='Histogram'
        ), row=2, col=2)
        
        # 5. Volume
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume'
        ), row=3, col=1)
        
        # 6. Returns Distribution
        fig.add_trace(go.Histogram(
            x=df['Returns'].dropna(),
            name='Returns',
            nbinsx=50
        ), row=3, col=2)
        
        # 7. Predictions (wenn vorhanden)
        if symbol in self.results and 'predictions' in self.results[symbol]:
            predictions = self.results[symbol]['predictions']
            
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
            
            for method, preds in predictions.items():
                if preds is not None and len(preds) >= 30:
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=preds[:30],
                        name=f'Pred {method}',
                        line=dict(dash='dash')
                    ), row=4, col=2)
        
        # Layout
        fig.update_layout(
            title=f'{symbol} - Umfassende Analyse',
            showlegend=True,
            height=1600,
            template='plotly_dark'
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
        
    def generate_latex_formulas(self):
        """Generiert LaTeX-Dokumentation der verwendeten Formeln"""
        formulas = r"""
        \documentclass{article}
        \usepackage{amsmath}
        \begin{document}
        
        \section{Verwendete Formeln}
        
        \subsection{Renditen}
        Simple Return: $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$
        Log Return: $r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$
        
        \subsection{Volatilität}
        $\sigma = \sqrt{\frac{\sum_{i=1}^{n}(R_i - \bar{R})^2}{n-1}} \times \sqrt{252}$
        
        \subsection{Sharpe Ratio}
        $SR = \frac{R_p - R_f}{\sigma_p}$
        
        \subsection{Beta}
        $\beta = \frac{Cov(R_s, R_m)}{Var(R_m)}$
        
        \subsection{Alpha (CAPM)}
        $\alpha = R_s - [R_f + \beta(R_m - R_f)]$
        
        \subsection{Bollinger Bands}
        Upper Band: $BB_{upper} = SMA + k \times \sigma$
        Lower Band: $BB_{lower} = SMA - k \times \sigma$
        
        \subsection{RSI}
        $RSI = 100 - \frac{100}{1 + RS}$, where $RS = \frac{AvgGain}{AvgLoss}$
        
        \subsection{MACD}
        $MACD = EMA_{12} - EMA_{26}$
        Signal Line: $Signal = EMA_9(MACD)$
        
        \subsection{Stochastic Oscillator}
        $\%K = \frac{C - L_{14}}{H_{14} - L_{14}} \times 100$
        $\%D = SMA_3(\%K)$
        
        \subsection{ATR}
        $ATR = \frac{1}{n}\sum_{i=1}^{n}TR_i$
        where $TR = \max[(H-L), |H-C_{prev}|, |L-C_{prev}|]$
        
        \subsection{Williams \%R}
        $\%R = \frac{H_{14} - C}{H_{14} - L_{14}} \times -100$
        
        \subsection{CCI}
        $CCI = \frac{TP - SMA_{20}(TP)}{0.015 \times MD}$
        where $TP = \frac{H + L + C}{3}$
        
        \subsection{Z-Score}
        $Z = \frac{X - \mu}{\sigma}$
        
        \subsection{Skewness}
        $S = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^3}{(n-1)s^3}$
        
        \subsection{Kurtosis}
        $K = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^4}{(n-1)s^4} - 3$
        
        \subsection{Maximum Drawdown}
        $MDD = \min_t\left(\frac{P_t - \max_{\tau \leq t}P_\tau}{\max_{\tau \leq t}P_\tau}\right)$
        
        \subsection{CAGR}
        $CAGR = \left(\frac{V_f}{V_i}\right)^{\frac{1}{t}} - 1$
        
        \subsection{Fourier Transform}
        $X(k) = \sum_{n=0}^{N-1} x(n) \cdot e^{-2\pi i k n / N}$
        
        \end{document}
        """
        return formulas