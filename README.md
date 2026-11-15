# Btc-Trading-AI-Bot
15mins
# Install dependencies
!pip -q install tensorflow xgboost scikit-learn pandas numpy matplotlib requests joblib
# Imports and reproducibility
import os, time, random, json, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt, requests, joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, log_loss
from xgboost import XGBClassifier
warnings.filterwarnings('ignore', category=FutureWarning)
SEED=42
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
print('TensorFlow:',tf.__version__, 'GPU:',tf.config.list_physical_devices('GPU'))
# Download the historical completed 15-minute candles
SYMBOL='BTCUSDT'; INTERVAL='15m'; DAYS=180
BASE_URL='https://api.binance.us/api/v3/klines'

def get_candles(days=DAYS):
    end_ms=int(pd.Timestamp.now(tz='UTC').timestamp()*1000)
    start_ms=end_ms-days*24*60*60*1000
    out=[]; session=requests.Session()
    while start_ms < end_ms:
        response=session.get(BASE_URL,params={'symbol':SYMBOL,'interval':INTERVAL,'startTime':start_ms,'endTime':end_ms,'limit':1000},timeout=30)
        response.raise_for_status()
        batch=response.json()
        if not batch:break
        out.extend(batch)
        nxt=int(batch[-1][0])+15*60*1000
        if nxt<=start_ms:break
        start_ms=nxt
        time.sleep(.12)
    if not out:raise RuntimeError('No candles returned. Check API availability.')
    cols=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base','taker_quote','ignore']
    df=pd.DataFrame(out,columns=cols)
    df['timestamp']=pd.to_datetime(df['open_time'],unit='ms',utc=True)
    for c in ['open','high','low','close','volume']:df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.set_index('timestamp')[['open','high','low','close','volume']].sort_index()
    df=df[~df.index.duplicated(keep='last')]

    # Filter to completed bars only
    df=df[df.index+pd.Timedelta(minutes=15)<=pd.Timestamp.now(tz='UTC')]

    # Reindex to a complete 15-minute grid to resolve gaps automatically
    full_idx=pd.date_range(start=df.index.min(), end=df.index.max(), freq='15min', tz='UTC')
    missing_before = len(full_idx) - len(df)
    if missing_before > 0:
        print(f'Warning: Found {missing_before} missing candle(s). Reindexing and filling gaps...')
        df=df.reindex(full_idx)
        # Forward-fill prices and volume, then backward-fill if any are left at the start
        df=df.ffill().bfill()

    return df

raw=get_candles()
print(raw.shape,raw.index.min(),raw.index.max())
display(raw.tail())

# Warning: Found 36 missing candle(s). Reindexing and filling gaps...
# (17279, 5) 2026-05-11 16:45:00+00:00 2026-11-07 16:15:00+00:00
# open	high	low	close	volume
# 2026-11-07 15:15:00+00:00	75882.06	75887.36	75850.01	75886.49	0.00230
# 2026-11-07 15:30:00+00:00	75869.51	75869.51	75766.49	75810.91	0.14407
# 2026-11-07 15:45:00+00:00	75809.88	75809.88	75602.93	75740.00	0.18477
# 2026-11-07 16:00:00+00:00	75758.40	75789.48	75746.36	75783.91	0.03150
# 2026-11-07 16:15:00+00:00	75783.90	75802.82	75765.36	75794.13	0.01730

# Causal technical features and next-candle target
FEE_BPS=10.0 # per side, illustrative
SLIPPAGE_BPS=3.0 # per side, illustrative
ROUND_TRIP_COST=(2*(FEE_BPS+SLIPPAGE_BPS))/10000
LABEL_THRESHOLD=max(ROUND_TRIP_COST,0.0015)
def rsi(close,n=14):
    delta=close.diff(); up=delta.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    down=(-delta.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+up/(down+1e-12))
def make_features(df):
    x=df.copy()
    x['ret1']=np.log(x.close).diff()
    for lag in [2,4,8,16]:x[f'ret{lag}']=np.log(x.close/x.close.shift(lag))
    x['range']=(x.high-x.low)/x.close
    x['body']=(x.close-x.open)/x.open
    x['vol_log']=np.log1p(x.volume)
    x['vol_z']=(x['vol_log']-x['vol_log'].rolling(48).mean())/(x['vol_log'].rolling(48).std()+1e-9)
    x['rsi']=rsi(x.close)/100
    ema12=x.close.ewm(span=12,adjust=False).mean()
    ema26=x.close.ewm(span=26,adjust=False).mean()
    x['macd']=(ema12-ema26)/x.close
    x['macd_signal']=(ema12-ema26).ewm(span=9,adjust=False).mean()/x.close
    for n in [12,48]:x[f'ema_dist_{n}']=(x.close/x.close.ewm(span=n,adjust=False).mean())-1
    tr=pd.concat([x.high-x.low,(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1)
    x['atr']=tr.ewm(alpha=1/14,adjust=False).mean()
    x['atr_pct']=x.atr/x.close
    x['volatility']=x.ret1.rolling(48).std()
    minute=x.index.hour*60+x.index.minute
    x['time_sin']=np.sin(2*np.pi*minute/1440)
    x['time_cos']=np.cos(2*np.pi*minute/1440)
    x['future_ret']=x.close.shift(-1)/x.close-1
    x['target']=np.select([x.future_ret < -LABEL_THRESHOLD,x.future_ret > LABEL_THRESHOLD],[0,2],default=1)
    return x
full=make_features(raw)
FEATURES=['ret1','ret2','ret4','ret8','ret16','range','body','vol_z','rsi','macd','macd_signal','ema_dist_12','ema_dist_48','atr_pct','volatility','time_sin','time_cos']
# Drop warmup and unlabeled last bar (future close unknown)
data=full.dropna(subset=FEATURES+['future_ret','atr']).copy()
print('Label counts:',data.target.value_counts().sort_index().to_dict(), 'Threshold:',LABEL_THRESHOLD)
assert len(data)>3000
# Label counts: {0: 1093, 1: 15015, 2: 1122} Threshold: 0.0026
# Time-ordered folds and leak-free sequence scaling
LOOKBACK=48; N_CLASSES=3
# two expanding chronological folds: train [0,60%], validate [60%,70%], test [70%,80%]; then train [0,70%], validate [70%,80%], test [80%,100%].
N=len(data)
FOLDS=[(int(.60*N),int(.70*N),int(.80*N)),(int(.70*N),int(.80*N),N)]
def fold_arrays(train_end,val_end,test_end):
    scaler=StandardScaler().fit(data[FEATURES].iloc[:train_end-1])
    arr=scaler.transform(data[FEATURES]).astype('float32')
    labels=data.target.to_numpy(dtype='int32')
    def build(start,end):
        ids=np.arange(max(start,LOOKBACK-1),end)
        X=np.stack([arr[i-LOOKBACK+1:i+1] for i in ids])
        return X,labels[ids],ids
    Xtr,ytr,itr=build(LOOKBACK-1,train_end-1)
    Xva,yva,iva=build(train_end,val_end-1)
    Xte,yte,ite=build(val_end,test_end)
    return scaler,(Xtr,ytr,itr),(Xva,yva,iva),(Xte,yte,ite)
print('Fold boundaries:',FOLDS)
# Fold boundaries: [(10338, 12061, 13784), (12061, 13784, 17230)]
# Define LSTM and Transformer classifiers
def compile_model(m):
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return m
def build_lstm():
    inp=keras.Input(shape=(LOOKBACK,len(FEATURES)))
    x=layers.LSTM(48,return_sequences=True)(inp)
    x=layers.LSTM(24)(x)
    x=layers.Dropout(.2)(x)
    out=layers.Dense(3,activation='softmax')(x)
    return compile_model(keras.Model(inp,out))
def build_transformer():
    inp=keras.Input(shape=(LOOKBACK,len(FEATURES)))
    x=layers.Dense(48)(inp)
    # Learned position embeddings to retain temporal order.
    positions=tf.range(start=0,limit=LOOKBACK,delta=1)
    x=x+layers.Embedding(input_dim=LOOKBACK,output_dim=48)(positions)
    for _ in range(2):
        z=layers.LayerNormalization()(x)
        a=layers.MultiHeadAttention(num_heads=4,key_dim=12,dropout=.1)(z,z)
        x=x+layers.Dropout(.1)(a)
        z=layers.LayerNormalization()(x)
        z=layers.Dense(96,activation='relu')(z)
        z=layers.Dense(48)(z)
        x=x+layers.Dropout(.1)(z)
    x=layers.LayerNormalization()(x)
    x=layers.GlobalAveragePooling1D()(x)
    out=layers.Dense(3,activation='softmax')(x)
    return compile_model(keras.Model(inp,out))
# ensure every class is represented, even in sparse class windows
def fit_xgb(Xtr,ytr):
    clf=XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,
                      objective='multi:softprob',num_class=3,eval_metric='mlogloss',random_state=SEED,n_jobs=2)
    clf.fit(Xtr[:,-1,:],ytr)
    return clf
EPOCHS=12; BATCH_SIZE=128
# Train each model per walk-forward fold; choose ensemble weights on validation only
from itertools import product

def choose_weights(probs,y):
    best=(float('inf'),None)
    for a in np.arange(0,1.01,.25):
        for b in np.arange(0,1.01-a,.25):
            c=round(1-a-b,6)
            if c < -1e-8:continue
            p=np.clip(a*probs[0]+b*probs[1]+c*probs[2],1e-7,1)
            loss=log_loss(y,p,labels=[0,1,2])
            if loss<best[0]:best=(loss,np.array([a,b,c]))
    return best
all_predictions=[]; trained=[]
for fold_id,(tr_end,va_end,te_end) in enumerate(FOLDS,1):
    print(f'\n=== FOLD {fold_id} ===')
    scaler,tr,va,te=fold_arrays(tr_end,va_end,te_end)
    Xtr,ytr,_=tr; Xva,yva,_=va; Xte,yte,ids=te
    if set(ytr)!={0,1,2}:raise ValueError('Training fold missing a class; adjust threshold/data.')
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)]
    lstm=build_lstm()
    lstm.fit(Xtr,ytr,validation_data=(Xva,yva),epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=callbacks,verbose=1)
    transformer=build_transformer()
    transformer.fit(Xtr,ytr,validation_data=(Xva,yva),epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=callbacks,verbose=1)
    xgb=fit_xgb(Xtr,ytr)
    pv=[lstm.predict(Xva,verbose=0),transformer.predict(Xva,verbose=0),xgb.predict_proba(Xva[:,-1,:])]
    val_loss,w=choose_weights(pv,yva)
    pt=[lstm.predict(Xte,verbose=0),transformer.predict(Xte,verbose=0),xgb.predict_proba(Xte[:,-1,:])]
    ensemble=sum(w[i]*pt[i] for i in range(3))
    frame=pd.DataFrame({'p_sell':ensemble[:,0],'p_hold':ensemble[:,1],'p_buy':ensemble[:,2],
                        'truth':yte,'fold':fold_id},index=data.index[ids])
    all_predictions.append(frame)
    trained.append((scaler,lstm,transformer,xgb,w))
    print('Validation log loss:',round(val_loss,4),'weights LSTM/Transformer/XGB:',w)
pred=pd.concat(all_predictions).sort_index()
assert not pred.index.duplicated().any()
print('Out-of-sample classification report:')
print(classification_report(pred.truth,pred[['p_sell','p_hold','p_buy']].to_numpy().argmax(axis=1),labels=[0,1,2],target_names=['SELL','HOLD','BUY'],zero_division=0))
ConfusionMatrixDisplay.from_predictions(pred.truth,pred[['p_sell','p_hold','p_buy']].to_numpy().argmax(axis=1),display_labels=['SELL','HOLD','BUY']);plt.show()
# Convert probabilities into trade decisions
MIN_CONF=0.50
p=pred[['p_sell','p_hold','p_buy']].to_numpy()
pred['signal']=np.where((p[:,2]>=MIN_CONF)&(p[:,2]>p[:,0])&(p[:,2]>p[:,1]),2,
                  np.where((p[:,0]>=MIN_CONF)&(p[:,0]>p[:,1])&(p[:,0]>p[:,2]),0,1))
pred['confidence']=p.max(axis=1)
print(pred.signal.map({0:'SELL',1:'HOLD',2:'BUY'}).value_counts())
display(pred.tail(10))
# Event-driven long-only spot backtest with next-open fills
INITIAL_CASH=10_000.0
RISK_PER_TRADE=.005 # 0.5% equity at stop, excluding gaps and execution costs
MAX_NOTIONAL_FRACTION=.25 # no leverage; max 25% equity invested
STOP_ATR=1.5; TAKE_ATR=3.0
FEE=FEE_BPS/10000; SLIP=SLIPPAGE_BPS/10000

def backtest(predictions, market):
    cash=INITIAL_CASH; qty=0.0; stop=np.nan; take=np.nan; pending=1
    records=[]; trades=[]; entry_time=None; entry_price=np.nan; entry_cost=0.0
    # Only use contiguous chronological candles; no trading across missing prediction periods.
    for i,(ts,row) in enumerate(predictions.iterrows()):
        bar=market.loc[ts]
        if i>0:
            prev_ts=predictions.index[i-1]
            if ts-prev_ts!=pd.Timedelta(minutes=15):
                # Unexpected gap: liquidate at current open, with costs, before applying signals.
                if qty>0:
                    fill=bar.open*(1-SLIP)
                    proceeds=qty*fill*(1-FEE)
                    cash+=proceeds
                    trades.append({'entry_time':entry_time,'exit_time':ts,'entry':entry_price,'exit':fill,'net_pnl':proceeds-entry_cost,'reason':'gap'})
                    qty=0
                pending=1
            # Execute signal from PREVIOUS close at CURRENT open.
            if qty>0 and pending==0:
                fill=bar.open*(1-SLIP); proceeds=qty*fill*(1-FEE);cash+=proceeds
                trades.append({'entry_time':entry_time,'exit_time':ts,'entry':entry_price,'exit':fill,'net_pnl':proceeds-entry_cost,'reason':'SELL signal'})
                qty=0
            if qty==0 and pending==2:
                atr=float(market.loc[prev_ts,'atr'])
                fill=bar.open*(1+SLIP)
                stop_candidate=fill-STOP_ATR*atr
                if stop_candidate>0 and atr>0:
                    equity=cash
                    risk_qty=(equity*RISK_PER_TRADE)/(STOP_ATR*atr+fill*(2*SLIP+2*FEE))
                    cash_qty=cash*MAX_NOTIONAL_FRACTION/(fill*(1+FEE))
                    buy_qty=max(0,min(risk_qty,cash_qty))
                    if buy_qty>0:
                        cost=buy_qty*fill*(1+FEE)
                        cash-=cost;qty=buy_qty;stop=stop_candidate;take=fill+TAKE_ATR*atr
                        entry_time=ts;entry_price=fill;entry_cost=cost
        # Stop/target checks for existing positions, including freshly opened ones.
        if qty>0:
            reason=None; exit_level=None
            if bar.open<=stop:reason='stop gap';exit_level=bar.open
            elif bar.open>=take:reason='target gap';exit_level=bar.open
            elif bar.low<=stop:reason='stop';exit_level=stop
            elif bar.high>=take:reason='target';exit_level=take
            if reason:
                fill=exit_level*(1-SLIP);proceeds=qty*fill*(1-FEE);cash+=proceeds
                trades.append({'entry_time':entry_time,'exit_time':ts,'entry':entry_price,'exit':fill,'net_pnl':proceeds-entry_cost,'reason':reason})
                qty=0
        # Mark to market at current candle close.
        equity=cash+qty*bar.close
        records.append((ts,equity,cash,qty))
        pending=int(row.signal)
    # Final close liquidation (cannot carry into an untested period)
    if qty>0:
        ts=predictions.index[-1];fill=market.loc[ts,'close']*(1-SLIP)
        proceeds=qty*fill*(1-FEE);cash+=proceeds
        trades.append({'entry_time':entry_time,'exit_time':ts,'entry':entry_price,'exit':fill,'net_pnl':proceeds-entry_cost,'reason':'end of test'})
        records[-1]=(ts,cash,cash,0)
    curve=pd.DataFrame(records,columns=['timestamp','equity','cash','qty']).set_index('timestamp')
    return curve,pd.DataFrame(trades)
# Backtest each test fold independently; concatenate per-fold normalized returns for an overall walk-forward series.
curves=[];trades_list=[]
for f in sorted(pred.fold.unique()):
    segment=pred[pred.fold==f]
    curve,trade=backtest(segment,full)
    curve['fold']=f;curves.append(curve)
    if len(trade):trade['fold']=f;trades_list.append(trade)
trades=pd.concat(trades_list,ignore_index=True) if trades_list else pd.DataFrame()
# Compute combined return path, resetting equity to 1 at the beginning of each fold.
returns=pd.concat([c.equity.pct_change().fillna(c.equity.iloc[0]/INITIAL_CASH-1) for c in curves]).sort_index()
combined_equity=INITIAL_CASH*(1+returns).cumprod()
print('Number of closed trades:',len(trades))
display(trades.tail() if len(trades) else pd.DataFrame({'info':['No trades under current thresholds']}))
# Number of closed trades: 1
entry_time	exit_time	entry	exit	net_pnl	reason	fold
0	2026-10-30 00:30:00+00:00	2026-10-30 01:00:00+00:00	77001.563541	77228.684442	2.364167	SELL signal	2
# Metrics and visualizations
equity=combined_equity
ret=returns
max_dd=(equity/equity.cummax()-1).min()
net_return=equity.iloc[-1]/INITIAL_CASH-1
sharpe=(np.sqrt(365*24*4)*ret.mean()/ret.std()) if ret.std()>0 else np.nan
if len(trades):
    wins=trades.loc[trades.net_pnl>0,'net_pnl'].sum()
    losses=-trades.loc[trades.net_pnl<0,'net_pnl'].sum()
    pf=wins/losses if losses>0 else np.nan
    winrate=(trades.net_pnl>0).mean()
else:pf=np.nan;winrate=np.nan
print(pd.Series({'Net return':f'{net_return:.2%}','Max drawdown':f'{max_dd:.2%}',
                 'Sharpe (illustrative)':round(sharpe,3),'Trades':len(trades),
                 'Win rate':f'{winrate:.2%}' if np.isfinite(winrate) else 'N/A',
                 'Profit factor':round(pf,3) if np.isfinite(pf) else 'N/A'}))
plt.figure(figsize=(12,4));plt.plot(equity.index,equity.values);plt.title('Walk-forward simulated equity (fees + slippage)');plt.ylabel('USDT');plt.grid(True);plt.show()
plt.figure(figsize=(12,3));plt.plot(equity.index,equity/equity.cummax()-1);plt.title('Drawdown');plt.grid(True);plt.show()
# Latest completed candle prediction (research signal)
scaler,lstm,transformer,xgb,w=trained[-1]
latest=full.dropna(subset=FEATURES).copy()
last_X=scaler.transform(latest[FEATURES].iloc[-LOOKBACK:]).astype('float32')[None,:,:]
pp=[lstm.predict(last_X,verbose=0)[0],transformer.predict(last_X,verbose=0)[0],xgb.predict_proba(last_X[:,-1,:])[0]]
prob=sum(w[i]*pp[i] for i in range(3))
signal='BUY' if prob[2]>=MIN_CONF and prob[2]>max(prob[0],prob[1]) else ('SELL' if prob[0]>=MIN_CONF and prob[0]>max(prob[1],prob[2]) else 'HOLD')
print('Completed candle UTC:',latest.index[-1]);print('Last close:',latest.close.iloc[-1]);print('Signal:',signal)
print(dict(zip(['SELL','HOLD','BUY'],np.round(prob,4))))
print('Highest model probability:',round(float(prob.max()),4),'(not necessarily calibrated confidence)')
# Completed candle UTC: 2026-11-07 16:15:00+00:00
Last close: 75794.13
Signal: HOLD
{'SELL': np.float64(0.0052), 'HOLD': np.float64(0.9913), 'BUY': np.float64(0.0036)}
Highest model probability: 0.9913 (not necessarily calibrated confidence)
