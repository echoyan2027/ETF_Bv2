import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from scipy.stats import linregress
import io
import datetime

# --- 0. 最佳均线参数配置 (字典) ---
BEST_MA_PARAMS = {
    "有色": (13, 45), "现金流": (15, 60), "游戏": (20, 60), "医疗": (10, 30),
    "恒生科技": (15, 45), "豆粕": (10, 30), "黄金": (20, 60), "证券": (10, 30),
    "军工": (10, 30), "化工": (13, 30), "纳斯达克": (15, 45), "半导体": (10, 30),
    "光伏": (15, 60), "机器人": (13, 30), "云计算": (10, 30)
}
DEFAULT_MA = (20, 60)

st.set_page_config(page_title="ETF三态全天候轮动平台", layout="wide", page_icon="📈")

# --- 侧边栏：数据上传区 ---
st.sidebar.header("📂 1. 核心数据导入 (进攻池)")
sector_files = st.sidebar.file_uploader("上传行业/个股 (CSV/XLSX)", type=['csv', 'xlsx'], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 2. 基准数据导入 (大盘状态判定)")
st.sidebar.info("💡 提示：为了计算 RSRS，基准文件必须包含 [最高价] 和 [最低价] 列。")
benchmark_file = st.sidebar.file_uploader("上传大盘/沪深300 (CSV/XLSX)", type=['csv', 'xlsx'],
                                          accept_multiple_files=False)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ 3. 避险资产导入 (左侧下跌防守池)")
safe_files = st.sidebar.file_uploader("上传防御资产 (CSV/XLSX)", type=['csv', 'xlsx'], accept_multiple_files=True)

# --- 侧边栏：参数区 ---
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ 策略核心参数设置 (已应用最新默认值)", expanded=True):
    st.sidebar.markdown("### 📅 回测时间设置")
    start_date = st.sidebar.date_input("回测开始日期", datetime.date(2016, 1, 1),
                                       help="2016年之前的数据将被用作指标的计算预热期")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 基础仓位配置")
    rebalance_freq_label = st.sidebar.selectbox("调仓频率",
                                                ["一月一调 (Month End)", "两周一调 (2-Weeks)", "一周一调 (Weekly)"],
                                                index=0)
    freq_map = {"一月一调 (Month End)": "M", "两周一调 (2-Weeks)": "2W-FRI", "一周一调 (Weekly)": "W-FRI"}
    rebalance_freq = freq_map[rebalance_freq_label]

    top_n = st.sidebar.number_input("持仓数量", min_value=1, value=2)
    weighting_method = st.sidebar.radio("资金分配模式", ["等权分配", "风险平价 (Risk Parity)"], index=0)
    initial_capital = st.sidebar.number_input("初始资金", value=100000)
    cost_rate = st.sidebar.number_input("单边交易费率", value=0.001, format="%.4f")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 RSRS 市场状态识别模型")
    rsrs_n = st.sidebar.number_input("RSRS 回归窗口 (N)", value=18)
    rsrs_m = st.sidebar.number_input("RSRS 标准化窗口 (M)", value=600)
    rsrs_upper = st.sidebar.number_input("上涨阈值 (Z-Score > X)", value=0.7, step=0.1)
    rsrs_lower = st.sidebar.number_input("下跌阈值 (Z-Score < Y)", value=-0.7, step=0.1)

    st.sidebar.markdown("### 🚨 极值强制干预机制")
    rsrs_overheat = st.sidebar.number_input("🔥 过热阈值 (Z > X 强制防守)", value=2.0, step=0.1)
    rsrs_panic = st.sidebar.number_input("🩸 恐慌阈值 (Z < Y 强制抄底)", value=-2.0, step=0.1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧮 右侧(上涨) 动量算法")
    momentum_type = st.sidebar.radio("动量算法", ["普通涨幅", "混合动量", "回归动量 (Slope * R²)"], index=2)

    if momentum_type == "回归动量 (Slope * R²)":
        lookback_days = st.sidebar.number_input("动量计算窗口", value=20)
        r2_threshold = st.sidebar.slider("基准 R² 硬阈值", min_value=0.0, max_value=1.0, value=0.70, step=0.05)
        use_dynamic_r2 = st.sidebar.checkbox("🌟 启用波动率自适应 R² 阈值", value=True)
        r2_sensitivity = 0.2 if use_dynamic_r2 else 0.0
    elif momentum_type == "普通涨幅":
        lookback_days = st.sidebar.number_input("动量计算窗口", value=20)
        r2_threshold, use_dynamic_r2, r2_sensitivity = 0, False, 0.0
    else:
        lookback_days, r2_threshold, use_dynamic_r2, r2_sensitivity = 20, 0, False, 0.0

    use_best_ma_filter = st.sidebar.checkbox("✅ 启用最佳均线过滤 (防假突破)", value=True,
                                             help="动量进攻时，强制要求标的处于专属的双均线多头排列才允许买入")

    st.sidebar.markdown("### 🪃 震荡市(横盘) 反转算法优化")
    reversal_window = st.sidebar.number_input("乖离率(Bias)抄底窗口", value=30)
    use_reversal_opt = st.sidebar.checkbox("🛡️ 开启抄底胜率优化 (防接飞刀)", value=True)
    if use_reversal_opt:
        rev_ma_long = st.sidebar.number_input("条件1: 长期趋势保护线 (跌破不抄底)", value=250)
        rev_rsi_th = st.sidebar.number_input("条件2: RSI超卖确认阈值 (<X才抄底)", value=40)
    else:
        rev_ma_long, rev_rsi_th = 250, 40


# --- 核心数据加载 ---
@st.cache_data
def load_and_clean_data(files):
    data_dict = {}
    if not isinstance(files, list): files = [files]

    for file in files:
        if file is None: continue
        try:
            df = None
            file_name = file.name.lower()

            if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                try:
                    temp_df = pd.read_excel(file, nrows=20, header=None)
                    header_row = 0
                    for i, row in temp_df.iterrows():
                        row_str = " ".join(row.astype(str).values)
                        if ("时间" in row_str or "Date" in row_str) and ("收盘" in row_str or "Close" in row_str):
                            header_row = i;
                            break
                    file.seek(0)
                    df = pd.read_excel(file, header=header_row, dtype=str)
                except Exception as e:
                    continue
            else:
                raw_bytes = file.getvalue()
                content = None
                for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb18030']:
                    try:
                        text = raw_bytes.decode(enc)
                        if "时间" in text or "Date" in text or "收盘" in text or "Close" in text:
                            content = text;
                            break
                    except:
                        continue
                if content is None: continue
                content = content.replace('\r\n', '\n').replace('\r', '\n')
                lines = content.split('\n')
                header_row = 0
                for i, line in enumerate(lines[:20]):
                    if ("时间" in line or "Date" in line) and ("收盘" in line or "Close" in line):
                        header_row = i;
                        break
                df = pd.read_csv(io.StringIO(content), header=header_row, dtype=str)

            if df is not None:
                cols = df.columns.tolist()
                date_col = next((c for c in cols if any(k in str(c) for k in ['时间', 'Date', '日期'])), None)
                close_col = next((c for c in cols if any(k in str(c) for k in ['收盘', 'Close', '最新', 'price'])),
                                 None)
                high_col = next((c for c in cols if any(k in str(c) for k in ['最高', 'High'])), None)
                low_col = next((c for c in cols if any(k in str(c) for k in ['最低', 'Low'])), None)
                name_col = next((c for c in cols if any(k in str(c) for k in ['简称', 'Name', '名称'])), None)

                if date_col and close_col:
                    rename_dict = {date_col: 'Date', close_col: 'Close'}
                    if name_col: rename_dict[name_col] = 'Name'
                    if high_col: rename_dict[high_col] = 'High'
                    if low_col: rename_dict[low_col] = 'Low'
                    df = df.rename(columns=rename_dict)

                    for col in ['Close', 'High', 'Low']:
                        if col in df.columns:
                            df[col] = df[col].str.replace(',', '').str.replace('"', '').str.replace("'", "")
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    df = df.dropna(subset=['Close', 'Date'])
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.dropna(subset=['Date']).sort_values('Date').set_index('Date')
                    df = df[~df.index.duplicated(keep='last')]

                    name = str(df['Name'].iloc[0]).strip() if name_col and not df['Name'].dropna().empty else \
                    file.name.split('.')[0]

                    keep_cols = ['Close']
                    if 'High' in df.columns and 'Low' in df.columns:
                        keep_cols.extend(['High', 'Low'])
                    data_dict[name] = df[keep_cols]
        except:
            pass
    return data_dict


def get_best_params(name):
    for key, params in BEST_MA_PARAMS.items():
        if key in name: return params
    return DEFAULT_MA


def calc_rsrs_zscore(benchmark_df, n=18, m=600):
    if 'High' not in benchmark_df.columns or 'Low' not in benchmark_df.columns:
        return None
    high = benchmark_df['High']
    low = benchmark_df['Low']
    cov = high.rolling(n).cov(low)
    var = low.rolling(n).var()
    beta = cov / var
    beta_mean = beta.rolling(m, min_periods=250).mean()
    beta_std = beta.rolling(m, min_periods=250).std()
    z_score = (beta - beta_mean) / beta_std
    return z_score.fillna(0)


def calc_slope_r2(y):
    if np.isnan(y).any() or (y <= 0).any(): return -100.0
    y_log = np.log(y)
    x = np.arange(len(y_log))
    slope, intercept, r_value, p_value, std_err = linregress(x, y_log)
    return slope * (r_value ** 2)


def calculate_momentum_score(df_close, config):
    m_type = config['momentum_type']
    if m_type == "普通涨幅":
        return df_close.pct_change(config['lookback']).fillna(-10000)
    elif m_type == "混合动量":
        return (0.4 * df_close.pct_change(20) + 0.3 * df_close.pct_change(60) + 0.3 * df_close.pct_change(120)).fillna(
            -10000)
    elif m_type == "回归动量 (Slope * R²)":
        score_base = df_close.rolling(config['lookback']).apply(calc_slope_r2, raw=True).fillna(-10000)
        if config['use_dynamic_r2']:
            vol = df_close.pct_change().rolling(60).std() * np.sqrt(252)
            vol_ratio = vol.div(vol.median(axis=1), axis=0) - 1
            dynamic_r2_th = (config['r2_threshold'] + (vol_ratio * config['r2_sensitivity'])).clip(lower=0.3,
                                                                                                   upper=0.95)
            r2_simple = df_close.pct_change().rolling(config['lookback']).corr(
                pd.Series(np.arange(len(df_close)), index=df_close.index)) ** 2
            score_base = score_base.mask(r2_simple < dynamic_r2_th, -10000)
        return score_base
    return df_close.pct_change(config['lookback']).fillna(-10000)


def apply_specific_ma_filter(score_df, price_df):
    filtered = score_df.copy()
    for col in price_df.columns:
        s_w, l_w = get_best_params(col)
        ma_short = price_df[col].rolling(s_w).mean()
        ma_long = price_df[col].rolling(l_w).mean()
        mask = (ma_short > ma_long) & (ma_short.notna()) & (ma_long.notna())
        filtered.loc[~mask, col] = -10000
    return filtered


def calculate_reversal_score(df_close, config, regime_df):
    window = config['reversal_window']
    ma = df_close.rolling(window).mean()
    bias = (df_close - ma) / ma
    score = -bias

    if config['use_reversal_opt']:
        ma_long = df_close.rolling(config['rev_ma_long']).mean()
        mask_downtrend = df_close < ma_long

        is_panic = regime_df['Panic'] == 1
        is_panic_df = pd.DataFrame(np.tile(is_panic.values, (len(df_close.columns), 1)).T, index=df_close.index,
                                   columns=df_close.columns)
        mask_downtrend = mask_downtrend & (~is_panic_df)

        ma_5 = df_close.rolling(5).mean()
        mask_falling = df_close < ma_5

        delta = df_close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        mask_not_oversold = rsi > config['rev_rsi_th']

        invalid = mask_downtrend | mask_falling | mask_not_oversold
        score = score.mask(invalid, -10000)

    return score.fillna(-10000)


def run_strategy(sector_dict, benchmark_df, safe_dict, config):
    if not sector_dict: return None, None, None, None, None

    all_dicts = {**sector_dict}
    risk_cols = list(sector_dict.keys())
    safe_cols = list(safe_dict.keys()) if safe_dict else []
    if safe_dict: all_dicts.update(safe_dict)

    df_close = pd.DataFrame({name: df['Close'] for name, df in all_dicts.items()}).ffill().dropna(how='all')
    if df_close.empty: return None, None, None, None, None

    regime_df = pd.DataFrame(index=df_close.index)
    regime_df['Bull'] = 1
    regime_df['Bear'] = 0
    regime_df['Osc'] = 0
    regime_df['Overheat'] = 0
    regime_df['Panic'] = 0
    regime_df['RSRS_Z'] = 0.0

    if benchmark_df is not None:
        bench_close = benchmark_df['Close'].reindex(df_close.index).ffill()
        rsrs_z = calc_rsrs_zscore(benchmark_df, config['rsrs_n'], config['rsrs_m'])

        if rsrs_z is not None:
            rsrs_z = rsrs_z.reindex(df_close.index).ffill()
            regime_df['RSRS_Z'] = rsrs_z

            is_overheat = rsrs_z >= config['rsrs_overheat']
            is_panic = rsrs_z <= config['rsrs_panic']
            is_bull = (rsrs_z > config['rsrs_upper']) & (~is_overheat)
            is_bear = (rsrs_z < config['rsrs_lower']) & (~is_panic)

            is_bear = is_bear | is_overheat
            is_osc = (~is_bull) & (~is_bear)

            regime_df['Bull'] = is_bull.astype(int)
            regime_df['Bear'] = is_bear.astype(int)
            regime_df['Osc'] = is_osc.astype(int)
            regime_df['Overheat'] = is_overheat.astype(int)
            regime_df['Panic'] = is_panic.astype(int)

    risk_close = df_close[risk_cols]
    score_momentum = calculate_momentum_score(risk_close, config)

    if config['use_best_ma_filter']:
        score_momentum = apply_specific_ma_filter(score_momentum, risk_close)

    score_reversal = calculate_reversal_score(risk_close, config, regime_df)

    final_risk_score = score_momentum.multiply(regime_df['Bull'], axis=0) + \
                       score_reversal.multiply(regime_df['Osc'], axis=0) + \
                       (score_momentum * 0 - 10000).multiply(regime_df['Bear'], axis=0)

    period_risk_score = final_risk_score.resample(config['freq']).last()
    risk_ranks = period_risk_score.rank(axis=1, ascending=False)
    risk_holdings = ((risk_ranks <= config['top_n']) & (period_risk_score > -5000)).astype(int)

    safe_holdings = pd.DataFrame(0, index=risk_holdings.index, columns=safe_cols)
    if safe_cols:
        used_slots = risk_holdings.sum(axis=1)
        empty_slots = config['top_n'] - used_slots

        daily_safe_ma = df_close[safe_cols].rolling(20).mean()
        daily_safe_score = (df_close[safe_cols] / daily_safe_ma) - 1
        safe_ma_score = daily_safe_score.resample(config['freq']).last()

        safe_ma_score[safe_ma_score <= 0] = -5000

        best_safe_asset = safe_ma_score.idxmax(axis=1)
        best_safe_max_score = safe_ma_score.max(axis=1)
        for dt in safe_holdings.index:
            if empty_slots.loc[dt] > 0 and best_safe_max_score.loc[dt] > -5000:
                target_safe = best_safe_asset.loc[dt]
                safe_holdings.loc[dt, target_safe] = empty_slots.loc[dt]

    total_holdings = pd.concat([risk_holdings, safe_holdings], axis=1).fillna(0)

    daily_signal = total_holdings.reindex(df_close.index).ffill()
    daily_pos = daily_signal.shift(1).fillna(0)

    if "风险平价" in config['weighting_method']:
        volatility = df_close.pct_change().rolling(20).std().replace(0, np.nan)
        inv_vol = 1 / volatility
        active_inv = inv_vol * daily_pos
        pos_weights = active_inv.div(active_inv.sum(axis=1), axis=0).fillna(0)
        target_weights = (inv_vol * daily_signal).div((inv_vol * daily_signal).sum(axis=1), axis=0).fillna(0)
    else:
        pos_weights = daily_pos.div(config['top_n'])
        target_weights = daily_signal.div(config['top_n'])

    daily_rets = df_close.pct_change().fillna(0)
    gross = (daily_rets * pos_weights).sum(axis=1)
    turnover = target_weights.diff().abs().sum(axis=1).shift(1).fillna(0)
    net = gross - (turnover * config['cost_rate'])

    bench_ret = benchmark_df['Close'].reindex(df_close.index).ffill().pct_change().fillna(
        0) if benchmark_df is not None else daily_rets.mean(axis=1)

    start_dt = pd.to_datetime(config['start_date'])
    valid_mask = net.index >= start_dt

    if not valid_mask.any():
        st.error(f"❌ 选定的回测开始时间 ({start_dt.strftime('%Y-%m-%d')}) 晚于数据最后日期，请重新设置。")
        return None, None, None, None, None

    net = net[valid_mask]
    bench_ret = bench_ret[valid_mask]
    daily_pos = daily_pos[valid_mask]
    final_risk_score = final_risk_score[valid_mask]
    target_weights = target_weights[valid_mask]
    regime_df = regime_df[valid_mask]

    res_df = pd.DataFrame({
        'Strategy': (1 + net).cumprod() * config['capital'],
        'Benchmark': (1 + bench_ret).cumprod() * config['capital']
    })

    return res_df, daily_pos, final_risk_score, target_weights, regime_df


def calculate_metrics(strategy, benchmark):
    days = (strategy.index[-1] - strategy.index[0]).days
    cagr = (strategy.iloc[-1] / strategy.iloc[0]) ** (365 / days) - 1 if days > 0 else 0
    d_ret = strategy.pct_change().fillna(0)
    vol = d_ret.std() * np.sqrt(252)
    sharpe = (cagr - 0.03) / vol if vol != 0 else 0
    dd = (strategy / strategy.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd != 0 else 0
    total_ret = strategy.iloc[-1] / strategy.iloc[0] - 1
    return cagr, sharpe, calmar, dd, total_ret


# --- Excel 导出辅助函数 ---
def convert_df_to_excel(df):
    """将 DataFrame 转换为 Excel 二进制流"""
    output = io.BytesIO()
    # 使用 openpyxl 引擎写入内存
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='回测净值曲线')
    return output.getvalue()


# --- 主界面逻辑 ---
st.title("🚀 ETF 三态全天候轮动实战平台 (可调回测期)")

if not sector_files:
    st.info("👋 请在左侧上传【进攻池】、【大盘基准(含最高/最低价)】和【防守池】。")
else:
    sector_dict = load_and_clean_data(sector_files)
    safe_dict = load_and_clean_data(safe_files) if safe_files else {}
    benchmark_df = None
    if benchmark_file:
        bench_dict = load_and_clean_data(benchmark_file)
        if bench_dict: benchmark_df = list(bench_dict.values())[0]

    if len(sector_dict) < 2:
        st.warning(f"⚠️ 当前仅成功加载 {len(sector_dict)} 个行业文件。请至少上传 2 个行业指数进行轮动。")
    elif benchmark_df is None:
        st.error("❌ 三态模型强依赖基准数据判定市场环境，请上传包含[Close, High, Low]的基准文件。")
    else:
        config = {
            'start_date': start_date,
            'momentum_type': momentum_type, 'lookback': lookback_days,
            'r2_threshold': r2_threshold, 'use_dynamic_r2': use_dynamic_r2, 'r2_sensitivity': r2_sensitivity,
            'use_best_ma_filter': use_best_ma_filter,
            'reversal_window': reversal_window, 'use_reversal_opt': use_reversal_opt,
            'rev_ma_long': rev_ma_long, 'rev_rsi_th': rev_rsi_th,
            'freq': rebalance_freq, 'top_n': top_n, 'capital': initial_capital, 'cost_rate': cost_rate,
            'weighting_method': weighting_method, 'rsrs_n': rsrs_n, 'rsrs_m': rsrs_m,
            'rsrs_upper': rsrs_upper, 'rsrs_lower': rsrs_lower,
            'rsrs_overheat': rsrs_overheat, 'rsrs_panic': rsrs_panic
        }

        with st.spinner('正在运行全天候引擎 (指标预热与回测截断中)...'):
            res_df, positions, scores, targets, regime_df = run_strategy(sector_dict, benchmark_df, safe_dict, config)

        if res_df is not None:
            last_date = targets.index[-1]
            last_w = targets.iloc[-1]
            buy_list = last_w[last_w > 0].sort_values(ascending=False)

            current_regime = regime_df.loc[last_date]
            if current_regime['Overheat'] == 1:
                regime_str = "🔥 极度过热 (Z>2) -> 强制防守避险"
                regime_color = "error"
            elif current_regime['Panic'] == 1:
                regime_str = "🩸 极度恐慌 (Z<-2) -> 强制抄底 (豁免长线风控)"
                regime_color = "warning"
            elif current_regime['Bull'] == 1:
                regime_str = "📈 右侧多头趋势 (执行 动量追涨策略)"
                regime_color = "success"
            elif current_regime['Bear'] == 1:
                regime_str = "📉 左侧空头趋势 (执行 防守避险策略)"
                regime_color = "error"
            else:
                regime_str = "🪃 震荡无序市 (执行 均值回归抄底策略)"
                regime_color = "warning"

            with st.container():
                st.info(f"📅 **最新信号日期**: {last_date.strftime('%Y-%m-%d')}  |  **大盘状态判读**: {regime_str}")

                if buy_list.empty:
                    st.warning("⚠️ **本周指令：全面空仓观望** (无满足条件的标的，或防守资产亦在下跌)")
                else:
                    getattr(st, regime_color if regime_color != "error" else "warning")(
                        f"✅ **本周建议持有 {len(buy_list)} 只标的**")
                    action_df = pd.DataFrame({
                        "标的名称": buy_list.index,
                        "建议仓位": [f"{w:.2%}" for w in buy_list.values],
                        "参考金额": [f"{w * initial_capital:,.0f} 元" for w in buy_list.values],
                        "底层逻辑得分": [f"{scores.loc[last_date, idx]:.4f}" if idx in scores.columns else "避险资产"
                                         for idx in buy_list.index]
                    }).reset_index(drop=True)
                    st.table(action_df)

            st.markdown("---")

            # --- 🌟 净值导出按钮置于此处 🌟 ---
            col_title, col_btn = st.columns([0.7, 0.3])
            with col_title:
                st.markdown("### 📊 策略历史业绩验证")
            with col_btn:
                excel_data = convert_df_to_excel(res_df)
                st.download_button(
                    label="📥 导出回测净值曲线 (Excel)",
                    data=excel_data,
                    file_name=f"策略回测净值_{datetime.date.today().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            cagr, sharpe, calmar, dd, total_ret = calculate_metrics(res_df['Strategy'], res_df['Benchmark'])

            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            kpi1.metric("总收益率", f"{total_ret:.2%}")
            kpi2.metric("年化收益", f"{cagr:.2%}")
            kpi3.metric("夏普比率", f"{sharpe:.2f}")
            kpi4.metric("卡玛比率", f"{calmar:.2f}")
            kpi5.metric("最大回撤", f"{dd:.2%}")

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2],
                                subplot_titles=("历史资金曲线", "RSRS 状态分级 (Z-Score)", "动态回撤"))

            fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Strategy'], name='三态全天候',
                                     line=dict(color='#E31A1C', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Benchmark'], name='基准净值',
                                     line=dict(color='#999999', dash='dash')), row=1, col=1)

            if 'RSRS_Z' in regime_df.columns:
                fig.add_trace(go.Scatter(x=regime_df.index, y=regime_df['RSRS_Z'], name='RSRS Z-Score', fill='tozeroy',
                                         line=dict(color='#17BECF', width=1)), row=2, col=1)
                fig.add_hline(y=rsrs_upper, line_dash="dot", line_color="red", row=2, col=1, annotation_text="多头线")
                fig.add_hline(y=rsrs_lower, line_dash="dot", line_color="green", row=2, col=1, annotation_text="空头线")
                fig.add_hline(y=rsrs_overheat, line_dash="solid", line_color="darkred", row=2, col=1,
                              annotation_text="极度过热")
                fig.add_hline(y=rsrs_panic, line_dash="solid", line_color="darkgreen", row=2, col=1,
                              annotation_text="极度恐慌")

            drawdown = res_df['Strategy'] / res_df['Strategy'].cummax() - 1
            fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name='回撤', fill='tozeroy',
                                     line=dict(color='#ff7f0e', width=1)), row=3, col=1)

            fig.update_layout(height=750, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            tab1, tab2 = st.tabs(["🗓️ 持仓热力图", "📅 年度回报表"])
            with tab1:
                monthly_pos = positions.resample('M').last().transpose()
                fig_hm = px.imshow(monthly_pos, labels=dict(x="日期", y="标的", color="状态"),
                                   color_continuous_scale=[[0, '#f2f2f2'], [1, '#d62728']], aspect="auto")
                st.plotly_chart(fig_hm, use_container_width=True)

            with tab2:
                s_yearly = res_df['Strategy'].resample('Y').last()
                b_yearly = res_df['Benchmark'].resample('Y').last()
                years = s_yearly.index.year
                s_rets = [s_yearly.iloc[0] / initial_capital - 1] + [s_yearly.iloc[i] / s_yearly.iloc[i - 1] - 1 for i
                                                                     in range(1, len(years))]
                b_rets = [b_yearly.iloc[0] / initial_capital - 1] + [b_yearly.iloc[i] / b_yearly.iloc[i - 1] - 1 for i
                                                                     in range(1, len(years))]
                df_yr = pd.DataFrame({'年份': years.astype(str), '策略': s_rets, '基准': b_rets})
                df_yr['超额'] = df_yr['策略'] - df_yr['基准']


                def style_color(val): return f"background-color: {'#ffcccc' if val > 0 else '#ccffcc'}"


                st.dataframe(df_yr.style.format({'策略': '{:.2%}', '基准': '{:.2%}', '超额': '{:.2%}'}).map(style_color,
                                                                                                            subset=[
                                                                                                                '策略',
                                                                                                                '超额']),
                             use_container_width=True, hide_index=True)