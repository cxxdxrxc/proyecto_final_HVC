# proyecto_final.py
# Proyecto Final – Python for Finance
# Dashboard único con todos los incisos en pestañas (acciones y criptomonedas).

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta

from dash import Dash, dcc, html, Input, Output, callback_context, exceptions
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import kurtosis, skew

# ============================================================
# 1. DATOS DE ACCIONES (Exploración + Riesgo)
# ============================================================

TICKERS = ["PG", "KO", "PEP", "CAT", "HON", "GE"]

DESC = {
    "PG":  "Procter & Gamble — Consumo masivo (hogar, cuidado personal).",
    "KO":  "Coca-Cola — Consumo masivo (bebidas no alcohólicas).",
    "PEP": "PepsiCo — Consumo masivo (snacks y bebidas).",
    "CAT": "Caterpillar — Industrial (maquinaria pesada).",
    "HON": "Honeywell — Industrial (aeroespacial, automatización).",
    "GE":  "General Electric — Industrial (energía, salud, aeroespacial).",
}

HOY_ACC = date.today()
INICIO_ACC = HOY_ACC - timedelta(days=365 * 15)  # ~15 años


def get_prices_actions(tickers=TICKERS, start=INICIO_ACC, end=HOY_ACC):
    # Descarga conjunta
    df = yf.download(
        tickers,
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=True,
        progress=False
    )

    # Si viene MultiIndex (Adj Close, Close, etc.), nos quedamos con Close o Adj Close
    if isinstance(df.columns, pd.MultiIndex):
        # usa "Adj Close" si existe, si no "Close"
        if "Adj Close" in df.columns.levels[0]:
            df = df["Adj Close"].copy()
        else:
            df = df["Close"].copy()

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna(how="all")

    # Ver qué tickers faltan en las columnas
    presentes = [c for c in df.columns if c in tickers]
    faltantes = [t for t in tickers if t not in presentes]

    # Reintentar cada faltante por separado
    for t in faltantes:
        try:
            s = yf.download(
                t,
                start=start,
                end=end + timedelta(days=1),
                auto_adjust=True,
                progress=False
            )["Close"]
            s = s.dropna()
            s.name = t
            df = pd.concat([df, s], axis=1)
        except Exception as e:
            print(f"No se pudo descargar {t}: {e}")

    # Nos quedamos solo con las columnas de los tickers originales que sí existen
    cols_finales = [t for t in tickers if t in df.columns]
    df = df[cols_finales]

    return df


# === Construcción de PRICES y RETURNS ===
PRICES = get_prices_actions()

# Convertir a numérico por seguridad
PRICES = PRICES.apply(pd.to_numeric, errors="coerce")

# Retornos diarios
RETURNS = PRICES.pct_change(fill_method=None).dropna(how="all")

# Lista real de acciones (para dropdowns)
ACCIONES_LIST = list(PRICES.columns)

# ============================================================
# 2. DATOS DE CRIPTOS (Data1 + Data2)
# ============================================================
# ========================
# 2. Cargar criptomonedas
# ========================

# Data1.csv y Data2.csv están en formato "largo" (OHLCV + ticker)
df1 = pd.read_csv("Data1.csv")
df2 = pd.read_csv("Data2.csv")

# Unirlos en un solo DataFrame
df_crypto_raw = pd.concat([df1, df2], ignore_index=True)

# Asegurarnos de que 'Date' sea datetime
df_crypto_raw["Date"] = pd.to_datetime(df_crypto_raw["Date"], errors="coerce")
df_crypto_raw = df_crypto_raw.dropna(subset=["Date"])

# Nos quedamos solo con las columnas que necesitamos
# (asegúrate de que estos nombres estén exactamente así en tus CSV)
df_crypto_raw = df_crypto_raw[["Date", "ticker", "Close"]]

# Pivoteamos: filas = fechas, columnas = ticker, valores = Close
CRYPTO_WIDE = df_crypto_raw.pivot_table(
    index="Date",
    columns="ticker",
    values="Close",
    aggfunc="last"          # por si hay duplicados en misma fecha
)

# Ordenar por fecha y convertir a numérico
CRYPTO_WIDE = CRYPTO_WIDE.sort_index()
CRYPTO_WIDE = CRYPTO_WIDE.apply(pd.to_numeric, errors="coerce")

# Retornos diarios de cripto (sin fill_method implícito)
CRYPTO_RET = CRYPTO_WIDE.pct_change(fill_method=None).dropna(how="all")

# Lista de criptos disponibles (columnas)
crypto_list = list(CRYPTO_WIDE.columns)

# Rango de fechas para los DatePicker de cripto
min_date_crypto = CRYPTO_RET.index.min()
max_date_crypto = CRYPTO_RET.index.max()

# último año como default
default_start_crypto = max_date_crypto - pd.Timedelta(days=365)
if default_start_crypto < min_date_crypto:
    default_start_crypto = min_date_crypto

# ============================================================
# 3. FUNCIONES AUXILIARES – RIESGO / BOLLINGER / VAR / CVaR
# ============================================================

def bollinger_bands(series, window=20, num_std=2.0):
    """Bandas de Bollinger de una serie de retornos."""
    ma = series.rolling(window=window, min_periods=window).mean()
    sd = series.rolling(window=window, min_periods=window).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return ma, upper, lower

def max_drawdown(series_prices: pd.Series) -> float:
    """Maximum Drawdown sobre una serie de precios."""
    roll_max = series_prices.cummax()
    drawdown = series_prices / roll_max - 1.0
    return float(drawdown.min())

def var_historico(x, alpha=0.95):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, 1 - alpha))

def cvar_historico(x, alpha=0.95):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    v = var_historico(x, alpha=alpha)
    cola = x[x <= v]
    if cola.size == 0:
        return np.nan
    return float(cola.mean())

# ============================================================
# 4. APP DASH – UNA SOLA APP CON TABS
# ============================================================

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ------------------- Layout de cada tab (inciso) -------------------

# INCISO 1 – Exploración de acciones (precios y retornos)
layout_inciso1 = html.Div([
    html.H2("Exploración de acciones: precios y retornos",
            style={"textAlign": "center"}),

    html.P("Selecciona acciones, elige si ver precios o retornos y filtra el rango de fechas.",
           style={"textAlign": "center"}),

    html.Div([
        html.Div([
            html.Label("Acciones"),
            dcc.Dropdown(
                id="acciones_sel",
                options=[{"label": t, "value": t} for t in ACCIONES_LIST],
                value=["PG", "KO", "PEP"],
                multi=True
            )
        ], style={"flex": "2"}),

        html.Div([
            html.Label("Vista"),
            dcc.RadioItems(
                id="view",
                options=[
                    {"label": "Precios", "value": "price"},
                    {"label": "Retornos", "value": "return"},
                ],
                value="price",
                labelStyle={"display": "block"}
            )
        ], style={"flex": "1", "marginLeft": "12px"}),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "10px"}),

    dcc.DatePickerRange(
        id="dates",
        start_date=INICIO_ACC,
        end_date=HOY_ACC,
        display_format="YYYY-MM-DD"
    ),

    html.Hr(),

    dcc.Graph(id="main_chart"),

    html.H3("Descripción breve de las empresas seleccionadas"),
    html.Div(id="desc_box", style={"whiteSpace": "pre-line", "lineHeight": "1.5"}),
], style={"maxWidth": "1000px", "margin": "0 auto", "padding": "10px"})


# INCISO 3a – Bandas de Bollinger para retornos de criptos
layout_inciso3a = html.Div([
    html.H2("Criptomonedas: Bandas de Bollinger de retornos",
            style={"textAlign": "center"}),

    html.P("Retornos diarios con bandas de Bollinger, banda central y detección de causas especiales.",
           style={"textAlign": "center", "color": "#555"}),

    html.Div([
        html.Div([
            html.Label("Criptomoneda"),
            dcc.Dropdown(
                id="crypto_boll_sel",
                options=[{"label": c, "value": c} for c in crypto_list],
                value=None,
                multi=False,
                placeholder="Selecciona una cripto..."
            ),
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Label("Rango de fechas"),
            dcc.DatePickerRange(
                id="date_range_boll",
                min_date_allowed=min_date_crypto,
                max_date_allowed=max_date_crypto,
                start_date=default_start_crypto,
                end_date=max_date_crypto,
                display_format="YYYY-MM-DD",
                style={"marginTop": "6px", "display": "block"}
            ),
        ], style={"marginBottom": "25px"}),

        html.Div([
            html.Div([
                html.Label("Ventana (días)"),
                dcc.Slider(
                    id="window_boll",
                    min=10, max=40, step=1,
                    value=20,
                    marks={10: "10", 20: "20", 30: "30", 40: "40"}
                )
            ], style={"flex": 1, "marginRight": "15px"}),

            html.Div([
                html.Label("Sigmas (σ)"),
                dcc.Slider(
                    id="sigma_boll",
                    min=1.0, max=3.0, step=0.5,
                    value=2.0,
                    marks={1.0: "1", 1.5: "1.5", 2.0: "2", 2.5: "2.5", 3.0: "3"}
                )
            ], style={"flex": 1})
        ], style={"display": "flex", "marginTop": "5px"}),
    ], style={"maxWidth": "750px", "margin": "0 auto"}),

    html.Hr(),
    dcc.Graph(id="graf_bollinger"),
    html.Div(
        id="texto_causas",
        style={"textAlign": "center", "marginTop": "10px",
               "fontWeight": "bold", "color": "#333"}
    )
], style={"maxWidth": "1000px", "margin": "0 auto", "padding": "15px"})


# INCISO 3b – Gráfico animado de precios cripto
layout_inciso3b = html.Div([
    html.H2("Criptomonedas: gráfico animado de precios",
            style={"textAlign": "center"}),

    html.P("Animación mensual de la evolución de los precios de las criptomonedas seleccionadas.",
           style={"textAlign": "center", "color": "#555"}),

    html.Div([
        html.Label("Criptomonedas"),
        dcc.Dropdown(
            id="crypto_sel_anim",
            options=[{"label": c, "value": c} for c in crypto_list],
            value=[],
            multi=True,
            placeholder="Select"
        ),
        html.Div([
            html.Button("Seleccionar todas", id="btn_anim_select_all", n_clicks=0),
            html.Button("Limpiar", id="btn_anim_clear", n_clicks=0,
                        style={"marginLeft": "10px"})
        ], style={"marginTop": "10px"})
    ], style={"maxWidth": "600px", "margin": "0 auto"}),

    html.Hr(),
    dcc.Graph(id="graf_anim_crypto")
], style={"maxWidth": "1000px", "margin": "0 auto", "padding": "15px"})


# INCISO 3c – Max Drawdown y CVaR 95% (criptos)
layout_inciso3c = html.Div([
    html.H2("Criptomonedas: Max Drawdown y CVaR 95%",
            style={"textAlign": "center"}),

    html.P("Indicadores de riesgo histórico (Max Drawdown y CVaR 95%) para cada criptomoneda.",
           style={"textAlign": "center", "color": "#555"}),

    dcc.Graph(id="graf_mdd"),
    dcc.Graph(id="graf_cvar")
], style={"maxWidth": "1000px", "margin": "0 auto", "padding": "15px"})


# ------------------- Layout principal con Tabs -------------------

app.layout = html.Div([
    html.H1("Proyecto Final – Python for Finance",
            style={"textAlign": "center"}),

    html.P("Exploración y análisis de riesgo de acciones y criptomonedas para un fondo de inversión.",
           style={"textAlign": "center", "marginBottom": "15px"}),

    dcc.Tabs(
        id="tabs-proyecto",
        value="inciso1",
        children=[
            dcc.Tab(label="Acciones: precios y retornos", value="inciso1"),
            dcc.Tab(label="Cripto: Bollinger retornos", value="inciso3a"),
            dcc.Tab(label="Cripto: gráfico animado", value="inciso3b"),
            dcc.Tab(label="Cripto: MDD y CVaR 95%", value="inciso3c"),
        ]
    ),

    html.Div(id="content-tabs")
])


# ============================================================
# 5. CALLBACK PARA CAMBIAR EL CONTENIDO DE LAS TABS
# ============================================================

@app.callback(
    Output("content-tabs", "children"),
    Input("tabs-proyecto", "value")
)
def render_tab_content(tab):
    if tab == "inciso1":
        return layout_inciso1
    elif tab == "inciso3a":
        return layout_inciso3a
    elif tab == "inciso3b":
        return layout_inciso3b
    elif tab == "inciso3c":
        return layout_inciso3c
    return html.Div("Tab no encontrada.")


# ============================================================
# 6. CALLBACKS DE CADA INCISO
# ============================================================

# ---------- Inciso 1 – acciones ----------

@app.callback(
    Output("main_chart", "figure"),
    Output("desc_box", "children"),
    Input("tickers", "value"),
    Input("view", "value"),
    Input("dates", "start_date"),
    Input("dates", "end_date"),
)
def update_view_acciones(tickers_sel, view, start_date, end_date):
    if not tickers_sel:
        fig = px.line(title="Select")
        return fig, "—"

    px_df = PRICES.loc[start_date:end_date, tickers_sel]
    rt_df = RETURNS.loc[start_date:end_date, tickers_sel]

    if view == "price":
        fig = px.line(px_df, title="Precios de cierre de acciones")
    else:
        fig = px.line(rt_df, title="Retornos diarios de acciones")

    fig.update_layout(
        hovermode="x unified",
        plot_bgcolor="white",
        legend_title_text="Ticker"
    )

    desc_text = "\n".join([f"{t}: {DESC.get(t, '—')}" for t in tickers_sel])
    return fig, desc_text


# ---------- Inciso 3a – Bollinger cripto ----------

@app.callback(
    Output("graf_bollinger", "figure"),
    Output("texto_causas", "children"),
    Input("crypto_boll_sel", "value"),
    Input("window_boll", "value"),
    Input("sigma_boll", "value"),
    Input("date_range_boll", "start_date"),
    Input("date_range_boll", "end_date")
)
def actualizar_bollinger(crypto, window, sigma, start_date, end_date):
    if crypto is None:
        fig = go.Figure()
        fig.update_layout(
            title="Select",
            plot_bgcolor="white"
        )
        return fig, ""

    s_ret_all = CRYPTO_RET[crypto].dropna()

    if start_date is not None:
        start = pd.to_datetime(start_date)
        s_ret_all = s_ret_all[s_ret_all.index >= start]
    if end_date is not None:
        end = pd.to_datetime(end_date)
        s_ret_all = s_ret_all[s_ret_all.index <= end]

    if s_ret_all.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No hay datos en el rango seleccionado para {crypto}",
            plot_bgcolor="white"
        )
        return fig, f"No hay datos suficientes para {crypto} en el rango seleccionado."

    ma, upper, lower = bollinger_bands(
        s_ret_all,
        window=int(window),
        num_std=float(sigma)
    )

    df_plot = pd.DataFrame({
        "Ret": s_ret_all,
        "MA": ma,
        "Upper": upper,
        "Lower": lower
    }).dropna()

    out_mask = (df_plot["Ret"] > df_plot["Upper"]) | (df_plot["Ret"] < df_plot["Lower"])
    df_out = df_plot[out_mask]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["Lower"],
        line=dict(color="rgba(120, 170, 255, 0.0)"),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["Upper"],
        line=dict(color="rgba(120, 170, 255, 0.0)"),
        fill="tonexty",
        fillcolor="rgba(120, 170, 255, 0.25)",
        name=f"Bandas Bollinger (±{sigma}σ)"
    ))
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["MA"],
        line=dict(color="black", width=2),
        name="Media móvil"
    ))
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["Ret"],
        line=dict(color="#1f77b4", width=1),
        name="Retorno diario"
    ))

    if not df_out.empty:
        fig.add_trace(go.Scatter(
            x=df_out.index,
            y=df_out["Ret"],
            mode="markers",
            marker=dict(color="red", size=6),
            name="Causas especiales"
        ))

    fig.update_layout(
        title=(f"Bandas de Bollinger de retornos — {crypto} "
               f"(ventana {window} días, ±{sigma}σ)"),
        xaxis_title="Fecha",
        yaxis_title="Retorno diario",
        plot_bgcolor="white",
        hovermode="x unified",
        legend_title="Series",
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    )

    n_out = len(df_out)
    total = len(df_plot)
    rango_txt = f"{df_plot.index.min().date()} a {df_plot.index.max().date()}"
    if n_out == 0:
        texto = (f"Para {crypto} no se detectan causas especiales en el periodo "
                 f"{rango_txt} con ventana de {window} días y bandas de ±{sigma}σ.")
    else:
        porc = 100 * n_out / total
        texto = (f"Para {crypto} se detectan {n_out} días fuera de las bandas de Bollinger "
                 f"({porc:.2f}% de las observaciones) en el periodo {rango_txt}, "
                 f"con ventana {window} y ±{sigma}σ.")

    return fig, texto


# ---------- Inciso 3b – Animación cripto ----------

def build_crypto_animation(selected_cryptos):
    if not selected_cryptos:
        fig = go.Figure()
        fig.update_layout(
            title="Select",
            plot_bgcolor="white"
        )
        return fig

    df_raw = CRYPTO_WIDE[selected_cryptos].copy()
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw = df_raw.sort_index()

    # Serie mensual (primer precio del mes, forward fill)
    px_m_cols = {}
    for c in selected_cryptos:
        s = df_raw[c].dropna()
        if s.empty:
            continue
        s_m = s.resample("MS").first().ffill()
        px_m_cols[c] = s_m

    if not px_m_cols:
        fig = go.Figure()
        fig.update_layout(
            title="No hay datos mensuales suficientes para las criptomonedas seleccionadas",
            plot_bgcolor="white"
        )
        return fig

    px_m = pd.concat(px_m_cols.values(), axis=1)
    px_m.columns = list(px_m_cols.keys())

    date_col = px_m.index.name or "Date"
    w = px_m.reset_index().melt(id_vars=date_col, var_name="Crypto", value_name="Price")
    w[date_col] = pd.to_datetime(w[date_col], errors="coerce")
    w = w.dropna(subset=[date_col, "Price"]).sort_values([date_col, "Crypto"])
    w["Crypto"] = pd.Categorical(w["Crypto"], categories=selected_cryptos, ordered=True)

    unique_dates = pd.to_datetime(w[date_col].drop_duplicates().sort_values()).tolist()
    if not unique_dates:
        fig = go.Figure()
        fig.update_layout(
            title="No hay fechas válidas para la animación",
            plot_bgcolor="white"
        )
        return fig

    min_date = unique_dates[0]

    def traces_for_date(d):
        traces = []
        for c in selected_cryptos:
            sub = w[(w["Crypto"] == c) & (w[date_col] <= d)].sort_values(date_col)
            if sub.empty:
                traces.append(go.Scatter(x=[], y=[], name=c, mode="lines"))
            else:
                traces.append(go.Scatter(x=sub[date_col], y=sub["Price"], name=c, mode="lines"))
        return traces

    init_traces = traces_for_date(unique_dates[0])

    frames = []
    for i, d in enumerate(unique_dates):
        frames.append(
            go.Frame(
                data=traces_for_date(d),
                name=str(i),
                layout=go.Layout(xaxis=dict(range=[min_date, d], type="date"))
            )
        )

    slider_steps = [{
        "args": [[str(i)], {"frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}}],
        "label": pd.to_datetime(d).strftime("%Y-%m"),
        "method": "animate"
    } for i, d in enumerate(unique_dates)]

    slider = {
        "active": 0,
        "pad": {"t": 30, "b": 0},
        "x": 0.5, "y": -0.15,
        "xanchor": "center", "yanchor": "top",
        "len": 0.9,
        "currentvalue": {"prefix": "Mes: ", "visible": True},
        "steps": slider_steps,
    }

    updatemenus = [{
        "type": "buttons",
        "showactive": False,
        "direction": "left",
        "x": 0.5, "y": -0.25,
        "xanchor": "center", "yanchor": "top",
        "pad": {"r": 10, "t": 10},
        "buttons": [
            {"label": "▶︎", "method": "animate",
             "args": [None, {"frame": {"duration": 180, "redraw": True},
                             "fromcurrent": True,
                             "transition": {"duration": 80}}]},
            {"label": "⏸︎", "method": "animate",
             "args": [[None], {"frame": {"duration": 0, "redraw": False},
                               "mode": "immediate",
                               "transition": {"duration": 0}}]},
        ],
    }]

    fig = go.Figure(
        data=init_traces,
        frames=frames,
        layout=go.Layout(
            title="Evolución de precios — Criptomonedas (mensual, acumulativo, USD)",
            xaxis=dict(type="date", range=[min_date, unique_dates[0]], title="Fecha"),
            yaxis=dict(title="Precio (USD)"),
            plot_bgcolor="white",
            hovermode="x unified",
            legend_title_text="Crypto",
            updatemenus=updatemenus,
            sliders=[slider],
            margin=dict(l=60, r=40, t=60, b=120)
        )
    )

    return fig

@app.callback(
    Output("crypto_sel_anim", "value"),
    Input("btn_anim_select_all", "n_clicks"),
    Input("btn_anim_clear", "n_clicks"),
    prevent_initial_call=True
)
def actualizar_dropdown_anim(n_all, n_clear):
    ctx = callback_context
    if not ctx.triggered:
        raise exceptions.PreventUpdate

    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if btn_id == "btn_anim_select_all":
        return crypto_list
    elif btn_id == "btn_anim_clear":
        return []
    raise exceptions.PreventUpdate

@app.callback(
    Output("graf_anim_crypto", "figure"),
    Input("crypto_sel_anim", "value")
)
def actualizar_grafico_anim(cryptos_sel):
    if not cryptos_sel:
        fig = go.Figure()
        fig.update_layout(
            title="Select",
            plot_bgcolor="white"
        )
        return fig
    return build_crypto_animation(cryptos_sel)


# ---------- Inciso 3c – Max Drawdown y CVaR 95% (criptos) ----------

@app.callback(
    Output("graf_mdd", "figure"),
    Output("graf_cvar", "figure"),
    Input("tabs-proyecto", "value")  # se recalcula al entrar a la tab
)
def actualizar_riesgo_cripto(tab):
    if tab != "inciso3c":
        raise exceptions.PreventUpdate

    mdd = {t: max_drawdown(CRYPTO_WIDE[t].dropna()) for t in CRYPTO_WIDE.columns}
    cvar95 = {t: cvar_historico(CRYPTO_RET[t].dropna().values, alpha=0.95)
              for t in CRYPTO_RET.columns}

    df_risk = pd.DataFrame({"MaxDrawdown": mdd, "CVaR95": cvar95})
    df_risk_sorted = df_risk.sort_values(["MaxDrawdown", "CVaR95"],
                                         ascending=[True, False])

    df_plot = df_risk_sorted.reset_index().rename(columns={"index": "Crypto"})

    fig_mdd = px.bar(df_plot, x="Crypto", y="MaxDrawdown",
                     title="Max Drawdown — por criptomoneda")
    fig_mdd.update_layout(plot_bgcolor="white")

    fig_cvar = px.bar(df_plot, x="Crypto", y="CVaR95",
                      title="CVaR 95% (histórico) — por criptomoneda")
    fig_cvar.update_layout(plot_bgcolor="white")

    return fig_mdd, fig_cvar


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    # Para desarrollo local; en Render usarás gunicorn con `server`
    app.run_server(debug=False, host="0.0.0.0", port=8050)
