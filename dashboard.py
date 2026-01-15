# -*- coding: utf-8 -*-
"""
Dashboard interativo do backtest TCIM - Versão Otimizada (UI Ajustada)
"""
from __future__ import annotations

from pathlib import Path
import base64
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


FAVICON_PATH = Path("favicon.ico")

# Defaults replicados localmente
DEFAULT_BACKTEST_VERSION_BY_REGION = {
    "America": ["1.2.3", "1.0.3"],
    "Asia": ["1.2.1", "1.0.1"],
    "Europe": ["1.2.2", "1.0.2"],
}
DEFAULT_ACTIVE_VERSION_BY_REGION = {
    "America": "1.2.3",
    "Asia": "1.2.1",
    "Europe": "1.2.2",
}

# Configuração da página
st.set_page_config(
    page_title="TCIM Dashboard",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "TCIM",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------
# CSS Personalizado (Visual Pro + Ajustes Sidebar)
# ------------------------------------------------------
st.markdown("""
<style>
    /* Estilo para os cartões de métricas (KPIs) */
    div[data-testid="stMetric"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #b0b0b0;
    }
    
    /* Ajuste de contraste para inputs na sidebar */
    [data-testid="stSidebar"] input[type="number"] {
        background-color: #2b2b2b; /* Fundo mais claro que o preto total */
        border: 1px solid #444;
        color: #ffffff;
        border-radius: 4px;
    }
    
    /* Melhorar legibilidade dos labels na sidebar */
    [data-testid="stSidebar"] label {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------
# Funções de suporte
# ------------------------------------------------------

def _csv_signature(csv_path: Path) -> tuple[int, int]:
    try:
        stats = csv_path.stat()
        return stats.st_mtime_ns, stats.st_size
    except FileNotFoundError:
        return 0, 0


@st.cache_data
def load_data(csv_path: Path, signature: tuple[int, int]) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, sep=";")
        df["DateParsed"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        df["AnalysisUTC"] = pd.to_datetime(df["AnalysisUTC"], errors="coerce")
        return df
    except Exception as exc:
        st.error(f"Erro ao ler CSV: {exc}")
        return pd.DataFrame()


def compute_directional_pnl(df: pd.DataFrame) -> pd.Series:
    pnl = pd.Series(np.nan, index=df.index, dtype="float")
    buys = df["TCIM_Vies"] == "COMPRA"
    sells = df["TCIM_Vies"] == "VENDA"
    pnl.loc[buys] = pd.to_numeric(df.loc[buys, "Up"], errors="coerce")
    pnl.loc[sells] = pd.to_numeric(df.loc[sells, "Down"], errors="coerce")
    return pnl


def _dynamic_stake_equity(
    df_filtered: pd.DataFrame,
    initial_capital: float,
    stake_pct_map: dict[tuple[str, str], float],
    reapply_map: dict[tuple[str, str], bool],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    capital = initial_capital
    stake_values = []
    capital_values = []
    
    for _, row in df_filtered.iterrows():
        if capital <= 0:
            capital = 0.0
            stake_values.append(0.0)
            capital_values.append(capital)
            continue
        
        pct = stake_pct_map.get((row["Region"], row["Version"]), 0.0)
        base = capital if reapply_map.get((row["Region"], row["Version"]), False) else initial_capital
        stake = base * pct
        
        pnl_val = row["PnL"] if pd.notna(row["PnL"]) else 0.0
        capital = max(0.0, capital + stake * pnl_val)
        
        stake_values.append(stake)
        capital_values.append(capital)
        
    stake_series = pd.Series(stake_values, index=df_filtered.index)
    capital_series = pd.Series(capital_values, index=df_filtered.index)
    growth = (capital_series / initial_capital) - 1.0
    return stake_series, capital_series, growth


def _region_metrics(df_regiao: pd.DataFrame) -> dict:
    total = len(df_regiao)
    acertos = df_regiao["Acertou"].sum()
    taxa = acertos / total if total > 0 else np.nan

    ganhos = df_regiao[df_regiao["Acertou"]]["SignalPnL"].dropna()
    perdas = df_regiao[~df_regiao["Acertou"]]["SignalPnL"].dropna()

    media_ganho = ganhos.mean() if not ganhos.empty else np.nan
    media_perda = perdas.mean() if not perdas.empty else np.nan

    total_ganho = ganhos.sum() if not ganhos.empty else np.nan
    total_perda = perdas.sum() if not perdas.empty else np.nan

    payoff_rr = np.nan
    if not np.isnan(media_ganho) and not np.isnan(media_perda) and media_perda != 0:
        payoff_rr = abs(media_ganho / media_perda)

    fator_lucro = np.nan
    if not np.isnan(total_ganho) and not np.isnan(total_perda) and total_perda != 0:
        fator_lucro = abs(total_ganho / total_perda)

    exp = np.nan
    if not np.isnan(media_ganho) and not np.isnan(media_perda) and not np.isnan(taxa):
        exp = taxa * media_ganho + (1 - taxa) * media_perda

    return {
        "entradas": total,
        "taxa": taxa,
        "ganho_medio": media_ganho,
        "perda_medio": media_perda,
        "payoff_rr": payoff_rr,
        "fator_lucro": fator_lucro,
        "E": exp,
    }


def _default_version_for_region(regiao: str) -> str | None:
    live_map = DEFAULT_ACTIVE_VERSION_BY_REGION or {}
    backtest_map = DEFAULT_BACKTEST_VERSION_BY_REGION or {}
    val = live_map.get(regiao)
    if val is None:
        val = backtest_map.get(regiao)
    if isinstance(val, (list, tuple)):
        return val[0] if val else None
    return str(val) if val is not None else None


def _available_versions_for_region(regiao: str, df_entries: pd.DataFrame) -> list[str]:
    configured = DEFAULT_BACKTEST_VERSION_BY_REGION.get(regiao, []) or []
    if not isinstance(configured, list):
        configured = [configured]
    from_csv = (
        df_entries.loc[df_entries["Region"] == regiao, "Version"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    combined = sorted({*configured, *from_csv})
    return combined


def _format_number_br(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return f"{0:.{decimals}f}".replace(".", ",")
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def _format_currency_br(value: float, decimals: int = 2) -> str:
    return f"${_format_number_br(value, decimals)}"


def _format_percent_br(value: float, decimals: int = 2, show_sign: bool = False) -> str:
    sign = ""
    if show_sign:
        if value > 0:
            sign = "+"
        elif value < 0:
            sign = "-"
    return f"{sign}{_format_number_br(abs(value), decimals)}%"

def highlight_pnl_col(val):
    if pd.isna(val):
        return ''
    color = '#ff4b4b' if val < 0 else '#2dc937'
    return f'color: {color}'

# ------------------------------------------------------
# Funções Auxiliares de Drawdown (Datas)
# ------------------------------------------------------
def _calculate_dd_window(series_values: pd.Series, date_series: pd.Series) -> tuple[int, int, pd.Timestamp, pd.Timestamp]:
    """Retorna índice pico, índice vale, data pico, data vale"""
    if series_values.empty:
        return 0, 0, pd.NaT, pd.NaT
    
    # Índice do ponto mais baixo (vale)
    trough_idx = int(series_values.idxmin())
    
    # Serie até o ponto do vale para achar o pico anterior
    # Importante: estamos assumindo que series_values já é o DD ou Equity dependendo do contexto.
    # Mas para achar datas corretas, precisamos analisar a curva de DD.
    
    # Se series_values for o Drawdown (ex: 0, -0.1, -0.2...), o pico anterior é o último 0 antes do vale.
    # Mas aqui vamos assumir que passamos a série de DD já calculada.
    
    # Simplificação: O DD máximo ocorre no 'trough_idx'. 
    # O início do DD é o último ponto onde DD era 0 antes de trough_idx.
    
    # Fatia até o vale
    slice_dd = series_values.iloc[:trough_idx+1]
    
    # Início do DD é onde o DD era 0 (ou quase 0) mais recentemente
    # Se series_values é DD puro (ex: 0.0, -0.05), procuramos o max (que é 0)
    peaks = slice_dd[slice_dd == 0]
    if peaks.empty:
        # Se não tem zero (começou já caindo), pega o primeiro
        peak_idx = 0
    else:
        peak_idx = int(peaks.index[-1])
        
    start_date = date_series.iloc[peak_idx]
    end_date = date_series.iloc[trough_idx]
    
    return peak_idx, trough_idx, start_date, end_date


# ------------------------------------------------------
# Interface principal
# ------------------------------------------------------

def main() -> None:
    logo_path = Path("logo.png")

    # --- SIDEBAR ---
    with st.sidebar:
        if logo_path.exists():
            logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
            st.markdown(
                f'<div style="text-align:center;">'
                f'<img src="data:image/png;base64,{logo_b64}" style="width:140px;" />'
                f'<div style="font-size:36px; font-weight:700; margin-top:6px;">TCIM</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        st.divider()
        st.header("Configurações")

        default_csv = Path("tcim_backtest_results.csv")
        csv_path_str = st.text_input("Arquivo CSV", value=str(default_csv))
        csv_path = Path(csv_path_str)

        if not csv_path.exists():
            st.error(f"Arquivo não encontrado: {csv_path}")
            return

        if st.button("🔄 Recarregar Dados"):
            load_data.clear()

        file_signature = _csv_signature(csv_path)
        df = load_data(csv_path, file_signature)
       
        if df.empty:
            st.warning("CSV vazio ou inválido.")
            return

        df["PnL"] = compute_directional_pnl(df)
        df_entries = df[df["PnL"].notna()].copy()

        st.subheader("Filtros Globais")
        regioes_disponiveis = sorted(df_entries["Region"].dropna().unique())
        regioes_sel = st.multiselect("Regiões Ativas", options=regioes_disponiveis, default=regioes_disponiveis)
        
        versao_sel_por_regiao: dict[str, str | None] = {}
        versoes_por_regiao: dict[str, list[str]] = {}
        
        for reg in regioes_disponiveis:
            versoes = _available_versions_for_region(reg, df_entries)
            versoes_por_regiao[reg] = versoes
            default_version = _default_version_for_region(reg) or "Todas"
            opcoes = ["Todas"] + versoes
            
            if len(opcoes) > 2:
                selecao = st.selectbox(
                    f"Versão ({reg})",
                    options=opcoes,
                    index=opcoes.index(default_version) if default_version in opcoes else 0,
                )
                versao_sel_por_regiao[reg] = None if selecao == "Todas" else selecao
            else:
                versao_sel_por_regiao[reg] = None

        # Filtro de Data
        date_source = df["DateParsed"].dropna()
        if date_source.empty:
            date_source = df_entries["DateParsed"].dropna()
        min_date = date_source.min()
        max_date = date_source.max()
        if pd.notna(min_date):
            min_date = min_date.date()
        if pd.notna(max_date):
            max_date = max_date.date()

        date_range = st.date_input(
            "Período de Análise",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date

        # --- GESTÃO DE RISCO (Expander Ajustado) ---
        stake_por_regiao_versao: dict[tuple[str, str], float] = {}
        reapply_por_regiao_versao: dict[tuple[str, str], bool] = {}
        default_percent = 10.0

        with st.expander("⚙️ Gestão de Risco", expanded=False):
            capital_inicial = st.number_input("Capital Inicial ($)", min_value=100.0, value=1000.0, step=100.0)
            st.divider()
            
            st.caption("Alocação por Trade (% do Capital)")
            for reg in regioes_disponiveis:
                st.markdown(f"**{reg}**")
                for ver in versoes_por_regiao.get(reg, []):
                    # Layout ajustado: Input menor (usando colunas vazias para "espremer")
                    # c_in: input, c_chk: checkbox, c_buff: espaço vazio
                    c_in, c_chk, c_buff = st.columns([1.2, 0.8, 0.5]) 
                    with c_in:
                        pct = st.number_input(
                            f"% ({ver})",
                            min_value=1.0, max_value=100.0, value=default_percent, step=1.0,
                            key=f"pct_{reg}_{ver}",
                            label_visibility="visible" 
                        )
                    with c_chk:
                        # Espaço vertical para alinhar checkbox com input
                        st.write("") 
                        st.write("")
                        reapply = st.checkbox("Juros Comp.", value=False, key=f"re_{reg}_{ver}")
                    
                    stake_por_regiao_versao[(reg, ver)] = pct / 100.0
                    reapply_por_regiao_versao[(reg, ver)] = reapply
                st.divider()

        n_piores = st.slider("Top Loss (qtd)", 3, 50, 10)
        st.caption(f"Registros: {len(df)}")


    # --- FILTRAGEM ---
    mask = pd.Series(True, index=df_entries.index)
    if regioes_sel:
        mask &= df_entries["Region"].isin(regioes_sel)
    
    version_series = df_entries["Version"].astype(str)
    version_series = version_series.where(df_entries["Version"].notna(), "")
    
    for reg, versao in versao_sel_por_regiao.items():
        if versao:
            mask &= ~((df_entries["Region"] == reg) & (version_series != versao))
    
    mask &= df_entries["DateParsed"].between(pd.to_datetime(start_date), pd.to_datetime(end_date))

    df_filtered = df_entries[mask].copy()
    df_filtered["Version"] = df_filtered["Version"].astype(str).replace("nan", "")
    df_filtered["Mes"] = df_filtered["DateParsed"].dt.to_period("M").astype(str)
    df_filtered = df_filtered.sort_values("AnalysisUTC")
    
    # Cálculo
    stake_series, capital, ret_cum = _dynamic_stake_equity(
        df_filtered,
        capital_inicial,
        stake_por_regiao_versao,
        reapply_por_regiao_versao,
    )
    df_filtered["StakeRV"] = stake_series.values
    df_filtered["Capital"] = capital.values
    df_filtered["RetornoAcum"] = ret_cum.values

    # Reset de índice para cálculos de DD com datas
    df_filtered = df_filtered.reset_index(drop=True)
    date_series_reset = df_filtered["AnalysisUTC"]
    
    # Cálculos Globais de DD para exibição
    pico_acumulado = df_filtered["Capital"].cummax()
    dd_pct_series = (df_filtered["Capital"] - pico_acumulado) / pico_acumulado
    dd_abs_series = df_filtered["Capital"] - pico_acumulado
    
    # Encontrar as datas dos picos/vales
    _, _, start_dd_pct, end_dd_pct = _calculate_dd_window(dd_pct_series, date_series_reset)
    _, _, start_dd_abs, end_dd_abs = _calculate_dd_window(dd_abs_series, date_series_reset)
    
    max_dd_pct = dd_pct_series.min() if not dd_pct_series.empty else 0.0
    max_dd_abs = dd_abs_series.min() if not dd_abs_series.empty else 0.0


    # --- DASHBOARD HEADER ---
    st.title("Relatório de Performance TCIM")
    st.markdown(
        f"**Período:** {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')} | **Regiões:** {', '.join(regioes_sel) if regioes_sel else 'Todas'}"
    )

    if df_filtered.empty:
        st.warning("⚠️ Nenhuma entrada encontrada com os filtros selecionados.")
        return

    # --- KPIS ---
    df_stats_global = df_filtered[df_filtered["TCIM_Vies"].notna() & (df_filtered["TCIM_Vies"] != "FORA")].copy()

    if not df_stats_global.empty:
        df_stats_global["SignalPnL"] = np.where(
            df_stats_global["TCIM_Vies"] == "COMPRA",
            pd.to_numeric(df_stats_global["Up"], errors="coerce"),
            pd.to_numeric(df_stats_global["Down"], errors="coerce"),
        )
        df_stats_global["Acertou"] = df_stats_global["SignalPnL"] > 0
        metrics_global = _region_metrics(df_stats_global)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        final_capital = df_filtered["Capital"].iloc[-1]
        roi_total = ((final_capital - capital_inicial) / capital_inicial) * 100

        kpi1.metric("Capital Final", _format_currency_br(final_capital), _format_percent_br(roi_total, 2, show_sign=True))
        kpi2.metric("Total Trades", metrics_global["entradas"])
        kpi3.metric("Win Rate Global", _format_percent_br(metrics_global["taxa"] * 100, 2))
        kpi4.metric("Fator de Lucro", _format_number_br(metrics_global["fator_lucro"], 2))

    st.divider()

    # --- TABS ---
    tab_metodo, tab_curva, tab_regiao, tab_mensal, tab_trades = st.tabs(
        ["📘 Método", "📈 Curva & Drawdown", "🌍 Análise Regional", "📅 Análise Mensal", "💰 Trades & Dados"]
    )

    with tab_metodo:
        st.subheader("Sobre o Método TCIM")
        c_desc, c_tele = st.columns([2, 1])
        with c_desc:
            st.markdown(
                """
                O **TCIM** (Tendência, Contexto, Impulso e Mitigação) é um algoritmo quantitativo que gera viés probabilístico 
                cerca de 30 minutos antes da abertura das sessões globais.
                """
            )
        with c_tele:
            st.info("Telegram Channel info placeholder")

    with tab_curva:
        st.subheader("Evolução do Patrimônio e Risco")
        
        # Opção de Visualização
        tipo_grafico = st.radio(
            "Estilo do Gráfico:",
            ["Clássico (Com Marcações)", "Analítico (Subplots)"],
            horizontal=True
        )
        
        # Exibição dos Dados de DD (Texto)
        c_dd1, c_dd2 = st.columns(2)
        
        str_periodo_pct = ""
        if pd.notna(start_dd_pct) and pd.notna(end_dd_pct):
            str_periodo_pct = f"({start_dd_pct.strftime('%d/%m')} a {end_dd_pct.strftime('%d/%m')})"
            
        str_periodo_abs = ""
        if pd.notna(start_dd_abs) and pd.notna(end_dd_abs):
            str_periodo_abs = f"({start_dd_abs.strftime('%d/%m')} a {end_dd_abs.strftime('%d/%m')})"

        c_dd1.metric("Max Drawdown (%)", _format_percent_br(max_dd_pct * 100, 2))
        c_dd1.caption(f"Período: {str_periodo_pct}")
        
        c_dd2.metric("Max Drawdown ($)", _format_number_br(max_dd_abs, 2))
        c_dd2.caption(f"Período: {str_periodo_abs}")
        
        st.markdown("---")

        if tipo_grafico == "Analítico (Subplots)":
            # --- MODELO NOVO ---
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05,
                row_heights=[0.70, 0.30],
                subplot_titles=("Curva de Capital", "Drawdown Subaquático (%)")
            )
            fig.add_trace(
                go.Scatter(x=df_filtered["AnalysisUTC"], y=df_filtered["Capital"],
                    mode='lines', name='Capital', line=dict(color='#00CC96', width=2),
                    fill='tozeroy', fillcolor='rgba(0, 204, 150, 0.1)'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_filtered["AnalysisUTC"], y=dd_pct_series,
                    mode='lines', name='Drawdown', line=dict(color='#EF553B', width=1),
                    fill='tozeroy', fillcolor='rgba(239, 85, 59, 0.2)'),
                row=2, col=1
            )
            fig.update_layout(template="plotly_dark", hovermode="x unified", height=600,
                              margin=dict(l=40, r=40, t=40, b=40), legend=dict(orientation="h", y=1.1))
            fig.update_yaxes(tickformat=",.0f", title="Saldo ($)", row=1, col=1)
            fig.update_yaxes(tickformat=".1%", title="Queda (%)", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

        else:
            # --- MODELO CLÁSSICO ---
            fig = px.line(
                df_filtered, x="AnalysisUTC", y="Capital", color="Region",
                title="Crescimento do Capital", markers=True, template="plotly_dark"
            )
            fig.update_layout(hovermode="x unified", height=500)
            
            # Adicionar retângulos de DD
            if pd.notna(start_dd_pct) and pd.notna(end_dd_pct):
                fig.add_vrect(
                    x0=start_dd_pct, x1=end_dd_pct,
                    fillcolor="rgba(255, 99, 71, 0.2)", line_width=0,
                    annotation_text="DD Máx", annotation_position="top left"
                )
            st.plotly_chart(fig, use_container_width=True)


    with tab_regiao:
        st.subheader("Performance Detalhada por Região")
        region_stats_rows = []
        for reg, df_reg in df_stats_global.groupby("Region"):
            stats = _region_metrics(df_reg)
            region_stats_rows.append(
                {
                    "Região": reg,
                    "Trades": stats["entradas"],
                    "Taxa Acerto": stats["taxa"],
                    "Ganho Médio": stats["ganho_medio"],
                    "Perda Média": stats["perda_medio"],
                    "Payoff (R:R)": stats["payoff_rr"],
                    "Fator Lucro": stats["fator_lucro"],
                    "Expectativa (E)": stats["E"],
                }
            )
        df_region_view = pd.DataFrame(region_stats_rows)
        st.dataframe(
            df_region_view,
            column_config={
                "Taxa Acerto": st.column_config.ProgressColumn("Taxa de Acerto", format="%.2f%%", min_value=0, max_value=1),
                "Ganho Médio": st.column_config.NumberColumn(format="$%.2f"),
                "Perda Média": st.column_config.NumberColumn(format="$%.2f"),
                "Payoff (R:R)": st.column_config.NumberColumn(format="%.2f"),
                "Fator Lucro": st.column_config.NumberColumn(format="%.2f"),
                "Expectativa (E)": st.column_config.NumberColumn(format="%.4f"),
            },
            hide_index=True,
            use_container_width=True,
        )

    with tab_mensal:
        st.subheader("Consistência Mensal")
        monthly_rows = []
        for (reg, mes), df_group in df_stats_global.groupby(["Region", "Mes"]):
            stats = _region_metrics(df_group)
            monthly_rows.append(
                {"Região": reg, "Mes": mes, "Trades": stats["entradas"],
                 "Taxa Acerto": stats["taxa"], "Payoff (R:R)": stats["payoff_rr"], "Fator Lucro": stats["fator_lucro"]}
            )
        st.dataframe(pd.DataFrame(monthly_rows).sort_values(["Região", "Mes"]), hide_index=True, use_container_width=True)

    with tab_trades:
        col_worst, col_raw = st.columns([1, 2])
        with col_worst:
            st.markdown(f"### 🔻 Top {n_piores} Maiores Perdas")
            perdas = df_filtered[df_filtered["PnL"] < 0].nsmallest(n_piores, "PnL")
            st.dataframe(
                perdas[["Date", "Region", "Symbol", "PnL"]].style.format({"PnL": "${:.2f}"}).map(highlight_pnl_col, subset=["PnL"]),
                use_container_width=True, hide_index=True
            )
        with col_raw:
            st.markdown("### 📋 Diário de Trades")
            st.dataframe(
                df_filtered[["Date", "Region", "Symbol", "PnL", "TCIM_Vies", "TCIM_Score"]].style
                .format({"PnL": "${:.2f}", "TCIM_Score": "{:.2f}"})
                .map(highlight_pnl_col, subset=["PnL"]),
                height=500, use_container_width=True, hide_index=True
            )

if __name__ == "__main__":
    main()