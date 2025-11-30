"""
Dashboard interativo do backtest TCIM.
"""
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import base64
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Primeira chamada do Streamlit: configuração da página
st.set_page_config(
    page_title="TCIM Dashboard",
    page_icon="TCIM",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------
# Funções de suporte
# ------------------------------------------------------

@st.cache_data
def load_data(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, sep=";")
        df["DateParsed"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        df["AnalysisUTC"] = pd.to_datetime(df["AnalysisUTC"], errors="coerce")
        return df
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erro ao ler CSV: {exc}")
        return pd.DataFrame()


def compute_directional_pnl(df: pd.DataFrame) -> pd.Series:
    pnl = pd.Series(np.nan, index=df.index, dtype="float")
    # Mantive as strings de comparação sem acento caso o CSV venha sem acento
    buys = df["TCIM_Vies"] == "COMPRA"
    sells = df["TCIM_Vies"] == "VENDA"
    pnl.loc[buys] = pd.to_numeric(df.loc[buys, "Up"], errors="coerce")
    pnl.loc[sells] = pd.to_numeric(df.loc[sells, "Down"], errors="coerce")
    return pnl


def equity_curve(
    pnl: pd.Series,
    initial_capital: float,
    stake_per_trade: float | pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    pnl = pnl.fillna(0.0)
    stake_series = (
        pd.Series(stake_per_trade, index=pnl.index)
        if isinstance(stake_per_trade, (int, float))
        else pd.Series(stake_per_trade).reindex(pnl.index).fillna(0.0)
    )
    capital_values = []
    capital = initial_capital
    for val, stake in zip(pnl, stake_series):
        capital += stake * val
        capital_values.append(capital)
    capital_series = pd.Series(capital_values, index=pnl.index)
    growth = (capital_series / initial_capital) - 1.0
    return capital_series, growth


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


# ------------------------------------------------------
# Interface principal
# ------------------------------------------------------


def main() -> None:
    logo_path = Path("logo.png")

    # Sidebar
    with st.sidebar:
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        st.divider()
        st.header("Configurações")

        default_csv = Path("tcim_backtest_results.csv")
        csv_path_str = st.text_input("Arquivo CSV", value=str(default_csv))
        csv_path = Path(csv_path_str)

        if not csv_path.exists():
            st.error(f"Arquivo não encontrado: {csv_path}")
            return

        df = load_data(csv_path)
        if df.empty:
            st.warning("CSV vazio ou inválido.")
            return

        df["PnL"] = compute_directional_pnl(df)
        df_entries = df[df["PnL"].notna()].copy()

        st.subheader("Filtros")
        regioes_disponiveis = sorted(df_entries["Region"].dropna().unique())
        regioes_sel = st.multiselect("Regiões", options=regioes_disponiveis, default=regioes_disponiveis)

        min_date = df_entries["DateParsed"].min()
        max_date = df_entries["DateParsed"].max()

        date_range = st.date_input(
            "Período",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date

        st.subheader("Gestão de Capital")
        capital_inicial = st.number_input("Capital Inicial ($)", min_value=100.0, value=1000.0, step=100.0)
        st.caption("Informe o caixa por trade para cada região")
        stake_por_regiao: dict[str, float] = {}
        default_stake = 100.0
        for reg in regioes_disponiveis:
            stake_por_regiao[reg] = st.number_input(
                f"Caixa por Trade ($) - {reg}",
                min_value=0.0,
                value=default_stake,
                step=10.0,
            )

        n_piores = st.slider("Listar top perdas", 3, 50, 10)

        st.info(f"Total de registros brutos: {len(df)}")

    # Filtros aplicados
    mask = pd.Series(True, index=df_entries.index)
    if regioes_sel:
        mask &= df_entries["Region"].isin(regioes_sel)
    mask &= df_entries["DateParsed"].between(pd.to_datetime(start_date), pd.to_datetime(end_date))

    df_filtered = df_entries[mask].copy()
    df_filtered["Mes"] = df_filtered["DateParsed"].dt.to_period("M").astype(str)

    # Ordena cronologicamente para curva e aplica stake por regiao
    df_filtered = df_filtered.sort_values("AnalysisUTC")
    df_filtered["StakeRegion"] = df_filtered["Region"].map(stake_por_regiao).fillna(0.0)

    # Calculos globais
    capital, ret_cum = equity_curve(df_filtered["PnL"], capital_inicial, df_filtered["StakeRegion"])
    df_filtered["Capital"] = capital.values
    df_filtered["RetornoAcum"] = ret_cum.values

    # Header
    st.title("Relatório de Performance TCIM")
    st.markdown(
        f"**Período:** {start_date} a {end_date} | **Regiões:** {', '.join(regioes_sel) if regioes_sel else 'Todas'}"
    )

    if df_filtered.empty:
        st.warning("Nenhuma entrada encontrada com os filtros selecionados.")
        return

    # KPIs globais
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

        kpi1.metric("Capital Final", f"${final_capital:,.2f}", f"{roi_total:+.2f}%")
        kpi2.metric("Total Trades", metrics_global["entradas"])
        kpi3.metric("Win Rate Global", f"{metrics_global['taxa']*100:.2f}%")
        kpi4.metric("Fator de Lucro", f"{metrics_global['fator_lucro']:.2f}")

    st.divider()

    # Abas
    tab_metodo, tab_curva, tab_regiao, tab_mensal, tab_trades = st.tabs(
        ["Método", "Curva de Capital", "Análise Regional", "Análise Mensal", "Piores Trades & Dados"]
    )

    with tab_metodo:
        st.subheader("Sobre o Método TCIM")
        st.markdown(
            """
O TCIM (Tendência, Contexto, Impulso e Mitigação) gera viés probabilístico (compra, venda ou fora) cerca de 30 minutos antes da abertura de Ásia, Europa e América.

**Blocos**
- Tendência: EMAs 20/50 e slopes.
- Contexto: preço vs VWAP/EMA50 e distância em ATR.
- Impulso: ADX.
- Mitigação: alerta de volatilidade extrema, pavios e esticamentos.

Scores somam sinais positivos/negativos. Score >= 2.5 compra; <= -2.5 venda; intermediário fora. Cada decisão vem com motivos e alertas.
"""
        )
        st.markdown(
    """
### **Sincronize-se com o Mercado!**

Receba o viés analítico do TCIM e alertas de risco 
diretamente no seu Telegram antes da abertura:<br>

⏰ **Horários de Disparo (Pré-Sessão):**<br>
🇺🇸 **América:** 10:15<br>
🇯🇵 **Ásia:** 20:30<br>
🇪🇺 **Europa:** 03:30
""",
    unsafe_allow_html=True
)

        telegram_path = Path("telegram.png")
        if telegram_path.exists():
            img_b64 = base64.b64encode(telegram_path.read_bytes()).decode("ascii")
            st.markdown(
                f'<a href="https://t.me/TCIM_viesBot" target="_blank" rel="noopener">'
                f'<img src="data:image/png;base64,{img_b64}" alt="Telegram" style="height:24px;vertical-align:middle;margin-right:8px;">'
                f'</a>'
                f'<a href="https://t.me/TCIM_viesBot" target="_blank" rel="noopener">@TCIM_viesBot</a>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('[@TCIM_viesBot](https://t.me/TCIM_viesBot)')

    with tab_curva:
        st.subheader("Evolução Simulada do Patrimônio")
        fig = px.line(
            df_filtered,
            x="AnalysisUTC",
            y="Capital",
            color="Region",
            title="Crescimento do Capital por Região",
            markers=True,
            template="plotly_dark",
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        capital_series = df_filtered["Capital"].reset_index(drop=True)
        capital_with_start = pd.concat([pd.Series([capital_inicial]), capital_series], ignore_index=True)
        pico = capital_with_start.cummax()
        dd_series = (capital_with_start - pico) / pico
        max_dd_pct = dd_series.min() if not dd_series.empty else 0.0
        max_dd_abs = (capital_with_start - pico).min() if not capital_with_start.empty else 0.0
        avg_stake = df_filtered["StakeRegion"].mean()
        boxes_lost = abs(max_dd_abs) / avg_stake if avg_stake > 0 else 0.0
        st.markdown(
            f"**Drawdown máximo:** {max_dd_pct*100:.2f}% ({max_dd_abs:,.2f} absoluto) ou {boxes_lost:.2f} caixas médias"
        )

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
                "Taxa Acerto": st.column_config.ProgressColumn(
                    "Taxa de Acerto",
                    format="%.2f%%",
                    min_value=0,
                    max_value=1,
                ),
                "Ganho Médio": st.column_config.NumberColumn(format="$%.4f"),
                "Perda Média": st.column_config.NumberColumn(format="$%.4f"),
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
                {
                    "Região": reg,
                    "Mes": mes,
                    "Trades": stats["entradas"],
                    "Taxa Acerto": stats["taxa"],
                    "Payoff (R:R)": stats["payoff_rr"],
                    "Fator Lucro": stats["fator_lucro"],
                }
            )

        df_monthly_view = pd.DataFrame(monthly_rows).sort_values(["Região", "Mes"])

        st.dataframe(
            df_monthly_view,
            column_config={
                "Taxa Acerto": st.column_config.NumberColumn(format="%.2f%%"),
                "Payoff (R:R)": st.column_config.NumberColumn(format="%.2f"),
                "Fator Lucro": st.column_config.NumberColumn(format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
        )

    with tab_trades:
        col_worst, col_raw = st.columns([1, 2])

        with col_worst:
            st.markdown(f"### Top {n_piores} Maiores Perdas")
            perdas = df_filtered[df_filtered["PnL"] < 0].nsmallest(n_piores, "PnL")
            st.dataframe(
                perdas[["Date", "Region", "Symbol", "PnL"]],
                column_config={"PnL": st.column_config.NumberColumn(format="$%.4f")},
                hide_index=True,
            )

        with col_raw:
            st.markdown("### Dados Completos")
            st.dataframe(
                df_filtered[["Date", "Region", "Symbol", "PnL", "TCIM_Vies", "TCIM_Score", "TCIM_Motivos"]],
                height=400,
                column_config={"PnL": st.column_config.NumberColumn(format="$%.4f")},
            )


if __name__ == "__main__":
    main()