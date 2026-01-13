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




FAVICON_PATH = Path("favicon.ico")

# Defaults replicados localmente para evitar depender do config em ambiente GitHub.
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

# Primeira chamada do Streamlit: configuração da página
st.set_page_config(
    page_title="TCIM Dashboard",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else "TCIM",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------
# Funções de suporte
# ------------------------------------------------------

def _csv_signature(csv_path: Path) -> tuple[int, int]:
    stats = csv_path.stat()
    return stats.st_mtime_ns, stats.st_size


@st.cache_data
def load_data(csv_path: Path, signature: tuple[int, int]) -> pd.DataFrame:
    try:
        # O argumento 'signature' forca o recarregamento se o arquivo mudou
        df = pd.read_csv(csv_path, sep=";")
        df["DateParsed"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        df["AnalysisUTC"] = pd.to_datetime(df["AnalysisUTC"], errors="coerce")
        return df
    except Exception as exc:
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
        if capital <= 0:
            capital = 0.0
            capital_values.append(capital)
            continue
        capital = max(0.0, capital + stake * val)
        capital_values.append(capital)
    capital_series = pd.Series(capital_values, index=pnl.index)
    growth = (capital_series / initial_capital) - 1.0
    return capital_series, growth


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


# ------------------------------------------------------
# Interface principal
# ------------------------------------------------------


def main() -> None:
    logo_path = Path("logo.png")

    # Sidebar

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
        st.header("Configuracoes")

        default_csv = Path("tcim_backtest_results.csv")
        csv_path_str = st.text_input("Arquivo CSV", value=str(default_csv))
        csv_path = Path(csv_path_str)

        if not csv_path.exists():
            st.error(f"Arquivo nao encontrado: {csv_path}")
            return

        if st.button("Recarregar CSV"):
            load_data.clear()

        file_signature = _csv_signature(csv_path)
        df = load_data(csv_path, file_signature)
       
        if df.empty:
            st.warning("CSV vazio ou invalido.")
            return

        df["PnL"] = compute_directional_pnl(df)
        df_entries = df[df["PnL"].notna()].copy()

        st.subheader("Filtros")
        regioes_disponiveis = sorted(df_entries["Region"].dropna().unique())
        regioes_sel = st.multiselect("Regioes", options=regioes_disponiveis, default=regioes_disponiveis)
        versao_sel_por_regiao: dict[str, str | None] = {}
        versoes_por_regiao: dict[str, list[str]] = {}
        for reg in regioes_disponiveis:
            versoes = _available_versions_for_region(reg, df_entries)
            versoes_por_regiao[reg] = versoes
            default_version = _default_version_for_region(reg) or "Todas"
            opcoes = ["Todas"] + versoes
            selecao = st.selectbox(
                f"Versao - {reg}",
                options=opcoes,
                index=opcoes.index(default_version) if default_version in opcoes else 0,
            )
            versao_sel_por_regiao[reg] = None if selecao == "Todas" else selecao

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
            "Periodo",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date

        st.subheader("Gestao de Capital")
        capital_inicial = st.number_input("Capital Inicial ($)", min_value=100.0, value=1000.0, step=100.0)
        st.caption("Informe o percentual de caixa por trade para cada regiao/versao")
        stake_por_regiao_versao: dict[tuple[str, str], float] = {}
        reapply_por_regiao_versao: dict[tuple[str, str], bool] = {}
        default_percent = 10.0
        for reg in regioes_disponiveis:
            for ver in versoes_por_regiao.get(reg, []):
                col_pct, col_reapply = st.columns([2, 1])
                with col_pct:
                    pct = st.number_input(
                        f"Caixa por Trade (%) - {reg} {ver}",
                        min_value=5.0,
                        max_value=100.0,
                        value=default_percent,
                        step=5.0,
                        key=f"pct_{reg}_{ver}",
                    )
                with col_reapply:
                    reapply = st.checkbox(
                        "Capital dinâmico",
                        value=False,
                        key=f"reapply_{reg}_{ver}",
                    )
                stake_por_regiao_versao[(reg, ver)] = pct / 100.0
                reapply_por_regiao_versao[(reg, ver)] = reapply

        n_piores = st.slider("Listar top perdas", 3, 50, 10)

        st.info(f"Total de registros brutos: {len(df)}")

    # Filtros aplicados
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
    df_filtered["Version"] = df_filtered["Version"].astype(str)
    df_filtered["Version"] = df_filtered["Version"].replace("nan", "")
    df_filtered["Mes"] = df_filtered["DateParsed"].dt.to_period("M").astype(str)

    # Ordena cronologicamente para curva e aplica stake por regiao
    df_filtered = df_filtered.sort_values("AnalysisUTC")
    # Calculos globais
    stake_series, capital, ret_cum = _dynamic_stake_equity(
        df_filtered,
        capital_inicial,
        stake_por_regiao_versao,
        reapply_por_regiao_versao,
    )
    df_filtered["StakeRV"] = stake_series.values
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

Scores somam sinais positivos/negativos. Ex.: Score >= 2.5 compra; <= -2.5 venda; intermediário fora. Cada decisão vem com motivos e alertas.
"""
        )
        st.markdown(
    """
### **Sincronize-se com o Mercado!**

Receba o viés analítico do TCIM e alertas de risco 
diretamente no seu Telegram antes da abertura:<br>

⏰ **Horários de Disparo (Pré-Sessão):**<br>
🇺🇸 **América:** 10:16 *(Versão 1.0.3 e 1.2.3)*<br>
🇯🇵 **Ásia:** 20:31 *(Versão 1.0.1 e 1.2.1)*<br>
🇪🇺 **Europa:** 03:46 *(Versão 1.0.2 e 1.2.2)*
""",
    unsafe_allow_html=True
)

        telegram_path = Path("telegram.png")
        if telegram_path.exists():
            img_b64 = base64.b64encode(telegram_path.read_bytes()).decode("ascii")
            st.markdown(
                f'<a href="https://t.me/+OB9T7OXQ2o1iYmJh" target="_blank" rel="noopener">'
                f'<img src="data:image/png;base64,{img_b64}" alt="Telegram" style="height:24px;vertical-align:middle;margin-right:8px;">'
                f'</a>'
                f'<a href="https://t.me/+OB9T7OXQ2o1iYmJh" target="_blank" rel="noopener">2k Extra - Técnicas de Decisão</a>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('[2k Extra - Técnicas de Decisão](https://t.me/+OB9T7OXQ2o1iYmJh)')

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
        st.plotly_chart(fig, width="stretch")

        capital_series = df_filtered["Capital"].reset_index(drop=True)
        capital_with_start = pd.concat([pd.Series([capital_inicial]), capital_series], ignore_index=True)
        date_series = pd.concat(
            [pd.Series([pd.NaT]), df_filtered["DateParsed"].reset_index(drop=True)],
            ignore_index=True,
        )
        pico = capital_with_start.cummax()
        dd_series = (capital_with_start - pico) / pico
        max_dd_pct = dd_series.min() if not dd_series.empty else 0.0
        max_dd_abs = (capital_with_start - pico).min() if not capital_with_start.empty else 0.0
        avg_stake = df_filtered["StakeRV"].mean()
        boxes_lost = abs(max_dd_abs) / avg_stake if avg_stake > 0 else 0.0
        dd_period = ""
        if not dd_series.empty:
            trough_idx = int(dd_series.idxmin())
            peak_slice = capital_with_start.iloc[: trough_idx + 1]
            peak_val = peak_slice.max()
            peak_indices = peak_slice[peak_slice == peak_val].index
            peak_idx = int(peak_indices[-1]) if len(peak_indices) else 0
            start_date_dd = date_series.iloc[peak_idx]
            end_date_dd = date_series.iloc[trough_idx]
            if pd.notna(start_date_dd) and pd.notna(end_date_dd):
                dd_period = f" - de {start_date_dd:%d/%m/%Y} ate {end_date_dd:%d/%m/%Y}"
        st.markdown(
            f"**Drawdown máximo:** {max_dd_pct*100:.2f}% ({max_dd_abs:,.2f} absoluto) ou {boxes_lost:.2f} caixas médias{dd_period}"
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
            width="stretch",
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
            width="stretch",
        )

    with tab_trades:
        col_worst, col_raw = st.columns([1, 2])

        with col_worst:
            st.markdown(f"### Top {n_piores} Maiores Perdas")
            perdas = df_filtered[df_filtered["PnL"] < 0].nsmallest(n_piores, "PnL")
            st.dataframe(
                perdas[["Date", "Region", "Version", "Symbol", "PnL"]],
                column_config={"PnL": st.column_config.NumberColumn(format="$%.4f")},
                hide_index=True,
            )

        with col_raw:
            st.markdown("### Dados Completos")
            st.dataframe(
                df_filtered[["Date", "Region", "Version", "Symbol", "PnL", "TCIM_Vies", "TCIM_Score", "TCIM_Motivos"]],
                height=400,
                column_config={"PnL": st.column_config.NumberColumn(format="$%.4f")},
            )


if __name__ == "__main__":
    main()
