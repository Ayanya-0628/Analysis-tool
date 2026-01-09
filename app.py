import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr, t
import itertools
import io

# ==========================================
# 1. 核心统计工具
# ==========================================

def get_stars(p_value):
    """将P值转换为星号"""
    if p_value < 0.001: return '***'
    if p_value < 0.01:  return '**'
    if p_value < 0.05:  return '*'
    return 'ns'

def pairwise_lsd_test_with_mse(stats_df, mse, df_resid, alpha=0.05):
    """Fisher's LSD 检验"""
    results = []
    group_names = stats_df.index.tolist()
    
    for g1, g2 in itertools.combinations(group_names, 2):
        m1, n1 = stats_df.loc[g1, 'mean'], stats_df.loc[g1, 'count']
        m2, n2 = stats_df.loc[g2, 'mean'], stats_df.loc[g2, 'count']
        
        diff = m1 - m2
        se = np.sqrt(mse * (1/n1 + 1/n2))
        
        if se <= 1e-10: 
            p_val = 1.0
        else:
            t_stat = abs(diff) / se
            p_val = 2 * (1 - t.cdf(t_stat, df_resid))
        
        reject = p_val < alpha
        results.append([g1, g2, diff, p_val, reject])
        
    return results

def solve_clique_cld(means, pairwise_data):
    """最大团算法生成字母标记"""
    groups = [str(g).strip() for g in means.index.tolist()]
    n = len(groups)
    g_to_i = {g: i for i, g in enumerate(groups)}
    adj = np.ones((n, n), dtype=bool) 
    
    if pairwise_data:
        for row in pairwise_data:
            g1, g2, reject = str(row[0]).strip(), str(row[1]).strip(), row[4]
            if reject: 
                if g1 in g_to_i and g2 in g_to_i:
                    i, j = g_to_i[g1], g_to_i[g2]
                    adj[i, j] = False
                    adj[j, i] = False

    np.fill_diagonal(adj, False) # 移除自环

    cliques = []
    def bron_kerbosch(R, P, X):
        if len(P) == 0 and len(X) == 0:
            cliques.append(R)
            return
        pivot = next(iter(P.union(X))) if P.union(X) else None
        neighbors_pivot = {idx for idx in range(n) if adj[pivot, idx]} if pivot is not None else set()
        for v in list(P - neighbors_pivot):
            neighbors_v = {idx for idx in range(n) if adj[v, idx]}
            bron_kerbosch(R.union({v}), P.intersection(neighbors_v), X.intersection(neighbors_v))
            P.remove(v)
            X.add(v)

    bron_kerbosch(set(), set(range(n)), set())
    
    clique_means = []
    for clq in cliques:
        avg_mean = np.mean([means.iloc[i] for i in clq])
        clique_means.append((avg_mean, clq))
    
    clique_means.sort(key=lambda x: x[0], reverse=True)
    
    letters_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    group_letters = {i: "" for i in range(n)}
    
    for idx, (avg, clq) in enumerate(clique_means):
        char = letters_list[idx] if idx < len(letters_list) else "?"
        for node_idx in clq:
            group_letters[node_idx] += char
            
    final_res = {}
    original_index = means.index.tolist()
    for i in range(n):
        l_str = "".join(sorted(group_letters[i]))
        final_res[str(original_index[i]).strip()] = l_str
        
    return final_res

# ==========================================
# 2. 核心流程：生成多级表头数据
# ==========================================

def run_comprehensive_analysis(df, factors, targets, test_factor):
    results = {}
    
    # 数据清洗
    work_df = df.copy()
    for f in factors:
        work_df[f] = work_df[f].astype(str).str.strip()
        
    group_factors = [f for f in factors if f != test_factor]
    
    # 容器
    anova_rows = []
    main_effects_rows = []
    sliced_comparison_rows = []
    
    for target in targets:
        try:
            # --- A. 全模型 ANOVA ---
            formula = f"{target} ~ {' * '.join(factors)}"
            model = ols(formula, data=work_df).fit()
            
            aov_table = sm.stats.anova_lm(model, typ=2)
            global_mse = model.mse_resid
            global_df_resid = model.df_resid
            
            # 记录 ANOVA (F值+星号)
            for source, row in aov_table.iterrows():
                if source == 'Residual': continue
                f_str = f"{row['F']:.2f}{get_stars(row['PR(>F)'])}"
                anova_rows.append({
                    'Trait': target,
                    'Source': source,
                    'F_Sig': f_str
                })
            
            # --- B. 主效应 (分离 Mean, SD, Letter) ---
            for factor in factors:
                stats = work_df.groupby(factor)[target].agg(['mean', 'std', 'count']).fillna(0)
                
                if len(stats) < 2:
                    letters = {str(k).strip(): 'a' for k in stats.index}
                else:
                    pairwise_res = pairwise_lsd_test_with_mse(stats, global_mse, global_df_resid, alpha=0.05)
                    letters = solve_clique_cld(stats['mean'], pairwise_res)
                
                for lvl in stats.index:
                    mean_val = stats.loc[lvl, 'mean']
                    sd_val = stats.loc[lvl, 'std']
                    lvl_str = str(lvl).strip()
                    let = letters.get(lvl_str, 'a')
                    
                    main_effects_rows.append({
                        'Factor': factor,
                        'Level': lvl_str,
                        'Trait': target,
                        'Mean': mean_val,
                        'SD': sd_val,
                        'Letter': let
                    })

            # --- C. 切片比较 (分离 Mean, SD, Letter) ---
            if not group_factors:
                iter_groups = [( "All", work_df )] 
            else:
                iter_groups = work_df.groupby(group_factors)

            for group_keys, sub_df in iter_groups:
                if not isinstance(group_keys, tuple): group_keys = (group_keys,)
                
                current_info = {'Trait': target}
                if group_factors:
                    for k, val in zip(group_factors, group_keys):
                        current_info[k] = str(val)
                
                stats = sub_df.groupby(test_factor)[target].agg(['mean', 'std', 'count']).fillna(0)
                
                if len(stats) < 2:
                    letters = {str(k).strip(): 'a' for k in stats.index}
                else:
                    pairwise_res = pairwise_lsd_test_with_mse(stats, global_mse, global_df_resid, alpha=0.05)
                    letters = solve_clique_cld(stats['mean'], pairwise_res)
                
                for lvl in stats.index:
                    mean_val = stats.loc[lvl, 'mean']
                    sd_val = stats.loc[lvl, 'std']
                    lvl_str = str(lvl).strip()
                    let = letters.get(lvl_str, 'a')
                    
                    row = current_info.copy()
                    row[test_factor] = lvl_str
                    # 独立列
                    row['Mean'] = mean_val
                    row['SD'] = sd_val
                    row['Letter'] = let
                    sliced_comparison_rows.append(row)
                    
        except Exception as e:
            pass

    # --- D. 构建三线表 (Multi-Index Pivot) ---
    
    # 1. ANOVA 表 (Index=Source, Columns=Trait, Value=F_Sig)
    if anova_rows:
        anova_df = pd.DataFrame(anova_rows)
        results['anova_table'] = anova_df.pivot_table(
            index='Source', columns='Trait', values='F_Sig', aggfunc='first'
        )
    else:
        results['anova_table'] = pd.DataFrame()

    # 2. 主效应表 (Index=Factor+Level, Columns=Trait -> [Mean, SD, Letter])
    if main_effects_rows:
        me_df = pd.DataFrame(main_effects_rows)
        # 核心：多值 Pivot
        me_pivot = me_df.pivot_table(
            index=['Factor', 'Level'], 
            columns='Trait', 
            values=['Mean', 'SD', 'Letter'], 
            aggfunc='first'
        )
        # 调整列顺序：让指标在最上层，Mean/SD/Letter 在下层
        # swaplevel 将 (Mean, Yield) 变成 (Yield, Mean)
        # sort_index 让同一个指标的 Mean/SD/Letter 挨在一起
        results['main_effects_table'] = me_pivot.swaplevel(0, 1, axis=1).sort_index(axis=1)
    else:
        results['main_effects_table'] = pd.DataFrame()

    # 3. 切片比较表 (Index=Group+TestFactor, Columns=Trait -> [Mean, SD, Letter])
    if sliced_comparison_rows:
        sc_df = pd.DataFrame(sliced_comparison_rows)
        pivot_index = group_factors + [test_factor]
        
        sc_pivot = sc_df.pivot_table(
            index=pivot_index, 
            columns='Trait', 
            values=['Mean', 'SD', 'Letter'], 
            aggfunc='first'
        )
        results['sliced_table'] = sc_pivot.swaplevel(0, 1, axis=1).sort_index(axis=1)
    else:
        results['sliced_table'] = pd.DataFrame()
        
    # 4. 相关性
    if len(targets) > 1:
        corr_matrix = pd.DataFrame(index=targets, columns=targets)
        for t1 in targets:
            for t2 in targets:
                if t1 == t2:
                    corr_matrix.loc[t1, t2] = "-"
                else:
                    valid = df[[t1, t2]].dropna()
                    if len(valid) > 2:
                        r, p = pearsonr(valid[t1], valid[t2])
                        corr_matrix.loc[t1, t2] = f"{r:.2f}{get_stars(p)}"
                    else:
                        corr_matrix.loc[t1, t2] = "NaN"
        results['correlation'] = corr_matrix
    else:
        results['correlation'] = pd.DataFrame()

    return results

# ==========================================
# 3. Streamlit 界面
# ==========================================

st.set_page_config(page_title="论文三线表生成器 (分离版)", layout="wide", page_icon="📝")
st.title("📝 论文数据生成器 (Separated Mean/SD)")
st.info("✅ 已满足：组内比较、主效应均已将 **Mean(均值)**、**SD(标准差)**、**Letter(字母)** 拆分为独立列，且按指标分组排列。")

with st.sidebar:
    st.header("1. 数据上传")
    uploaded_file = st.file_uploader("上传 Excel/CSV", type=['xlsx', 'csv'])
    
    st.header("2. 参数设置")
    factors = []
    targets = []
    test_factor = None
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            df.columns = df.columns.astype(str)
            all_cols = df.columns.tolist()
            
            st.markdown("---")
            factors = st.multiselect("因子 (X)", all_cols)
            
            if factors:
                default_idx = len(factors) - 1
                test_factor = st.selectbox("比较因子 (用于组内比较)", factors, index=default_idx)
            
            targets = st.multiselect("指标 (Y)", all_cols)
            
            run_btn = st.button("生成三线表数据", type="primary")
            
        except Exception as e:
            st.error(f"读取错误: {e}")

if uploaded_file and factors and targets and test_factor and run_btn:
    st.divider()
    with st.spinner("正在构建多级表头数据..."):
        try:
            res = run_comprehensive_analysis(df, factors, targets, test_factor)
            
            # 使用 Tabs 展示
            tab1, tab2, tab3, tab4 = st.tabs(["📝 组内比较 (切片)", "📝 主效应比较", "📝 方差分析 (F值)", "🔗 相关性"])
            
            with tab1:
                st.subheader(f"Table 1. 组内差异 (按 {test_factor})")
                st.caption("列结构：指标 -> [Letter, Mean, SD]")
                if not res['sliced_table'].empty:
                    st.dataframe(res['sliced_table'], use_container_width=True)
                else:
                    st.warning("无数据")

            with tab2:
                st.subheader("Table 2. 主效应差异")
                if not res['main_effects_table'].empty:
                    st.dataframe(res['main_effects_table'], use_container_width=True)
                else:
                    st.warning("无数据")

            with tab3:
                st.subheader("Table 3. 方差分析结果")
                st.caption("F-value + Significance Stars")
                if not res['anova_table'].empty:
                    st.dataframe(res['anova_table'], use_container_width=True)
                else:
                    st.warning("无数据")

            with tab4:
                st.subheader("Figure 1. 相关性矩阵")
                st.dataframe(res['correlation'], use_container_width=True)
            
            # 导出逻辑
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer) as writer:
                # 写入所有生成的表格
                if not res['sliced_table'].empty: 
                    res['sliced_table'].to_excel(writer, sheet_name='Table_组内比较')
                
                if not res['main_effects_table'].empty: 
                    res['main_effects_table'].to_excel(writer, sheet_name='Table_主效应')
                
                if not res['anova_table'].empty: 
                    res['anova_table'].to_excel(writer, sheet_name='Table_方差分析')
                
                if not res['correlation'].empty: 
                    res['correlation'].to_excel(writer, sheet_name='相关分析')
                
            st.download_button(
                "📥 下载三线表数据 (Excel)",
                data=buffer.getvalue(),
                file_name="Three_Line_Tables_Separated.xlsx",
                mime="application/vnd.ms-excel"
            )
            
        except Exception as e:
            st.error(f"分析失败: {e}")
            import traceback
            st.text(traceback.format_exc())
