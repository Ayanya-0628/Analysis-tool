import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr, t
import itertools
import io

# ==========================================
# 1. 核心统计工具 (保持不变)
# ==========================================

def get_stars(p_value):
    """将P值转换为星号"""
    if p_value < 0.001: return '***'
    if p_value < 0.01:  return '**'
    if p_value < 0.05:  return '*'
    return 'ns'

def pairwise_lsd_test_with_mse(stats_df, mse, df_resid, alpha=0.05):
    """
    使用外部传入的 MSE (来自全模型) 进行 Fisher's LSD 检验
    """
    results = []
    group_names = stats_df.index.tolist()
    
    # 两两比较
    for g1, g2 in itertools.combinations(group_names, 2):
        m1, n1 = stats_df.loc[g1]
        m2, n2 = stats_df.loc[g2]
        
        diff = m1 - m2
        # LSD 标准误
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
    """
    最大团算法生成字母标记
    """
    groups = [str(g).strip() for g in means.index.tolist()]
    n = len(groups)
    g_to_i = {g: i for i, g in enumerate(groups)}
    
    # 初始化：默认全连接
    adj = np.ones((n, n), dtype=bool) 
    
    # 根据显著性断开连接
    if pairwise_data:
        for row in pairwise_data:
            g1 = str(row[0]).strip()
            g2 = str(row[1]).strip()
            reject = row[4]
            if reject: 
                if g1 in g_to_i and g2 in g_to_i:
                    i, j = g_to_i[g1], g_to_i[g2]
                    adj[i, j] = False
                    adj[j, i] = False

    # 移除自环 (关键修复)
    np.fill_diagonal(adj, False)

    # Bron-Kerbosch 最大团算法
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
    
    # 分配字母
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
        key_str = str(original_index[i]).strip()
        final_res[key_str] = l_str
        
    return final_res

# ==========================================
# 2. 核心流程：全能分析 (主效应 + 切片)
# ==========================================

def run_comprehensive_analysis(df, factors, targets, test_factor):
    results = {}
    
    # 0. 数据清洗
    work_df = df.copy()
    for f in factors:
        work_df[f] = work_df[f].astype(str).str.strip()
        
    # 分组因子 (用于切片比较的背景因子)
    group_factors = [f for f in factors if f != test_factor]
    
    # 容器
    anova_rows = []
    main_effects_rows = []
    sliced_comparison_rows = []
    
    for target in targets:
        try:
            # --- A. 全模型 ANOVA & 全局误差 ---
            formula = f"{target} ~ {' * '.join(factors)}"
            model = ols(formula, data=work_df).fit()
            
            # 记录 ANOVA
            aov_table = sm.stats.anova_lm(model, typ=2)
            for source, row in aov_table.iterrows():
                if source == 'Residual': continue
                anova_rows.append({
                    'Trait': target,
                    'Source': source,
                    'Df': int(row['df']),
                    'F-value': row['F'],
                    'P-value': row['PR(>F)'],
                    'Signif': get_stars(row['PR(>F)'])
                })
            
            # 获取全局 Pooled MSE (用于所有后续比较，保证检验效能一致)
            global_mse = model.mse_resid
            global_df_resid = model.df_resid
            
            # --- B. 主效应比较 (Main Effects) ---
            # 遍历每一个因子，计算整体均值差异
            for factor in factors:
                # 1. 计算该因子的边际均值
                stats = work_df.groupby(factor)[target].agg(['mean', 'count'])
                
                # 2. LSD 比较
                if len(stats) < 2:
                    letters = {str(k).strip(): 'a' for k in stats.index}
                else:
                    pairwise_res = pairwise_lsd_test_with_mse(stats, global_mse, global_df_resid, alpha=0.05)
                    letters = solve_clique_cld(stats['mean'], pairwise_res)
                
                # 3. 记录
                for lvl in stats.index:
                    mean_val = stats.loc[lvl, 'mean']
                    lvl_str = str(lvl).strip()
                    let = letters.get(lvl_str, 'a')
                    
                    main_effects_rows.append({
                        'Factor': factor,
                        'Level': lvl_str,
                        'Trait': target,
                        'Mean': mean_val,
                        'Letter': let,
                        'Label': f"{mean_val:.2f} {let}"
                    })

            # --- C. 组内切片比较 (Sliced Comparison) ---
            # 逻辑：固定 group_factors，比较 test_factor
            
            # 确定遍历的分组
            if not group_factors:
                iter_groups = [( "All", work_df )] # 单因素情况
            else:
                iter_groups = work_df.groupby(group_factors)

            for group_keys, sub_df in iter_groups:
                if not isinstance(group_keys, tuple): group_keys = (group_keys,)
                
                # 基础信息
                current_info = {'Trait': target}
                if group_factors:
                    for k, val in zip(group_factors, group_keys):
                        current_info[k] = str(val)
                
                # 计算待测因子的均值
                stats = sub_df.groupby(test_factor)[target].agg(['mean', 'count'])
                
                # 比较
                if len(stats) < 2:
                    letters = {str(k).strip(): 'a' for k in stats.index}
                else:
                    pairwise_res = pairwise_lsd_test_with_mse(stats, global_mse, global_df_resid, alpha=0.05)
                    letters = solve_clique_cld(stats['mean'], pairwise_res)
                
                # 记录
                for lvl in stats.index:
                    mean_val = stats.loc[lvl, 'mean']
                    lvl_str = str(lvl).strip()
                    let = letters.get(lvl_str, 'a')
                    
                    row = current_info.copy()
                    row[test_factor] = lvl_str
                    row['Mean'] = mean_val
                    row['Letter'] = let
                    row['Label'] = f"{mean_val:.2f} {let}"
                    sliced_comparison_rows.append(row)
                    
        except Exception as e:
            # print(f"Error in {target}: {e}")
            pass

    # --- D. 整理输出 ---
    
    # 1. ANOVA
    results['anova'] = pd.DataFrame(anova_rows)
    
    # 2. 主效应
    if main_effects_rows:
        me_df = pd.DataFrame(main_effects_rows)
        results['main_effects'] = me_df
        # Pivot: Index=Factor+Level, Col=Trait
        results['main_effects_pivot'] = me_df.pivot_table(
            index=['Factor', 'Level'], columns='Trait', values='Label', aggfunc='first'
        ).reset_index()
    else:
        results['main_effects'] = pd.DataFrame()
        results['main_effects_pivot'] = pd.DataFrame()

    # 3. 切片比较
    if sliced_comparison_rows:
        sliced_df = pd.DataFrame(sliced_comparison_rows)
        # 整理列顺序
        cols = group_factors + [test_factor, 'Trait', 'Mean', 'Letter', 'Label']
        final_cols = [c for c in cols if c in sliced_df.columns]
        results['sliced_comparison'] = sliced_df[final_cols]
        
        # Pivot
        pivot_index = group_factors + [test_factor]
        results['sliced_pivot'] = sliced_df.pivot_table(
            index=pivot_index, columns='Trait', values='Label', aggfunc='first'
        ).reset_index()
    else:
        results['sliced_comparison'] = pd.DataFrame()
        results['sliced_pivot'] = pd.DataFrame()
        
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

st.set_page_config(page_title="农业统计平台 (全能版)", layout="wide", page_icon="🌾")
st.title("🌾 简单的数据分析")
st.info("✅ 功能：方差分析 | 主效应多重比较 | 组内比较 (固定主因子比较副因子) | 相关性分析")

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
            st.write("👉 **步骤 1: 选择所有参与因子的列**")
            factors = st.multiselect("因子 (X)", all_cols)
            
            if factors:
                st.markdown("👉 **步骤 2: 选择用于组内比较的因子**")
                st.caption("例如：选“处理”，则分析会展示“品种A下的处理差异”、“品种B下的处理差异”。")
                default_idx = len(factors) - 1
                test_factor = st.selectbox("比较因子 (Test Factor)", factors, index=default_idx)
            
            st.markdown("👉 **步骤 3: 选择指标**")
            targets = st.multiselect("指标 (Y)", all_cols)
            
            run_btn = st.button("开始全能分析", type="primary")
            
        except Exception as e:
            st.error(f"读取错误: {e}")

if uploaded_file and factors and targets and test_factor and run_btn:
    st.divider()
    with st.spinner("正在进行全方位分析..."):
        try:
            res = run_comprehensive_analysis(df, factors, targets, test_factor)
            
            # 使用 Tabs 分开展示
            tab1, tab2, tab3, tab4 = st.tabs(["📊 主效应比较", "🔍 组内切片比较", "📑 方差分析", "🔗 相关性"])
            
            with tab1:
                st.subheader("1. 主效应比较 (Main Effects)")
                st.caption("展示每个因子（如不同品种、不同处理）的整体均值差异，忽略其他因子的影响。")
                if not res['main_effects_pivot'].empty:
                    st.dataframe(res['main_effects_pivot'], use_container_width=True)
                    with st.expander("查看详细数据 (含 Mean 和 Letter 独立列)"):
                        st.dataframe(res['main_effects'], use_container_width=True)
                else:
                    st.warning("无数据生成")

            with tab2:
                st.subheader(f"2. 组内切片比较 (按 {test_factor} 进行比较)")
                group_others = [f for f in factors if f != test_factor]
                st.caption(f"展示在固定背景 ({' + '.join(group_others) if group_others else '无'}) 下，{test_factor} 的差异。")
                
                if not res['sliced_pivot'].empty:
                    st.dataframe(res['sliced_pivot'], use_container_width=True)
                    with st.expander("查看详细数据 (含 Mean 和 Letter 独立列)"):
                        st.dataframe(res['sliced_comparison'], use_container_width=True)
                else:
                    st.warning("无数据生成")
                
            with tab3:
                st.subheader("3. 全模型方差分析表")
                st.dataframe(res['anova'], use_container_width=True)
                
            with tab4:
                st.subheader("4. 相关性分析")
                st.dataframe(res['correlation'], use_container_width=True)
            
            # 导出
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer) as writer:
                # 主效应
                if not res['main_effects_pivot'].empty: 
                    res['main_effects_pivot'].to_excel(writer, sheet_name='主效应_宽表', index=False)
                if not res['main_effects'].empty: 
                    res['main_effects'].to_excel(writer, sheet_name='主效应_明细', index=False)
                
                # 切片比较
                if not res['sliced_pivot'].empty: 
                    res['sliced_pivot'].to_excel(writer, sheet_name='组内切片_宽表', index=False)
                if not res['sliced_comparison'].empty: 
                    res['sliced_comparison'].to_excel(writer, sheet_name='组内切片_明细', index=False)
                
                # 其他
                if not res['anova'].empty: 
                    res['anova'].to_excel(writer, sheet_name='ANOVA', index=False)
                if not res['correlation'].empty: 
                    res['correlation'].to_excel(writer, sheet_name='相关分析')
                
            st.download_button(
                "📥 下载全能分析报告 (Excel)",
                data=buffer.getvalue(),
                file_name="Comprehensive_Analysis_Report.xlsx",
                mime="application/vnd.ms-excel"
            )
            
        except Exception as e:
            st.error(f"分析失败: {e}")
            import traceback

            st.text(traceback.format_exc())
