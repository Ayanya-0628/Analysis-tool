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

def solve_clique_cld(means, pairwise_data, use_uppercase=False):
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
    
    if use_uppercase:
        letters_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    else:
        letters_list = "abcdefghijklmnopqrstuvwxyz"
        
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
# 2. 核心流程：定制化列结构 (Updated)
# ==========================================

def run_comprehensive_analysis(df, factors, targets, test_factor):
    results = {}
    
    # 数据清洗
    work_df = df.copy()
    for f in factors:
        work_df[f] = work_df[f].astype(str).str.strip()
        
    group_factors = [f for f in factors if f != test_factor]
    
    anova_rows = []
    main_effects_rows = []
    sliced_comparison_rows = []
    
    for target in targets:
        try:
            # --- A. ANOVA ---
            formula = f"{target} ~ {' * '.join(factors)}"
            model = ols(formula, data=work_df).fit()
            
            aov_table = sm.stats.anova_lm(model, typ=2)
            global_mse = model.mse_resid
            global_df_resid = model.df_resid
            
            for source, row in aov_table.iterrows():
                if source == 'Residual': continue
                f_str = f"{row['F']:.2f}{get_stars(row['PR(>F)'])}"
                anova_rows.append({
                    'Trait': target,
                    'Source': source,
                    'F_Sig': f_str
                })
            
            # --- B. 主效应 (大写字母, 无 SD) ---
            for factor in factors:
                stats = work_df.groupby(factor)[target].agg(['mean', 'std', 'count']).fillna(0)
                
                if len(stats) < 2:
                    letters = {str(k).strip(): 'A' for k in stats.index}
                else:
                    pairwise_res = pairwise_lsd_test_with_mse(stats, global_mse, global_df_resid, alpha=0.05)
                    letters = solve_clique_cld(stats['mean'], pairwise_res, use_uppercase=True)
                
                for lvl in stats.index:
                    mean_val = stats.loc[lvl, 'mean']
                    sd_val = stats.loc[lvl, 'std']
                    lvl_str = str(lvl).strip()
                    let = letters.get(lvl_str, 'A')
                    
                    # 组合: Mean + Letter (大写)
                    mean_let = f"{mean_val:.2f} {let}"
                    
                    main_effects_rows.append({
                        'Factor': factor,
                        'Level': lvl_str,
                        'Trait': target,
                        'Mean_Letter': mean_let, 
                        # SD 这里计算了但不放入 Pivot
                        'SD': sd_val 
                    })

            # --- C. 切片比较 (Mean, Letter, SD) ---
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
                    letters = solve_clique_cld(stats['mean'], pairwise_res, use_uppercase=False)
                
                for lvl in stats.index:
                    mean_val = stats.loc[lvl, 'mean']
                    sd_val = stats.loc[lvl, 'std']
                    lvl_str = str(lvl).strip()
                    let = letters.get(lvl_str, 'a')
                    
                    row = current_info.copy()
                    row[test_factor] = lvl_str
                    
                    # 基础数据
                    row['Mean'] = mean_val
                    row['SD'] = sd_val
                    row['Letter'] = let
                    
                    # 组合数据
                    row['Mean_Letter'] = f"{mean_val:.2f} {let}"
                    
                    sliced_comparison_rows.append(row)
                    
        except Exception as e:
            pass

    # --- D. 生成表格 (Pivot) ---
    
    # 1. ANOVA 表
    if anova_rows:
        anova_df = pd.DataFrame(anova_rows)
        results['anova_table'] = anova_df.pivot_table(
            index='Source', columns='Trait', values='F_Sig', aggfunc='first'
        )
    else:
        results['anova_table'] = pd.DataFrame()

    # 2. 主效应表 (Mean_Letter ONLY)
    if main_effects_rows:
        me_df = pd.DataFrame(main_effects_rows)
        # 仅 Pivot Mean_Letter，移除 SD
        me_pivot = me_df.pivot_table(
            index=['Factor', 'Level'], 
            columns='Trait', 
            values=['Mean_Letter'], 
            aggfunc='first'
        )
        # 调整列顺序
        results['main_effects_table'] = me_pivot.swaplevel(0, 1, axis=1).sort_index(axis=1)
    else:
        results['main_effects_table'] = pd.DataFrame()

    # 3. 切片比较 (两种格式)
    if sliced_comparison_rows:
        sc_df = pd.DataFrame(sliced_comparison_rows)
        pivot_index = group_factors + [test_factor]
        
        # 格式一：Mean, Letter, SD 三个分开 (指定顺序)
        sc_pivot_sep = sc_df.pivot_table(
            index=pivot_index, 
            columns='Trait', 
            values=['Mean', 'Letter', 'SD'], 
            aggfunc='first'
        )
        # 交换层级: (Trait, Type)
        sc_pivot_sep = sc_pivot_sep.swaplevel(0, 1, axis=1)
        # 先按指标排序
        sc_pivot_sep = sc_pivot_sep.sort_index(axis=1, level=0)
        
        # 【关键修改】强制重排 Level 1 的顺序: Mean -> Letter -> SD
        # 获取所有排好序的指标
        sorted_traits = sc_pivot_sep.columns.get_level_values(0).unique()
        # 构建新的列索引顺序
        new_columns = []
        for t in sorted_traits:
            for val in ['Mean', 'Letter', 'SD']:
                new_columns.append((t, val))
        
        # 应用重排
        results['sliced_table_sep'] = sc_pivot_sep.reindex(columns=new_columns)
        
        # 格式二：仅 Mean + Letter 组合
        sc_pivot_comb = sc_df.pivot_table(
            index=pivot_index, 
            columns='Trait', 
            values=['Mean_Letter'], 
            aggfunc='first'
        )
        results['sliced_table_comb'] = sc_pivot_comb.swaplevel(0, 1, axis=1).sort_index(axis=1)
    else:
        results['sliced_table_sep'] = pd.DataFrame()
        results['sliced_table_comb'] = pd.DataFrame()
        
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

st.set_page_config(page_title="论文数据助手", layout="wide", page_icon="📊")
st.title("📊 简单的数据分析")
# ==================== 新增部分开始 ====================
with st.expander("ℹ️ 使用说明 & 数据格式示例 (点击展开)"):
    st.markdown("""
    **数据准备指南：**
    1. 请准备 Excel (.xlsx) 或 CSV 文件。
    2. **第一行**必须是列名（如：品种、处理、产量、株高）。
    3. 数据应为**长格式 (Long Format)**，即每一行代表一个重复或样本。
    """)
    
    # 创建一个虚拟的示例数据
    demo_data = pd.DataFrame({
       '品种': ['V1', 'V1', 'V1' ],
        '处理': ['CK', 'CK', 'CK'],
        '重复': ['R1', 'R2', 'R3'],
        '产量(kg)': [500.2, 520.5, 480.1],
        '株高(cm)': [100.5, 105.2, 98.4]
    })
    
    # 展示表格
    st.table(demo_data)
    st.caption("注：因子列（如品种、处理）可以是文字或数字；指标列（如产量）必须是数字。")
# ==================== 新增部分结束 ====================

st.info(...) # 原来的 info 代码

# ... (后面的侧边栏代码不变) ...
st.info("""
✅ **组内比较**：
   - **格式 A**：Mean, Letter, SD 严格按此顺序分列 (方便作图)
   - **格式 B**：Mean+Letter 组合 (方便制表)
✅ **主效应**：
   - 仅保留 **Mean + 大写Letter** (移除 SD 列)
""")

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
            
            run_btn = st.button("生成分析结果", type="primary")
            
        except Exception as e:
            st.error(f"读取错误: {e}")

if uploaded_file and factors and targets and test_factor and run_btn:
    st.divider()
    with st.spinner("正在生成多格式数据..."):
        try:
            res = run_comprehensive_analysis(df, factors, targets, test_factor)
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 组内 (分列-作图)", 
                "📑 组内 (组合-制表)", 
                "🏆 主效应 (大写)", 
                "🧮 ANOVA", 
                "🔗 相关性"
            ])
            
            with tab1:
                st.subheader(f"1. 组内比较 - 分列数据 (按 {test_factor})")
                st.caption("顺序：Mean -> Letter -> SD | 适合导入 Origin/GraphPad")
                if not res['sliced_table_sep'].empty:
                    st.dataframe(res['sliced_table_sep'], use_container_width=True)
                else:
                    st.warning("无数据")

            with tab2:
                st.subheader(f"2. 组内比较 - 组合标签 (按 {test_factor})")
                st.caption("结构：Mean_Letter | 适合直接粘贴到 Word 表格")
                if not res['sliced_table_comb'].empty:
                    st.dataframe(res['sliced_table_comb'], use_container_width=True)
                else:
                    st.warning("无数据")

            with tab3:
                st.subheader("3. 主效应比较 (Uppercase)")
                st.caption("结构：Mean_Letter Only (无 SD)")
                if not res['main_effects_table'].empty:
                    st.dataframe(res['main_effects_table'], use_container_width=True)
                else:
                    st.warning("无数据")

            with tab4:
                st.subheader("4. 方差分析 (F-value)")
                if not res['anova_table'].empty:
                    st.dataframe(res['anova_table'], use_container_width=True)
                else:
                    st.warning("无数据")

            with tab5:
                st.subheader("5. 相关性矩阵")
                st.dataframe(res['correlation'], use_container_width=True)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer) as writer:
                if not res['sliced_table_sep'].empty: 
                    res['sliced_table_sep'].to_excel(writer, sheet_name='组内_分列数据')
                if not res['sliced_table_comb'].empty: 
                    res['sliced_table_comb'].to_excel(writer, sheet_name='组内_组合标签')
                if not res['main_effects_table'].empty: 
                    res['main_effects_table'].to_excel(writer, sheet_name='主效应_大写')
                if not res['anova_table'].empty: 
                    res['anova_table'].to_excel(writer, sheet_name='ANOVA')
                if not res['correlation'].empty: 
                    res['correlation'].to_excel(writer, sheet_name='相关分析')
                
            st.download_button(
                "📥 下载完整数据包 (Excel)",
                data=buffer.getvalue(),
                file_name="Analysis_Formatted.xlsx",
                mime="application/vnd.ms-excel"
            )
            
        except Exception as e:
            st.error(f"分析失败: {e}")
            import traceback
            st.text(traceback.format_exc())





