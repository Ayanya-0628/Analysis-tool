import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr, t
import itertools
import io
import concurrent.futures
import os
import time
# 注意：在 Streamlit 中使用 multiprocessing 必须小心，但在函数式结构下通常没问题

# ==========================================
# 0. UI 美化工具
# ==========================================

def styled_tag(text, icon=""):
    st.markdown(f"""
    <div style="
        display: inline-flex;
        align-items: center;
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 15px;
        margin-bottom: 15px;
        margin-top: 5px;
        border: 1px solid #bbdefb;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    ">
        <span style="margin-right: 8px; font-size: 18px;">{icon}</span>
        {text}
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 核心统计工具
# ==========================================

def get_stars(p_value):
    if p_value < 0.001: return '***'
    if p_value < 0.01:  return '**'
    if p_value < 0.05:  return '*'
    return 'ns'

def pairwise_lsd_test_with_mse(stats_df, mse, df_resid, alpha=0.05):
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
    np.fill_diagonal(adj, False)
    cliques = []
    def bron_kerbosch(R, P, X):
        if len(P) == 0 and len(X) == 0:
            cliques.append(R)
            return
        union_px = P.union(X)
        if not union_px: pivot = None
        else: pivot = next(iter(union_px))
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
    
    letters_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if use_uppercase else "abcdefghijklmnopqrstuvwxyz"
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
# 2. Worker 函数 (核心计算逻辑)
# ==========================================

def process_single_target(target, df_data, factors, test_factor, mse_strategy):
    """
    单个指标的计算逻辑，将被多进程或直接调用。
    """
    res = {
        'anova_rows': [],
        'main_effects_rows': [],
        'sliced_comparison_rows': [],
        'error': None
    }
    
    try:
        # 极速预检查：去空值
        current_df = df_data.dropna(subset=[target] + factors).copy()
        
        if current_df.empty or len(current_df) < 3:
            return res 

        group_factors = [f for f in factors if f != test_factor]

        # 1. 全局 ANOVA
        factor_terms = [f'Q("{f}")' for f in factors]
        formula_rhs = " * ".join(factor_terms)
        formula = f"Q('{target}') ~ {formula_rhs}"
        
        model = ols(formula, data=current_df).fit()
        
        global_mse = model.mse_resid
        global_df_resid = model.df_resid
        
        aov_table = sm.stats.anova_lm(model, typ=2)
        aov_table.index = [idx.replace('Q("', '').replace('")', '') for idx in aov_table.index]

        for source, row in aov_table.iterrows():
            if source == 'Residual': continue
            f_str = f"{row['F']:.2f}{get_stars(row['PR(>F)'])}"
            res['anova_rows'].append({
                'Trait': target,
                'Source': source,
                'F_Sig': f_str
            })
        
        # 2. 主效应
        for factor in factors:
            stats = current_df.groupby(factor)[target].agg(['mean', 'std', 'count']).fillna(0)
            
            if mse_strategy == 'oneway':
                try:
                    sub_formula = f"Q('{target}') ~ C(Q('{factor}'))"
                    sub_model = ols(sub_formula, data=current_df).fit()
                    current_mse = sub_model.mse_resid
                    current_df_resid = sub_model.df_resid
                except:
                    current_mse = global_mse
                    current_df_resid = global_df_resid
            else:
                current_mse = global_mse
                current_df_resid = global_df_resid

            if len(stats) < 2:
                letters = {str(k).strip(): 'A' for k in stats.index}
            else:
                pairwise_res = pairwise_lsd_test_with_mse(stats, current_mse, current_df_resid, alpha=0.05)
                letters = solve_clique_cld(stats['mean'], pairwise_res, use_uppercase=True)
            
            for lvl in stats.index:
                lvl_str = str(lvl).strip()
                mean_val = stats.loc[lvl, 'mean']
                res['main_effects_rows'].append({
                    'Factor': factor,
                    'Level': lvl_str,
                    'Trait': target,
                    'Mean_Letter': f"{mean_val:.2f} {letters.get(lvl_str, 'A')}", 
                    'SD': stats.loc[lvl, 'std']
                })

        # 3. 组内比较
        if not group_factors:
            iter_groups = [( "All", current_df )] 
        else:
            iter_groups = current_df.groupby(group_factors)

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
                try:
                    local_formula = f"Q('{target}') ~ C(Q('{test_factor}'))"
                    local_model = ols(local_formula, data=sub_df).fit()
                    local_mse = local_model.mse_resid
                    local_df = local_model.df_resid
                except:
                    local_mse = global_mse
                    local_df = global_df_resid
                
                pairwise_res = pairwise_lsd_test_with_mse(stats, local_mse, local_df, alpha=0.05)
                letters = solve_clique_cld(stats['mean'], pairwise_res, use_uppercase=False)
            
            for lvl in stats.index:
                lvl_str = str(lvl).strip()
                mean_val = stats.loc[lvl, 'mean']
                let = letters.get(lvl_str, 'a')
                
                row = current_info.copy()
                row[test_factor] = lvl_str
                row['Mean'] = mean_val
                row['SD'] = stats.loc[lvl, 'std']
                row['Letter'] = let
                row['Mean_Letter'] = f"{mean_val:.2f} {let}"
                
                res['sliced_comparison_rows'].append(row)
                
    except Exception as e:
        res['error'] = f"指标 '{target}' 出错: {str(e)}"
    
    return res

# ==========================================
# 3. 后端逻辑 (Cached + Adaptive)
# ==========================================

@st.cache_data(show_spinner=False) 
def compute_all_stats(df, factors, valid_targets, test_factor, mse_strategy):
    """
    后端核心计算函数：根据任务量自适应选择串行或多进程并行。
    """
    
    # 准备工作
    work_df = df.copy()
    for f in factors:
        work_df[f] = work_df[f].astype(str).str.strip()
    
    num_tasks = len(valid_targets)
    
    # 【核心优化】
    # 任务数 < 5: 串行计算。避免进程启动开销（约0.5-1秒），让单指标瞬间出结果。
    # 任务数 >= 5: 多进程计算。利用多核 CPU 加速大量计算。
    use_multiprocessing = num_tasks >= 5
    
    max_cpu = os.cpu_count() or 4
    workers = max(1, max_cpu - 1)

    results_list = []
    errors = []

    # 准备任务参数列表
    tasks = []
    for t in valid_targets:
        subset_df = work_df[[t] + factors]
        tasks.append((t, subset_df, factors, test_factor, mse_strategy))

    start_time = time.time()
    
    if use_multiprocessing:
        # 🚀 大量任务：多进程并行
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_single_target, *task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    results_list.append(res)
                except Exception as e:
                    errors.append(f"System Error: {e}")
    else:
        # 🐢 少量任务：直接串行 (速度最快，无启动延迟)
        for task in tasks:
            try:
                res = process_single_target(*task)
                results_list.append(res)
            except Exception as e:
                errors.append(f"Calculation Error: {e}")

    elapsed = time.time() - start_time
    
    return results_list, errors, elapsed

def process_results_to_dfs(results_list, factors, test_factor, valid_targets, work_df):
    """
    将计算结果列表转换为 DataFrame
    """
    all_anova = []
    all_main = []
    all_sliced = []
    errors = []

    for res in results_list:
        if res.get('error'):
            errors.append(res['error'])
        else:
            all_anova.extend(res['anova_rows'])
            all_main.extend(res['main_effects_rows'])
            all_sliced.extend(res['sliced_comparison_rows'])

    final_res = {'errors': errors}

    # 表格 1: ANOVA
    if all_anova:
        final_res['anova_table'] = pd.DataFrame(all_anova).pivot_table(
            index='Source', columns='Trait', values='F_Sig', aggfunc='first'
        )
    else:
        final_res['anova_table'] = pd.DataFrame()

    # 表格 2: Main Effects
    if all_main:
        me_df = pd.DataFrame(all_main)
        me_pivot = me_df.pivot_table(
            index=['Factor', 'Level'], columns='Trait', values=['Mean_Letter'], aggfunc='first'
        )
        final_res['main_effects_table'] = me_pivot.swaplevel(0, 1, axis=1).sort_index(axis=1)
    else:
        final_res['main_effects_table'] = pd.DataFrame()

    # 表格 3: Sliced
    if all_sliced:
        sc_df = pd.DataFrame(all_sliced)
        group_factors = [f for f in factors if f != test_factor]
        pivot_index = group_factors + [test_factor]
        
        # 分列数据
        sc_pivot_sep = sc_df.pivot_table(
            index=pivot_index, columns='Trait', values=['Mean', 'Letter', 'SD'], aggfunc='first'
        )
        sc_pivot_sep = sc_pivot_sep.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
        
        # 排序美化
        sorted_traits = sc_pivot_sep.columns.get_level_values(0).unique()
        new_columns = []
        for t in sorted_traits:
            for val in ['Mean', 'Letter', 'SD']:
                if (t, val) in sc_pivot_sep.columns:
                    new_columns.append((t, val))
        final_res['sliced_table_sep'] = sc_pivot_sep.reindex(columns=new_columns)
        
        # 组合数据
        sc_pivot_comb = sc_df.pivot_table(
            index=pivot_index, columns='Trait', values=['Mean_Letter'], aggfunc='first'
        )
        final_res['sliced_table_comb'] = sc_pivot_comb.swaplevel(0, 1, axis=1).sort_index(axis=1)
    else:
        final_res['sliced_table_sep'] = pd.DataFrame()
        final_res['sliced_table_comb'] = pd.DataFrame()

    # 表格 4: Correlation
    if len(valid_targets) > 1:
        num_df = work_df[valid_targets].apply(pd.to_numeric, errors='coerce')
        corr_df = num_df.corr() 
        pval_df = num_df.corr(method=lambda x, y: pearsonr(x, y)[1]) 
        
        corr_matrix = pd.DataFrame(index=valid_targets, columns=valid_targets)
        for r_idx in valid_targets:
            for c_idx in valid_targets:
                if r_idx == c_idx:
                    corr_matrix.loc[r_idx, c_idx] = "-"
                else:
                    r = corr_df.loc[r_idx, c_idx]
                    p = pval_df.loc[r_idx, c_idx]
                    if pd.isna(r):
                        corr_matrix.loc[r_idx, c_idx] = "NaN"
                    else:
                        corr_matrix.loc[r_idx, c_idx] = f"{r:.2f}{get_stars(p)}"
        final_res['correlation'] = corr_matrix
    else:
        final_res['correlation'] = pd.DataFrame()
        
    return final_res

# ==========================================
# 4. Streamlit 界面
# ==========================================

st.set_page_config(page_title="极速数据分析", layout="wide", page_icon="⚡")
st.title("⚡ 极速统计分析 (Pro)")

# 侧边栏
with st.sidebar:
    styled_tag("数据上传", icon="📂")
    uploaded_file = st.file_uploader("选择 Excel/CSV 文件", type=['xlsx', 'csv'])
    
    styled_tag("因子选择", icon="🧬")
    
    factors = []
    targets = []
    test_factor = None
    mse_strategy = 'oneway' 
    
    df = None
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                if len(sheet_names) > 1:
                    st.success(f"📂 包含 {len(sheet_names)} 个Sheet")
                    selected_sheet = st.selectbox("选择工作表:", sheet_names)
                    df = excel_file.parse(selected_sheet)
                else:
                    df = excel_file.parse(0)
            
            df.columns = df.columns.astype(str)
            all_cols = df.columns.tolist()
            
            st.markdown("---")
            factors = st.multiselect("因子 (X)", all_cols)
            
            if factors:
                default_idx = len(factors) - 1
                test_factor = st.selectbox("比较因子 (用于组内比较)", factors, index=default_idx)
            
            targets = st.multiselect("指标 (Y)", all_cols)
            
            st.markdown("---")
            with st.expander("⚙️ 模型设置 (默认单因素)", expanded=False):
                strategy_label = st.radio(
                    "误差计算方式 (主效应)",
                    ('多因素模型误差(GLM)', '单因素模型误差'),
                    index=1
                )
                mse_strategy = 'full' if '多因素' in strategy_label else 'oneway'
            
        except Exception as e:
            st.error(f"读取错误: {e}")

# 主界面区域
if not (uploaded_file and factors and targets and test_factor):
    with st.expander("ℹ️ 使用说明 (点击展开)", expanded=True):
        col1, col2 = st.columns([0.45, 0.55]) 
        with col1:
            st.markdown("### 📋 数据准备示例")
            demo_data = pd.DataFrame({
               '品种': ['V1', 'V1', 'V1', 'V2'],
                '处理': ['CK', 'CK', 'CK', 'CK'],
                '重复': ['R1', 'R2', 'R3', 'R1'],
                '产量(kg)': [500.2, 520.5, 480.1, 600.5],
                '株高(cm)': [100.5, 105.2, 98.4, 110.2]
            })
            st.dataframe(demo_data, hide_index=True, use_container_width=True)
        with col2:
            st.markdown("""
            ### 🛠️ 操作提示
            1. **左侧上传数据**，选择对应的因子和指标。
            2. **下方点击“启动分析”**。
            3. 结果生成后可下载 Excel。
            
            ### 🚀 性能说明
            - **少量指标 (<5个)**：采用串行极速模式，无启动延迟。
            - **大量指标 (>=5个)**：自动开启多进程并行加速。
            """)
        st.info("👈 请在左侧上传数据并配置参数")
else:
    st.markdown("###") 
    c1, c2, c3 = st.columns([1, 2, 1])
    
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False

    with c2:
        if st.button("🚀 启动分析", type="primary", use_container_width=True):
            st.session_state.run_analysis = True

    if st.session_state.run_analysis:
        st.divider()
        
        valid_targets = []
        for t_col in targets:
            if pd.to_numeric(df[t_col], errors='coerce').notna().sum() > 0:
                valid_targets.append(t_col)
        
        if not valid_targets:
            st.error("所选指标均为空或非数值！")
            st.stop()

        with st.spinner(f"正在分析 {len(valid_targets)} 个指标..."):
            # 传递 df 的副本
            raw_results, exec_errors, elapsed_time = compute_all_stats(
                df, factors, valid_targets, test_factor, mse_strategy
            )

        if exec_errors:
            with st.expander("⚠️ 计算过程中的警告", expanded=False):
                for err in exec_errors:
                    st.warning(err)

        final_res = process_results_to_dfs(raw_results, factors, test_factor, valid_targets, df)
        
        st.success(f"✅ 分析完成！耗时: {elapsed_time:.2f} 秒 (已缓存)")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 组内 (分列)", 
            "📑 组内 (组合)", 
            "🏆 主效应", 
            "🧮 ANOVA", 
            "🔗 相关性"
        ])
        
        with tab1:
            if not final_res['sliced_table_sep'].empty:
                st.dataframe(final_res['sliced_table_sep'], use_container_width=True)
            else: st.warning("无数据")

        with tab2:
            if not final_res['sliced_table_comb'].empty:
                st.dataframe(final_res['sliced_table_comb'], use_container_width=True)
            else: st.warning("无数据")

        with tab3:
            if not final_res['main_effects_table'].empty:
                st.dataframe(final_res['main_effects_table'], use_container_width=True)
            else: st.warning("无数据")

        with tab4:
            if not final_res['anova_table'].empty:
                st.dataframe(final_res['anova_table'], use_container_width=True)
            else: st.warning("无数据")

        with tab5:
            if not final_res['correlation'].empty:
                st.dataframe(final_res['correlation'], use_container_width=True)
            else: st.info("无相关性数据")
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer) as writer:
            if not final_res['sliced_table_sep'].empty: 
                final_res['sliced_table_sep'].to_excel(writer, sheet_name='组内_分列数据')
            if not final_res['sliced_table_comb'].empty: 
                final_res['sliced_table_comb'].to_excel(writer, sheet_name='组内_组合标签')
            if not final_res['main_effects_table'].empty: 
                final_res['main_effects_table'].to_excel(writer, sheet_name='主效应_大写')
            if not final_res['anova_table'].empty: 
                final_res['anova_table'].to_excel(writer, sheet_name='ANOVA')
            if not final_res['correlation'].empty: 
                final_res['correlation'].to_excel(writer, sheet_name='相关分析')
            
        st.download_button(
            "📥 下载完整结果 (Excel)",
            data=buffer.getvalue(),
            file_name=f"Analysis_Result.xlsx",
            mime="application/vnd.ms-excel"
        )
