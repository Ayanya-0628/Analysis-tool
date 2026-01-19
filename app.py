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

# ==========================================
# 1. 核心统计工具 (保持不变)
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
# 2. 并行化核心逻辑 (保持不变)
# ==========================================

def process_single_target(target, df_data, factors, test_factor, mse_strategy):
    res = {
        'anova_rows': [],
        'main_effects_rows': [],
        'sliced_comparison_rows': [],
        'error': None
    }
    
    try:
        current_df = df_data.dropna(subset=[target] + factors).copy()
        
        if current_df.empty or len(current_df) < 3:
            return res 

        group_factors = [f for f in factors if f != test_factor]

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
                pairwise_res = pairwise_lsd_test_with_mse(stats, global_mse, global_df_resid, alpha=0.05)
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

def run_parallel_analysis(df, factors, targets, test_factor, mse_strategy):
    results = {}
    errors = []
    
    work_df = df.copy()
    for f in factors:
        work_df[f] = work_df[f].astype(str).str.strip()
    
    valid_targets = []
    for t_col in targets:
        work_df[t_col] = pd.to_numeric(work_df[t_col], errors='coerce')
        if not work_df[t_col].dropna().empty:
            valid_targets.append(t_col)
        else:
            errors.append(f"指标 '{t_col}' 全为空值，跳过。")

    all_anova = []
    all_main = []
    all_sliced = []

    max_workers = os.cpu_count() or 4
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.write(f"🚀 正在启动 {max_workers} 个 CPU 核心进行并行计算...")
    
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_target = {
            executor.submit(process_single_target, t, work_df[[t] + factors], factors, test_factor, mse_strategy): t 
            for t in valid_targets
        }
        
        completed_count = 0
        total_tasks = len(valid_targets)
        
        for future in concurrent.futures.as_completed(future_to_target):
            t_name = future_to_target[future]
            try:
                data = future.result()
                if data['error']:
                    errors.append(data['error'])
                else:
                    all_anova.extend(data['anova_rows'])
                    all_main.extend(data['main_effects_rows'])
                    all_sliced.extend(data['sliced_comparison_rows'])
            except Exception as exc:
                errors.append(f"{t_name} 进程崩溃: {exc}")
            
            completed_count += 1
            if total_tasks > 0:
                progress = completed_count / total_tasks
                progress_bar.progress(progress)
            status_text.write(f"正在处理: {completed_count}/{total_tasks} ({t_name})")

    elapsed_time = time.time() - start_time
    status_text.success(f"✅ 分析完成！耗时: {elapsed_time:.2f} 秒")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()

    if all_anova:
        results['anova_table'] = pd.DataFrame(all_anova).pivot_table(
            index='Source', columns='Trait', values='F_Sig', aggfunc='first'
        )
    else:
        results['anova_table'] = pd.DataFrame()

    if all_main:
        me_df = pd.DataFrame(all_main)
        me_pivot = me_df.pivot_table(
            index=['Factor', 'Level'], columns='Trait', values=['Mean_Letter'], aggfunc='first'
        )
        results['main_effects_table'] = me_pivot.swaplevel(0, 1, axis=1).sort_index(axis=1)
    else:
        results['main_effects_table'] = pd.DataFrame()

    if all_sliced:
        sc_df = pd.DataFrame(all_sliced)
        group_factors = [f for f in factors if f != test_factor]
        pivot_index = group_factors + [test_factor]
        
        sc_pivot_sep = sc_df.pivot_table(
            index=pivot_index, columns='Trait', values=['Mean', 'Letter', 'SD'], aggfunc='first'
        )
        sc_pivot_sep = sc_pivot_sep.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
        
        sorted_traits = sc_pivot_sep.columns.get_level_values(0).unique()
        new_columns = []
        for t in sorted_traits:
            for val in ['Mean', 'Letter', 'SD']:
                if (t, val) in sc_pivot_sep.columns:
                    new_columns.append((t, val))
        results['sliced_table_sep'] = sc_pivot_sep.reindex(columns=new_columns)
        
        sc_pivot_comb = sc_df.pivot_table(
            index=pivot_index, columns='Trait', values=['Mean_Letter'], aggfunc='first'
        )
        results['sliced_table_comb'] = sc_pivot_comb.swaplevel(0, 1, axis=1).sort_index(axis=1)
    else:
        results['sliced_table_sep'] = pd.DataFrame()
        results['sliced_table_comb'] = pd.DataFrame()

    if len(valid_targets) > 1:
        corr_df = work_df[valid_targets].corr() 
        pval_df = work_df[valid_targets].corr(method=lambda x, y: pearsonr(x, y)[1]) 
        
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
        results['correlation'] = corr_matrix
    else:
        results['correlation'] = pd.DataFrame()

    results['errors'] = errors
    return results

# ==========================================
# 3. Streamlit 界面 (布局优化版)
# ==========================================

st.set_page_config(page_title="数据分析", layout="wide", page_icon="⚡")
st.title("🌾 水稻科研数据分析")

# 侧边栏
with st.sidebar:
    st.header("1. 数据上传")
    uploaded_file = st.file_uploader("上传 Excel/CSV", type=['xlsx', 'csv'])
    
    st.header("2. 因子选择")
    factors = []
    targets = []
    test_factor = None
    mse_strategy = 'oneway' 
    
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
                    "误差计算方式",
                    ('多因素模型误差(GLM)', '单因素模型误差'),
                    index=1,
                    help="多因素：剥离其他因子干扰，MSE小。\n单因素：完全基于原始数据波动，MSE大。"
                )
                mse_strategy = 'full' if '多因素' in strategy_label else 'oneway'
            
        except Exception as e:
            st.error(f"读取错误: {e}")

# 主界面区域 (布局修复)
with st.expander("ℹ️ 使用说明(点击展开)", expanded=True):
    # 🟢 调整比例：左边 0.45，右边 0.55，让表格有更多空间
    col1, col2 = st.columns([0.45, 0.55]) 
    with col1:
        st.markdown("### 📋 数据准备示例")
        demo_data = pd.DataFrame({
           '品种': ['V1', 'V1', 'V2', 'V2'],
            '处理': ['CK', 'TR', 'CK', 'TR'],
            '产量': [500.2, 520.5, 600.5, 620.1],
        })
        # 🟢 关键修改：
        # 1. 恢复 st.dataframe
        # 2. use_container_width=True: 强制填满列宽，消灭右侧留白
        # 3. hide_index=True: 隐藏索引列，节省空间
        st.dataframe(demo_data, hide_index=True, use_container_width=True)
    with col2:
        st.markdown("""
        ### 🛠️ 操作提示
        1. **左侧上传数据**，选择对应的因子和指标。
        2. **下方点击“启动分析”**，无需在侧边栏底部寻找按钮。
        3. 结果生成后可直接下载 Excel。
        """)

if uploaded_file and factors and targets and test_factor:
    st.markdown("###") 
    run_btn = st.button("🚀 立即启动并行分析", type="primary", use_container_width=True)

    if run_btn:
        st.divider()
        res = run_parallel_analysis(df, factors, targets, test_factor, mse_strategy)
            
        if res.get('errors'):
            with st.expander("⚠️ 部分指标分析失败", expanded=False):
                for err in res['errors']:
                    st.warning(err)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 组内 (分列)", 
            "📑 组内 (组合)", 
            "🏆 主效应", 
            "🧮 ANOVA", 
            "🔗 相关性"
        ])
        
        with tab1:
            st.subheader(f"1. 组内比较 - 分列数据")
            if not res['sliced_table_sep'].empty:
                st.dataframe(res['sliced_table_sep'], width='stretch')
            else:
                st.warning("无数据")

        with tab2:
            st.subheader(f"2. 组内比较 - 组合标签")
            if not res['sliced_table_comb'].empty:
                st.dataframe(res['sliced_table_comb'], width='stretch')
            else:
                st.warning("无数据")

        with tab3:
            title_suffix = "(基于单因素误差)" if mse_strategy == 'oneway' else "(基于全模型误差)"
            st.subheader(f"3. 主效应比较 {title_suffix}")
            if not res['main_effects_table'].empty:
                st.dataframe(res['main_effects_table'], width='stretch')
            else:
                st.warning("无数据")

        with tab4:
            st.subheader("4. 方差分析 (F-value)")
            if not res['anova_table'].empty:
                st.dataframe(res['anova_table'], width='stretch')
            else:
                st.warning("无数据")

        with tab5:
            st.subheader("5. 相关性矩阵")
            if not res['correlation'].empty:
                st.dataframe(res['correlation'], width='stretch')
            else:
                st.info("数据不足以计算相关性")
        
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
            "📥 下载完整结果 (Excel)",
            data=buffer.getvalue(),
            file_name=f"Analysis_{mse_strategy}.xlsx",
            mime="application/vnd.ms-excel"
        )
elif uploaded_file:
    st.info("👈 请在左侧侧边栏选择【因子】和【指标】以激活分析按钮")
else:
    st.info("👈 请在左侧上传数据文件")
