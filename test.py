import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

# --- Matplotlib 和 Seaborn 美化设置 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings("ignore")

# --- 1. 数据加载与初步探查 ---
def load_and_inspect_data(file_path):
    """
    加载数据并进行初步检查
    
    参数:
        file_path: 数据文件路径
        
    返回:
        data: 加载的数据DataFrame
    """
    data = pd.read_csv(file_path)
    pd.set_option('display.max_columns', None)
    print("数据信息 (data.info()):")
    data.info()
    print("\n描述性统计 (data.describe()):")
    print(data.describe())
    print("\n数据前5行 (data.head()):")
    print(data.head())
    return data

# --- 2. 探索性数据分析 (EDA) ---
def eda(data):
    """
    进行探索性数据分析
    
    参数:
        data: 数据DataFrame
    """
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Heatmap of Features and Target')
    plt.savefig('EDA_相关性热力图.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    target_corr = correlation_matrix.iloc[1:, 0].abs().sort_values(ascending=False)
    top_features = target_corr.head(3).index
    print(f"\n与目标变量最相关的3个特征是: {list(top_features)}")
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(top_features):
        plt.subplot(1, 3, i + 1)
        sns.scatterplot(x=data[feature], y=data.iloc[:, 0])
        plt.title(f'"{feature}" vs "{data.columns[0]}"')
        plt.xlabel(feature)
        plt.ylabel(data.columns[0])
    plt.tight_layout()
    plt.savefig('EDA_高相关特征散点图.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- 3. 数据准备 ---
def prepare_data(data):
    """
    准备数据集
    
    参数:
        data: 数据DataFrame
        
    返回:
        X_train, X_test, y_train, y_test: 划分后的训练集和测试集
    """
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# --- 4. 基线模型比较 ---
def baseline_model_comparison(X_train, y_train, X_test, y_test):
    """
    基线模型比较
    
    参数:
        X_train, y_train: 训练集特征和目标
        X_test, y_test: 测试集特征和目标
        
    返回:
        results_df: 模型评估结果DataFrame
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
        "LightGBM (Baseline)": lgb.LGBMRegressor(random_state=42, verbose=-1)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = {'R²': r2, 'MSE': mse}
        print(f"{name} -> R²: {r2:.4f}, MSE: {mse:.4f}")
    
    results_df = pd.DataFrame(results).T
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    results_df['R²'].plot(kind='bar', ax=ax1, title='R² Score Comparison of Different Models')
    ax1.set_ylabel('R² Score')
    ax1.tick_params(axis='x', rotation=45)
    
    results_df['MSE'].plot(kind='bar', ax=ax2, title='MSE Comparison of Different Models', color='orange')
    ax2.set_ylabel('MSE')
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('模型比较_性能对比.png', dpi=300, bbox_inches='tight')
    plt.show()
    return results_df

# --- 5. LightGBM超参数调优 ---
def hyperparameter_tuning(X_train, y_train):
    """
    LightGBM超参数调优
    
    参数:
        X_train, y_train: 训练集特征和目标
        
    返回:
        final_model: 调优后的最佳模型
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [-1, 10],
        'num_leaves': [31, 50],
        'learning_rate': [0.05, 0.1],
        'feature_fraction': [0.8, 0.9],
        'bagging_fraction': [0.8, 0.9]
    }
    
    lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    grid_search = GridSearchCV(
        estimator=lgb_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='r2'
    )
    grid_search.fit(X_train, y_train)
    
    final_model = grid_search.best_estimator_
    print(f"\n最佳参数组合: {grid_search.best_params_}")
    print(f"交叉验证最佳R²得分: {grid_search.best_score_:.4f}")
    return final_model

# --- 6. 最终模型评估 ---
def final_model_evaluation(final_model, X_test, y_test):
    """
    最终模型评估
    
    参数:
        final_model: 最终模型
        X_test, y_test: 测试集特征和目标
    """
    y_pred = final_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"测试集 MAE: {mae:.4f}")
    print(f"测试集 MSE: {mse:.4f}")
    print(f"测试集 R²: {r2:.4f}")
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, s=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'LightGBM Final Model: Actual vs Predicted (R² = {r2:.4f})')
    plt.axis('equal')
    plt.axis('square')
    plt.text(0.05, 0.95, f'MAE: {mae:.4f}\nMSE: {mse:.4f}\nR²: {r2:.4f}', transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.grid(True, alpha=0.3)
    plt.savefig('LightGBM_真实值vs预测值.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- 7. SHAP可视化解释分析 ---
def shap_analysis(final_model, X_test):
    """
    SHAP可视化解释分析
    
    参数:
        final_model: 最终模型
        X_test: 测试集特征
    """
    # --- 7.1 计算SHAP值 ---
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test)
    print(f"SHAP值形状: {shap_values.shape}")
    print(f"基准值(expected_value): {explainer.expected_value:.4f}")
    
    # --- 7.2 SHAP摘要图系列 ---
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    plt.subplot(2, 2, 1)
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    plt.title("SHAP Summary Plot (Beeswarm)", fontsize=14)
    
    plt.subplot(2, 2, 2)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance", fontsize=14)
    
    plt.subplot(2, 2, 3)
    shap.summary_plot(shap_values, X_test, plot_type="violin", show=False)
    plt.title("SHAP Violin Plot", fontsize=14)
    
    plt.subplot(2, 2, 4)
    shap.plots.heatmap(shap.Explanation(values=shap_values[:20], base_values=explainer.expected_value, data=X_test.values[:20], feature_names=X_test.columns.tolist()), show=False)
    plt.title("SHAP Heatmap (Top 20 Samples)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig('SHAP_综合分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- 7.3 SHAP依赖图 ---
    importances = np.abs(shap_values).mean(0)
    top_features_indices = np.argsort(importances)[-4:]
    top_features = X_test.columns[top_features_indices]
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X_test, interaction_index="auto", show=False)
        plt.title(f'SHAP Dependence Plot: {feature}')
        plt.tight_layout()
        plt.savefig(f'SHAP_依赖图_{feature}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # --- 7.4 SHAP瀑布图 ---
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))
    for i, idx in enumerate([0, len(X_test) // 2]):
        plt.subplot(1, 2, i + 1)
        shap.plots.waterfall(shap.Explanation(values=shap_values[idx], base_values=explainer.expected_value, data=X_test.iloc[idx].values, feature_names=X_test.columns.tolist()), show=False)
        plt.title(f'SHAP Waterfall Plot - Sample {idx}')
    plt.tight_layout()
    plt.savefig('SHAP_瀑布图.png', dpi=500, bbox_inches='tight')
    plt.show()
    
    # --- 7.5 SHAP力图 ---
    shap.initjs()
    sample_indices = [0, len(X_test) // 4, len(X_test) // 2, 3 * len(X_test) // 4, -1]
    for i, idx in enumerate(sample_indices):
        force_plot = shap.force_plot(explainer.expected_value, shap_values[idx], X_test.iloc[idx], matplotlib=True, show=False)
        plt.title(f"SHAP Force Plot - Sample {idx}")
        plt.savefig(f'SHAP_力图_样本_{i}.png', bbox_inches='tight', dpi=300)
        plt.show()
    
    # --- 7.6 SHAP部分依赖图 ---
    from sklearn.inspection import PartialDependenceDisplay
    top_pdp_features = top_features[:2]
    for feature in top_pdp_features:
        plt.figure(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(final_model, X_train, features=[feature], kind='average', grid_resolution=50, ax=plt.gca())
        plt.title(f'Partial Dependence Plot: {feature}')
        plt.xlabel(feature)
        plt.ylabel('Partial Dependence')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'SHAP_PDP_{feature}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 同时绘制部分依赖图和个体条件期望图
    for feature in top_pdp_features:
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(final_model, X_train, features=[feature], kind='both', grid_resolution=50, ax=ax)
        ax.set_title(f'Partial Dependence and ICE Plot: {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Partial Dependence / ICE')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'SHAP_PDP_ICE_{feature}.png', dpi=300, bbox_inches='tight')
        plt.show()

# --- 8. LIME解释分析 ---
def lime_analysis(final_model, X_train, X_test, y_test):
    """
    LIME解释分析
    
    参数:
        final_model: 最终模型
        X_train: 训练集特征
        X_test: 测试集特征
        y_test: 测试集目标
    """
    # --- 8.1 初始化LIME解释器 ---
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=['target'],
        mode='regression',
        discretize_continuous=True,
        random_state=42
    )
    
    # --- 8.2 LIME单个样本分析 ---
    def analyze_sample_with_lime(sample_idx, title_suffix=""):
        exp = lime_explainer.explain_instance(
            X_test.iloc[sample_idx].values,
            final_model.predict,
            num_features=len(X_test.columns)
        )
        explanation = exp.as_list()
        features = [item[0] for item in explanation]
        contributions = [item[1] for item in explanation]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        colors = ['red' if x < 0 else 'green' for x in contributions]
        bars = ax1.barh(features, contributions, color=colors, alpha=0.7)
        ax1.set_xlabel('LIME Feature Contributions')
        ax1.set_title(f'LIME Explanation - Sample {sample_idx}{title_suffix}')
        ax1.grid(True, alpha=0.3)
        
        for bar, contrib in zip(bars, contributions):
            width = bar.get_width()
            ax1.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height() / 2, f'{contrib:.3f}', ha='left' if width >= 0 else 'right', va='center')
        
        exp.save_to_file(f'LIME_解释_样本_{sample_idx}.html')
        
        actual_val = y_test.iloc[sample_idx]
        pred_val = final_model.predict([X_test.iloc[sample_idx].values])[0]
        
        info_text = f"""Sample {sample_idx} Details:
        Actual value: {actual_val:.4f}
        Predicted value: {pred_val:.4f}
        Error: {abs(actual_val - pred_val):.4f}
        
        LIME Explanation (Top 5 Features):"""
        for i, (feature, contrib) in enumerate(explanation[:5]):
            info_text += f"\n{i + 1}. {feature}: {contrib:+.4f}"
        
        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'LIME_解释_样本_{sample_idx}.png', dpi=300, bbox_inches='tight')
        plt.show()
        return explanation
    
    sample_indices = [0, len(X_test) // 4, len(X_test) // 2, 3 * len(X_test) // 4, -1]
    sample_names = ["首个", "1/4处", "中间", "3/4处", "最后"]
    lime_results = {}
    for idx, name in zip(sample_indices, sample_names):
        real_idx = idx if idx >= 0 else len(X_test) + idx
        print(f"\n正在分析{name}样本 (索引: {real_idx})")
        explanation = analyze_sample_with_lime(real_idx, f"({name} sample)")
        lime_results[f"样本_{real_idx}"] = explanation
    
    # --- 8.3 LIME全局特征重要性分析 ---
    n_samples = min(50, len(X_test))
    lime_global_importance = {feature: [] for feature in X_test.columns}
    
    for i in range(n_samples):
        if i % 10 == 0:
            print(f"处理进度: {i}/{n_samples}")
        exp = lime_explainer.explain_instance(X_test.iloc[i].values, final_model.predict, num_features=len(X_test.columns))
        explanation = exp.as_list()
        explanation_dict = dict(explanation)
        for feature in X_test.columns:
            matching_key = None
            for key in explanation_dict.keys():
                if feature in key or key in feature:
                    matching_key = key
                    break
            if matching_key:
                lime_global_importance[feature].append(explanation_dict[matching_key])
            else:
                lime_global_importance[feature].append(0)
    
    lime_avg_importance = {feature: np.mean(np.abs(values)) for feature, values in lime_global_importance.items()}
    
    # 可视化全局重要性对比
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    
    shap_importance = np.abs(shap_values).mean(0)
    shap_importance_dict = dict(zip(X_test.columns, shap_importance))
    
    features_sorted = sorted(shap_importance_dict.keys(), key=lambda x: shap_importance_dict[x], reverse=True)
    shap_vals = [shap_importance_dict[f] for f in features_sorted]
    lime_vals = [lime_avg_importance[f] for f in features_sorted]
    
    ax1.barh(features_sorted, shap_vals, alpha=0.7, label='SHAP', color='blue')
    ax1.set_xlabel('SHAP Importance')
    ax1.set_title('SHAP Global Feature Importance')
    ax1.grid(True, alpha=0.3)
    
    ax2.barh(features_sorted, lime_vals, alpha=0.7, label='LIME', color='orange')
    ax2.set_xlabel('LIME Importance')
    ax2.set_title('LIME Global Feature Importance')
    ax2.grid(True, alpha=0.3)
    
    ax3.scatter(shap_vals, lime_vals, alpha=0.7, s=60)
    for i, feature in enumerate(features_sorted):
        ax3.annotate(feature, (shap_vals[i], lime_vals[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('SHAP Importance')
    ax3.set_ylabel('LIME Importance')
    ax3.set_title('SHAP vs LIME Importance Correlation')
    ax3.grid(True, alpha=0.3)
    
    correlation = np.corrcoef(shap_vals, lime_vals)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('SHAP_vs_LIME_全局重要性对比.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- 9. SHAP vs LIME 深度对比分析 ---
def shap_vs_lime_comparison(features_sorted, shap_importance_dict, lime_avg_importance):
    """
    SHAP vs LIME深度对比分析
    
    参数:
        features_sorted: 按SHAP重要性排序的特征列表
        shap_importance_dict: SHAP重要性字典
        lime_avg_importance: LIME平均重要性字典
    """
    fig = plt.figure(figsize=(20, 16))
    
    # 重要性对比 - 使用极坐标创建雷达图
    features_radar = list(features_sorted[:8])
    shap_vals_radar = [shap_importance_dict[f] for f in features_radar]
    lime_vals_radar = [lime_avg_importance[f] for f in features_radar]
    
    shap_vals_radar = np.array(shap_vals_radar) / max(shap_vals_radar)
    lime_vals_radar = np.array(lime_vals_radar) / max(lime_vals_radar)
    
    angles = [n / len(features_radar) * 2 * np.pi for n in range(len(features_radar))]
    angles += angles[:1]
    
    shap_vals_radar = np.concatenate([shap_vals_radar, [shap_vals_radar[0]]])
    lime_vals_radar = np.concatenate([lime_vals_radar, [lime_vals_radar[0]]])
    
    ax1 = plt.subplot(2, 2, 1, projection='polar')
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.plot(angles, shap_vals_radar, 'o-', linewidth=2, label='SHAP', color='blue')
    ax1.fill(angles, shap_vals_radar, alpha=0.25, color='blue')
    ax1.plot(angles, lime_vals_radar, 'o-', linewidth=2, label='LIME', color='orange')
    ax1.fill(angles, lime_vals_radar, alpha=0.25, color='orange')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(features_radar)
    ax1.set_ylim(0, 1)
    ax1.set_title('Feature Importance Radar Chart Comparison', y=1.08)
    ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax1.grid(True)
    
    # 一致性分析
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(shap_vals, lime_vals, alpha=0.7, s=80)
    z = np.polyfit(shap_vals, lime_vals, 1)
    p = np.poly1d(z)
    ax2.plot(shap_vals, p(shap_vals), "r--", alpha=0.8)
    ax2.set_xlabel('SHAP Importance')
    ax2.set_ylabel('LIME Importance')
    ax2.set_title(f'Consistency Analysis (R = {correlation:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # 方法比较矩阵
    ax4 = plt.subplot(2, 2, 3)
    methods_comparison = np.array([
        [5, 3, 5, 5, 5, 4, 5],
        [3, 4, 3, 3, 3, 2, 5]
    ])
    im = ax4.imshow(methods_comparison, cmap='RdYlBu_r', aspect='auto')
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(['Theory', 'Efficiency', 'Scope', 'Stability', 'Visualization', 'Interaction', 'Compatibility'], rotation=45, ha='right')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['SHAP', 'LIME'])
    ax4.set_title('Method Comparison Score (1-5 points)')
    
    for i in range(2):
        for j in range(7):
            text = ax4.text(j, i, methods_comparison[i, j], ha="center", va="center", color="white", fontweight='bold')
    plt.colorbar(im, ax=ax4)
    plt.tight_layout()
    plt.savefig('SHAP_LIME_终极对比分析.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # --- 1. 数据加载与初步探查 ---
    file_path = '公众号Python机器学习ml-2025-7-26数据.csv'  # 请替换为您的数据文件路径
    data = load_and_inspect_data(file_path)
    
    # --- 2. 探索性数据分析 (EDA) ---
    eda(data)
    
    # --- 3. 数据准备 ---
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # --- 4. 基线模型比较 ---
    baseline_model_comparison(X_train, y_train, X_test, y_test)
    
    # --- 5. LightGBM超参数调优 ---
    final_model = hyperparameter_tuning(X_train, y_train)
    
    # --- 6. 最终模型评估 ---
    final_model_evaluation(final_model, X_test, y_test)
    
    # --- 7. SHAP可视化解释分析 ---
    shap_analysis(final_model, X_test)
    
    # --- 8. LIME解释分析 ---
    lime_analysis(final_model, X_train, X_test, y_test)
    
    # --- 9. SHAP vs LIME 深度对比分析 ---
    shap_vs_lime_comparison(features_sorted, shap_importance_dict, lime_avg_importance)