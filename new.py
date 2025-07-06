import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib

# 设置中文显示（适用于有中文标题/标签）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# -------------------- 原始实验数据 --------------------
t = np.array([0.25, 0.5, 1.0, 2.5, 3.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0])
C = np.array([14.0, 19.5, 37.5, 61.5, 71.5, 69.2, 59.5, 50.1, 36.6, 27.5, 21.5, 16.4, 11.5, 8.30, 5.90])

# -------------------- 三指数模型定义 --------------------
def three_exp_model(t, N, L, M, ka, alpha, beta):
    return N * np.exp(-ka * t) + L * np.exp(-alpha * t) + M * np.exp(-beta * t)

# -------------------- 方法1：残差法 --------------------
# Step 1：拟合 β 相
def single_exponential(t, M, beta):
    return M * np.exp(-beta * t)

t_beta = t[-5:]
C_beta = C[-5:]
params_beta, _ = curve_fit(single_exponential, t_beta, C_beta, p0=[10, 0.1])
M_res, beta_res = params_beta
C_beta_fit = single_exponential(t, M_res, beta_res)

# Step 2：拟合 α 相
C_res_alpha = C - C_beta_fit
def single_exponential_alpha(t, L, alpha):
    return L * np.exp(-alpha * t)

params_alpha, _ = curve_fit(single_exponential_alpha, t, C_res_alpha, p0=[20, 0.5])
L_res, alpha_res = params_alpha
C_alpha_fit = single_exponential_alpha(t, L_res, alpha_res)

# Step 3：拟合 ka 相
C_res_ka = C - C_beta_fit - C_alpha_fit
def single_exponential_ka(t, N, ka):
    return N * np.exp(-ka * t)

valid_idx = C_res_ka > 0
t_valid_ka = t[valid_idx]
C_valid_ka = C_res_ka[valid_idx]

params_ka, _ = curve_fit(single_exponential_ka, t_valid_ka, C_valid_ka, p0=[5, 0.2])
N_res, ka_res = params_ka
C_ka_fit = single_exponential_ka(t, N_res, ka_res)

# -------------------- 人工调参 --------------------
N = -235  # 初始中央室药物量
L = 192.95  # 初始周边室药物量
M = 49.89  # 周边室药物量
Ka = 0.373  # 吸收速率常数
alpha = 0.218  # 分布相混合一级速率常数
beta = 0.036  # 消除相混合一级速率常数

# -------------------- 模型浓度预测 --------------------
t_fit = np.linspace(0, t[-1], 300)
C_model = three_exp_model(t, N, L, M, Ka, alpha, beta)
C_fit = three_exp_model(t_fit, N, L, M, Ka, alpha, beta)

# -------------------- 可视化：拟合曲线 vs 原始数据 + 各相成分 --------------------
plt.figure(figsize=(10, 6))
plt.plot(t_fit, C_fit, label='总模型拟合', color='blue')
plt.scatter(t, C, label='实验数据', color='red')
plt.xlabel("时间 (h)")
plt.ylabel("浓度 (mg/L)")
plt.title("三指数模型拟合结果")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#  加入各成分的拟合曲线
plt.figure(figsize=(10, 6))
plt.plot(t_fit, C_fit, label='总模型拟合', color='blue')
plt.scatter(t, C, label='实验数据', color='red')
plt.plot(t, C_ka_fit, '--', label='ka 相（快吸收）', color='purple')
plt.plot(t, C_alpha_fit, '--', label='α 相（分布）', color='orange')
plt.plot(t, C_beta_fit, '--', label='β 相（消除）', color='green')

plt.xlabel("时间 (h)")
plt.ylabel("浓度 (mg/L)")
plt.title("三指数模型拟合结果及各相分量")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------- 残差分析图  --------------------
residuals = C - C_model
plt.figure(figsize=(10, 4))
plt.plot(t, residuals, 'o-', color='gray')
plt.axhline(0, linestyle='--', color='black')
plt.title("拟合残差图")
plt.xlabel("时间 (h)")
plt.ylabel("残差 (mg/L)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------- AUC 与模型评价指标 --------------------
AUC_model = N / Ka + L / alpha + M / beta
AUC_exp = np.trapezoid(C, t)
relative_error = (AUC_model - AUC_exp) / AUC_exp * 100
rmse = np.sqrt(np.mean((C_model - C) ** 2))
mape = np.mean(np.abs((C - C_model) / C)) * 100
corr_coef = np.corrcoef(C, C_model)[0, 1]

#  拟合优度 R²
ss_res = np.sum((C - C_model) ** 2)
ss_tot = np.sum((C - np.mean(C)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# -------------------- 输出结果 --------------------
print("\n【原始残数法拟合参数】")
print(f"N = {N_res:.2f}, ka = {ka_res:.4f} h⁻¹")
print(f"L = {L_res:.2f}, alpha = {alpha_res:.4f} h⁻¹")
print(f"M = {M_res:.2f}, beta = {beta_res:.4f} h⁻¹")

print("\n【人工调整后的拟合参数】")
print(f"N = {N:.2f}, ka = {Ka:.4f} h⁻¹")
print(f"L = {L:.2f}, alpha = {alpha:.4f} h⁻¹")
print(f"M = {M:.2f}, beta = {beta:.4f} h⁻¹")

print("\n【模型评价指标】")
print(f"AUC（模型） = {AUC_model:.2f} mg·h/L")
print(f"AUC（实验） = {AUC_exp:.2f} mg·h/L")
print(f"AUC相对误差: {relative_error:.2f}%")
print(f"RMSE = {rmse:.2f} mg/L")
print(f"MAPE = {mape:.2f}%")
print(f"相关系数 R = {corr_coef:.4f}")
print(f"拟合优度 R² = {r_squared:.4f}")

# -------------------- 推导药动学参数 --------------------
numerator = L * beta * (Ka - alpha) + M * alpha * (Ka - beta)
denominator = L * (Ka- alpha) + M * (Ka - beta)
K21 = numerator / denominator
K10 = (alpha * beta) / K21
K12 = alpha + beta - K21 - K10

t_half_a = 0.693 / Ka
t_half_alpha = 0.693 / alpha
t_half_beta = 0.693 / beta

print("\n【推导出的药动学参数】")
print(f"K21 = {K21:.4f} h⁻¹")
print(f"K10 = {K10:.4f} h⁻¹")
print(f"K12 = {K12:.4f} h⁻¹")

print("\n【半衰期】")
print(f"吸收相 t₁/₂(a) = {t_half_a:.2f} h")
print(f"分布相 t₁/₂(α) = {t_half_alpha:.2f} h")
print(f"消除相 t₁/₂(β) = {t_half_beta:.2f} h")


from scipy import stats
from scipy.signal import savgol_filter
# -------------------- 方法2：对数线性分段法 --------------------
# 原始时间与浓度数据
t_all_raw = np.array([0.25, 0.5, 1.0, 2.5, 3.5, 5.0, 7.5, 10, 15, 20, 25, 30, 40, 50, 60])
C_all_raw = np.array([14.0, 19.5, 37.5, 61.5, 71.5, 69.2, 59.5, 50.1, 36.6, 27.5, 21.5, 16.4, 11.5, 8.3, 5.9])

# 平滑处理并计算导数，自动寻找峰值位置
C_smooth = savgol_filter(C_all_raw, window_length=5, polyorder=2)
dCdt = np.gradient(C_smooth, t_all_raw)
peak_index = np.argmax(C_smooth)

# 自动划分三段
t1 = t_all_raw[:peak_index + 1]
C1 = C_all_raw[:peak_index + 1]
t2 = t_all_raw[peak_index + 1:peak_index + 6]  # 通常5个点足够拟合
C2 = C_all_raw[peak_index + 1:peak_index + 6]
t3 = t_all_raw[peak_index + 6:]
C3 = C_all_raw[peak_index + 6:]

# β相拟合
logC3 = np.log10(C3)
slope3, intercept3, r3, _, _ = stats.linregress(t3, logC3)
beta = -slope3 * 2.303
M = 10**intercept3

# α相拟合
Cr1 = C2 - M * np.exp(-beta * t2)
logCr1 = np.log10(Cr1)
slope2, intercept2, r2, _, _ = stats.linregress(t2, logCr1)
alpha = -slope2 * 2.303
L = 10**intercept2

# Ka相拟合
Cr1_ka = C1 - M * np.exp(-beta * t1)
Cr1_extrapolated_ka = L * np.exp(-alpha * t1)
Cr2_ka = Cr1_extrapolated_ka - Cr1_ka
logCr2_ka = np.log10(np.abs(Cr2_ka))
slope1, intercept1, r1, _, _ = stats.linregress(t1, logCr2_ka)
Ka = -slope1 * 2.303
N = -10**intercept1

# 模型参数输出
print("\nβ相参数:")
print("回归方程: logC = {:.5f} * t + {:.5f}".format(slope3, intercept3))
print("相关系数(R²): {:.4f}".format(r3**2))
print("β = {:.4f} h⁻¹".format(beta))
print("M = {:.2f} mg/L".format(M))

print("\nα相参数:")
print("回归方程: logCr₁ = {:.4f} * t + {:.4f}".format(slope2, intercept2))
print("相关系数(R²): {:.4f}".format(r2**2))
print("α = {:.4f} h⁻¹".format(alpha))
print("L = {:.2f} mg/L".format(L))

print("\nKa相参数:")
print("回归方程: log|Cr₂| = {:.4f} * t + {:.4f}".format(slope1, intercept1))
print("相关系数(R²): {:.4f}".format(r1**2))
print("Ka = {:.4f} h⁻¹".format(Ka))
print("N = {:.2f} mg/L".format(N))

# 其他药动学参数
numerator = L * beta * (Ka - alpha) + M * alpha * (Ka - beta)
denominator = L * (Ka - alpha) + M * (Ka - beta)
K21 = numerator / denominator
K10 = (alpha * beta) / K21
K12 = alpha + beta - K21 - K10

t_half_a = 0.693 / Ka
t_half_alpha = 0.693 / alpha
t_half_beta = 0.693 / beta
AUC_model = N / Ka + L / alpha + M / beta

# 实际 AUC（梯形法）
t_exp = np.concatenate([t1, t2, t3])
C_exp = np.concatenate([C1, C2, C3])
sorted_indices = np.argsort(t_exp)
t_exp_sorted = t_exp[sorted_indices]
C_exp_sorted = C_exp[sorted_indices]
AUC_actual = np.trapezoid(C_exp_sorted, t_exp_sorted)

# 残差分析图
C_model = three_exp_model(t, N, L, M, Ka, alpha, beta)
residuals = C - C_model
plt.figure(figsize=(10, 4))
plt.plot(t, residuals, 'o-', color='gray')
plt.axhline(0, linestyle='--', color='black')
plt.title("拟合残差图")
plt.xlabel("时间 (h)")
plt.ylabel("残差 (mg/L)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 模型评估部分
# 计算模型预测值在原始时间点的浓度
C_model_points = N * np.exp(-Ka * t_exp_sorted) + L * np.exp(-alpha * t_exp_sorted) + M * np.exp(-beta * t_exp_sorted)

# 计算评价指标
# 1. 相对误差
relative_error = (AUC_model - AUC_actual) / AUC_actual * 100

# 2. RMSE (均方根误差)
rmse = np.sqrt(np.mean((C_model_points - C_exp_sorted)**2))

# 3. MAPE (平均绝对百分比误差)
mape = np.mean(np.abs((C_exp_sorted - C_model_points) / C_exp_sorted)) * 100

# 4. 相关系数
corr_coef = np.corrcoef(C_exp_sorted, C_model_points)[0, 1]

# 5.拟合优度 R²
ss_res = np.sum((C_exp_sorted - C_model_points) ** 2)
ss_tot = np.sum((C_exp_sorted - np.mean(C_exp_sorted)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# 输出
print("\n药代动力学参数:")
print("K21 = {:.4f} h⁻¹".format(K21))
print("K10 = {:.4f} h⁻¹".format(K10))
print("K12 = {:.4f} h⁻¹".format(K12))
print("\n半衰期:")
print("吸收相 t₁/₂(a) = {:.2f} h".format(t_half_a))
print("分布相 t₁/₂(α) = {:.2f} h".format(t_half_alpha))
print("消除相 t₁/₂(β) = {:.2f} h".format(t_half_beta))
print("\nAUC（模型） = {:.2f} mg·h/L".format(AUC_model))
print("实际 AUC（梯形法） = {:.2f} mg·h/L".format(AUC_actual))
print(f"AUC相对误差: {relative_error:.2f}%")
print(f"RMSE: {rmse:.2f} mg/L")
print(f"MAPE: {mape:.2f}%")
print(f"预测值与实测值相关系数: {corr_coef:.4f}")
print(f"拟合优度 R² = {r_squared:.4f}")
# ---------------- 图1：划分后数据点 + 拟合曲线 ----------------
t_fit = np.linspace(0, t_all_raw[-1], 300)
C_fit = N * np.exp(-Ka * t_fit) + L * np.exp(-alpha * t_fit) + M * np.exp(-beta * t_fit)

plt.figure(figsize=(10, 6))
plt.plot(t_fit, C_fit, 'b-', label='模型拟合曲线')
plt.scatter(t1, C1, color='red', label='Ka 相数据')
plt.scatter(t2, C2, color='green', label='α 相数据')
plt.scatter(t3, C3, color='blue', label='β 相数据')
plt.xlabel("时间 (h)")
plt.ylabel("浓度 (mg/L)")
plt.title("三指数模型拟合结果（自动划分）")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------- 图2：每段对数线性回归 ----------------
plt.figure(figsize=(10, 6))

# 计算模型预测值
C_model = N * np.exp(-Ka * t_all_raw) + L * np.exp(-alpha * t_all_raw) + M * np.exp(-beta * t_all_raw)
logC_model = np.log10(C_model)

# Ka 段对数线
plt.plot(t1, logCr2_ka, 'ro', label='log|Cr2|')
plt.plot(t1, slope1 * t1 + intercept1, 'r--', label='Ka相拟合')

# α 段对数线
plt.plot(t2, logCr1, 'go', label='logCr1')
plt.plot(t2, slope2 * t2 + intercept2, 'g--', label='α相拟合')

# β 段对数线
plt.plot(t3, logC3, 'bo', label='logC')
plt.plot(t3, slope3 * t3 + intercept3, 'b--', label='β相拟合')

# 模型 logC 曲线
plt.plot(t_all_raw, logC_model, color='gray', linestyle='-', label='模型 logC 曲线')

plt.xlabel("时间 (h)")
plt.ylabel("log C")
plt.title("三指数模型对数线性拟合")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

