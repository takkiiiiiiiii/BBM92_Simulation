import numpy as np
from scipy.integrate import quad
from scipy.special import erfc, erf
# from scipy.stats import lognorm


def transmitivity_pdf(
        eta, mu_x, mu_y, sigma_x, sigma_y, zenith_angle_rad,
        w_L, w_Leq, tau_zen, varphi_mod, wavelength, h_OGS,
        h_atm, Cn2_profile, a):

    eta_l = compute_atm_loss(tau_zen, zenith_angle_rad)

    sigma_R_squared = rytov_variance(
        wavelength, zenith_angle_rad, h_OGS, h_atm, Cn2_profile)

    A_mod = mod_jitter(mu_x, mu_y, sigma_x, sigma_y, w_L, w_Leq, a)

    mu = (sigma_R_squared/2) * (1 + 2 * varphi_mod**2)

    term1 = (varphi_mod**2) / (2 * (A_mod * eta_l)**(varphi_mod**2))
    term2 = eta**(varphi_mod**2-1)
    term3 = erfc(
        (np.log(eta / (A_mod * eta_l)) + mu)
        / (np.sqrt(2) * np.sqrt(sigma_R_squared))
        )
    term4 = np.exp(
        (sigma_R_squared / 2) * varphi_mod**2 * (1 + varphi_mod**2)
    )
    # print(term1, term2, term3, term4)
    eta = term1 * term2 * term3 * term4

    return eta


def compute_atm_loss(tau_zen, zenith_angle_rad):
    tau_atm = tau_zen ** (1 / np.cos(zenith_angle_rad))
    if tau_atm > 1:
        raise ValueError("Atmospheric loss is larger than 1")

    return tau_atm


def rytov_variance(len_wave, zenith_angle_rad, h_OGS, h_atm, Cn2_profile):
    k = 2 * np.pi / len_wave
    sec_zenith = 1 / np.cos(zenith_angle_rad)

    def integrand(h):
        return Cn2_profile(h) * (h - h_OGS)**(5/6)

    integral, _ = quad(integrand, h_OGS, h_atm)

    sigma_R_squared = 2.25 * (k)**(7/6) * sec_zenith**(11/6) * integral

    return sigma_R_squared


def mod_jitter(mu_x, mu_y, sigma_x, sigma_y, w_L, w_Leq, a):
    A_0 = compute_A0(a, w_L)

    varphi_x = sigma_to_variance(sigma_x, w_Leq)
    varphi_y = sigma_to_variance(sigma_y, w_Leq)
    sigma_mod = compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y)
    varphi_mod = sigma_to_variance(sigma_mod, w_Leq)

    term1 = 1 / (varphi_mod**2)
    term2 = 1 / (2 * varphi_x**2)
    term3 = 1 / (2 * varphi_y**2)
    term4 = mu_x**2 / (2 * sigma_x**2 * varphi_x**2)
    term5 = mu_y**2 / (2 * sigma_y**2 * varphi_y**2)
    G = np.exp(term1 - term2 - term3 - term4 - term5)

    A_mod = A_0 * G

    return A_mod


def compute_A0(a, w_L):
    """_summary_

    Args:
        a (_type_): asparture radius
        w_L (_type_): beam waist at distance L
    """
    nu = (np.sqrt(np.pi) * a) / (np.sqrt(2) * w_L)
    A_0 = erf(nu)**2

    return A_0


def sigma_to_variance(sigma, w_Leq):
    return w_Leq/(2*sigma)


def compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y):
    numerator = (
        3 * mu_x**2 * sigma_x**4 +
        3 * mu_y**2 * sigma_y**4 +
        sigma_x**6 +
        sigma_y**6
    )
    sigma_mod_value = (numerator / 2) ** (1/3)
    return sigma_mod_value


def Cn2_profile(h, v_wind=21, Cn2_0=1e-13):
    term1 = 0.00594 * (v_wind/27)**2 * (1e-5 * h)**10 * np.exp(-h/1000)
    term2 = 2.7e-16 * np.exp(-h / 1500)
    term3 = Cn2_0 * np.exp(-h / 100)

    return term1 + term2 + term3


def qber_loss(e_0, p_dark, e_pol, p_AP, eta, n_s):
    denominator = (
        e_0 * (p_dark*(1+p_AP)) + (e_pol+e_0*p_AP) * (1-np.exp(-n_s*eta))
    )
    numerator = (p_dark*(1+p_AP)) + (1-np.exp(-n_s*eta)) * (1+p_AP)
    qber = denominator/numerator

    return qber


def weather_condition(tau_zen):
    if tau_zen == 0.91:
        return 'Clear sky', 23000  # H_atm for clear sky
    elif tau_zen == 0.85:
        return 'Slightly hazy', 15000  # H_atm for slightly hazy
    elif tau_zen == 0.75:
        return 'Noticeably hazy', 10000  # H_atm for noticeably hazy
    elif tau_zen == 0.53:
        return 'Poor visibility', 5000  # H_atm for poor visibility
    else:
        return 'Unknown condition', 10000  # Default value


def compute_slant_distance(h_s, H_g, zenith_angle_rad):
    return (h_s - H_g)/np.cos(zenith_angle_rad)


def equivalent_beam_width_squared(a, w_L):
    # w_L: beam radius at receiver before aperture clipping
    nu = (np.sqrt(np.pi) * a) / (np.sqrt(2) * w_L)
    numerator = np.sqrt(np.pi) * erf(nu)
    denominator = 2 * nu * np.exp(-nu**2)
    return w_L**2 * (numerator / denominator)


def compute_avg_qber(
        sigma_theta_x, sigma_theta_y, slant_distance,
        mu_x, mu_y, zenith_angle_rad, h_OGS, h_atm, w_L, tau_zen,
        Cn2_profile, a, e_0, p_dark, e_pol, p_AP, n_s, wavelength):
    sigma_x = sigma_theta_x * slant_distance
    sigma_y = sigma_theta_y * slant_distance

    w_Leq_squared = equivalent_beam_width_squared(a, w_L)
    w_Leq = np.sqrt(w_Leq_squared)
    sigma_mod = compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y)
    varphi_mod = sigma_to_variance(sigma_mod, w_Leq)

    # def integrand(eta):
    #     term_1 = transmitivity_pdf(
    #         eta, mu_x, mu_y, sigma_x, sigma_y, zenith_angle_rad,
    #         w_L, w_Leq, tau_zen, varphi_mod, wavelength, h_OGS,
    #         h_atm, Cn2_profile, a)
    #     term_2 = qber_loss(e_0, p_dark, e_pol, p_AP, eta, n_s)

    #     return term_1 * term_2

    def integrand_2(eta):
        term_1 = transmitivity_pdf(
            eta, mu_x, mu_y, sigma_x, sigma_y, zenith_angle_rad,
            w_L, w_Leq, tau_zen, varphi_mod, wavelength, h_OGS,
            h_atm, Cn2_profile, a)
        term_2 = compute_yield(eta, n_s, p_dark, p_AP)

        return term_1 * term_2

    def integrand_3(eta):
        term_1 = transmitivity_pdf(
            eta, mu_x, mu_y, sigma_x, sigma_y, zenith_angle_rad,
            w_L, w_Leq, tau_zen, varphi_mod, wavelength, h_OGS,
            h_atm, Cn2_profile, a)
        term_2 = (
            e_0 * (p_dark*(1+p_AP))
            + (e_pol+e_0*p_AP) * (1-np.exp(-n_s*eta))
        )

        return term_1 * term_2

    # res, _ = quad(integrand, 0, np.inf)

    avg_yield, _ = quad(integrand_2, 0, np.inf, limit=100, epsabs=1e-9, epsrel=1e-9)

    avg_err_bits, _ = quad(integrand_3, 0, np.inf, limit=100, epsabs=1e-9, epsrel=1e-9)

    return avg_err_bits/avg_yield, avg_yield

# 使用しない
def compute_Q_1_e_1_ex(
        sigma_theta_x, sigma_theta_y, slant_distance,
        mu_x, mu_y, zenith_angle_rad, h_OGS, h_atm, w_L, tau_zen,
        Cn2_profile, a, e_0, p_dark, e_pol, p_AP, n_s, n_d, wavelength):
    sigma_x = sigma_theta_x * slant_distance
    sigma_y = sigma_theta_y * slant_distance

    w_Leq_squared = equivalent_beam_width_squared(a, w_L)
    w_Leq = np.sqrt(w_Leq_squared)
    sigma_mod = compute_sigma_mod(mu_x, mu_y, sigma_x, sigma_y)
    varphi_mod = sigma_to_variance(sigma_mod, w_Leq)

    def integrand_Q_1_LB(eta):
        term_1 = transmitivity_pdf(
            eta, mu_x, mu_y, sigma_x, sigma_y, zenith_angle_rad,
            w_L, w_Leq, tau_zen, varphi_mod, wavelength, h_OGS,
            h_atm, Cn2_profile, a)
        term_2 = (
            (np.exp(-n_s) * n_s**2)
            / (n_s * n_d - n_d**2)
        )
        Q_mu = compute_yield(eta, n_s, p_dark, p_AP)
        Q_nu = compute_yield(eta, n_d, p_dark, p_AP)
        
        term_3 = (
            Q_nu * np.exp(n_d)
            - Q_mu * np.exp(n_s) * n_d**2/n_s**2
            - (p_dark*(1+p_AP)) * (n_s**2 - n_d**2)/n_s**2 
        )

        return term_1 * term_2 * term_3

    def integrand_e_1_UB(eta):
        term_1 = transmitivity_pdf(
            eta, mu_x, mu_y, sigma_x, sigma_y, zenith_angle_rad,
            w_L, w_Leq, tau_zen, varphi_mod, wavelength, h_OGS,
            h_atm, Cn2_profile, a)

        # Q_nu = compute_yield(eta, n_d, p_dark, p_AP)
        term_2 = (
            (e_0 * (p_dark*(1+p_AP))
            + (e_pol+e_0*p_AP) * (1-np.exp(-n_d*eta))) * np.exp(n_d)
            - e_0 * (p_dark*(1+p_AP))
        )

        term_3 = (
            n_s * np.exp(-n_s)/n_d
        )

        return term_1 * term_2 * term_3

    Q_1_lb, _ = quad(integrand_Q_1_LB, 0, np.inf, limit=100, epsabs=1e-9, epsrel=1e-9)

    e_1_temp, _ = quad(integrand_e_1_UB, 0, np.inf, limit=100, epsabs=1e-9, epsrel=1e-9)

    e_1_ub = e_1_temp / Q_1_lb

    return Q_1_lb, e_1_ub


# 使用しない
def compute_SKR(
        qber, avg_yield, Q_1, e_1, sifting_coefficient=0.5, p_estimation=0.75,
        kr_efficiency=1, decoy_coefficient=0.5, rep_rate=1e9):
    term_1 = rep_rate * sifting_coefficient * p_estimation * decoy_coefficient
    term_2 = -avg_yield * kr_efficiency * entropy_func(qber)

    term_3 = Q_1 * (1 - entropy_func(e_1))

    if (term_2 + term_3) < 0:
        return 0
    else:
        return term_1 * (term_2 + term_3)

def compute_yield(eta, n_s, p_dark, p_AP):
    param_q = (p_dark*(1+p_AP)) + (1-np.exp(-n_s*eta))*(1+p_AP)
    return param_q


def entropy_func(p):
    if p == 0 or p == 1:
        return 0
    p = np.asarray(p)
    res = -p * np.log2(p) - (1-p) * np.log2(1-p)
    return res

def compute_SKR_BBM92_finite(total_sifted_bits, qber_z, qber_x, f_ec):
    """
    論文(Ecker et al.)の式(1)に基づき、有限長鍵の長さを計算します。
    """
    # 2つの基底で得られるビット数は等しいと仮定
    N_s_z = total_sifted_bits / 2.0
    N_s_x = total_sifted_bits / 2.0
    
    # 位相誤り率をQBERから推定
    E_ph_z = qber_x
    E_ph_x = qber_z

    # Z基底から得られる安全な鍵の長さ
    term_z = N_s_z * (1 - entropy_func(E_ph_z) - f_ec * entropy_func(qber_z))
    
    # X基底から得られる安全な鍵の長さ
    term_x = N_s_x * (1 - entropy_func(E_ph_x) - f_ec * entropy_func(qber_x))
    
    final_key_length = term_z + term_x
    
    return max(0, final_key_length)



def compute_SKR_without_Eve(qber_alice, yield_alice, qber_bob, yield_bob, f_ec=1.16, rep_rate=1e9):
    """
    Eveを考慮せず、アリスとボブの性能からSecret Key Rateを計算します。
    
    Args:
        qber_alice (float): 衛星-アリス間のQBER
        yield_alice (float): アリスのYield（ふるい分け確率）
        qber_bob (float): 衛星-ボブ間のQBER
        yield_bob (float): ボブのYield
        [cite_start]f_ec (float): 誤り訂正符号の効率。1.16は論文の値 [cite: 137]
        rep_rate (float): 光子ソースの生成レート (pulses/sec)

    Returns:
        float: Secret Key Rate (bits/sec)
    """
    
    # --- ステップ1: システム全体の性能を計算 ---
    
    # アリスとボブ両方がビットを検出する確率（同時検出確率）
    yield_system = yield_alice * yield_bob
    
    # ふるい分け鍵におけるエラー率（アリスとボブの記録が食い違う確率）
    qber_system = (1 - qber_alice) * qber_bob + qber_alice * (1 - qber_bob)
    
    # --- ステップ2: 相互情報量からエラー訂正コストを引く ---
    
    # エラー訂正によって消費される情報量の割合
    # H(E_sys)は、QBERがqber_systemの場合のエントロピー
    error_correction_cost = f_ec * entropy_func(qber_system)
    
    # 1ビットあたりの共有情報量からエラー訂正コストを引く
    # これが、ふるい分け鍵1ビットから得られる安全な鍵の割合（秘匿率）
    secure_key_fraction = 1 - error_correction_cost
    
    # --- ステップ3: 最終的な鍵レートを計算 ---
    
    # 1秒あたりのふるい分けビット数
    sifted_key_rate = yield_system * rep_rate
    
    # 1秒あたりの安全な鍵のビット数 (SKR)
    skr = sifted_key_rate * secure_key_fraction
    
    # SKRは負の値を取り得ない
    return max(0, skr)

    
def compute_SKR_final(qber_system, yield_alice, yield_bob, f_ec=1.16, rep_rate=1e9):
    """
    システム全体のQBERと、個別のYieldからSKRを計算します。
    
    Args:
        qber_system (float): システム全体のQBER
        yield_alice (float): アリスのYield
        yield_bob (float): ボブのYield
        f_ec (float): 誤り訂正符号の効率
        rep_rate (float): 光子ソースの生成レート (pulses/sec)

    Returns:
        float: Secret Key Rate (bits/sec)
    """
    # 受け取った個別のYieldから、システム全体のYieldを計算
    yield_system = yield_alice * yield_bob
    
    # エラー訂正のコストを計算
    error_correction_cost = f_ec * entropy_func(qber_system)
    
    # 安全な鍵の割合を計算 (Eveは考慮しない)
    secure_key_fraction = 1 - error_correction_cost
    
    # 最終的なSKRを計算
    sifted_key_rate = yield_system * rep_rate
    skr = sifted_key_rate * secure_key_fraction
    
    return max(0, skr)


def yield_from_photon_number(n, Y0_A, Y0_B, eta_A, eta_B):
    term_A = 1 - (1 - Y0_A) * (1 - eta_A)**n
    term_B = 1 - (1 - Y0_B) * (1 - eta_B)**n
    Yn = term_A * term_B
    return Yn

def photon_number_probability(n, wavelength):
    """
    Calculates the probability P(n) of having n photons in a pulse
    """
    if n < 0:
        return 0.0
    numerator = (n + 1) * (wavelength ** n)
    denominator = (1 + wavelength) ** (n + 2)
    return numerator / denominator


def qber_ma_model(e0, ed, etaA, etaB, lambda_, Y0_A, Y0_B):
    numerator = 2 * (e0 - ed) * etaA * etaB * lambda_ * (1 + lambda_)
    denominator = (1 + etaA * lambda_) * (1 + etaB * lambda_) * (1 + etaA * lambda_ + etaB * lambda_ - etaA * etaB * lambda_)
    Q_lambda = compute_Q_lambda(lambda_, etaA, etaB, Y0_A, Y0_B)
    print(f'Q_lambda : {Q_lambda}')
    E_lambda = (e0 * Q_lambda - numerator / denominator) / Q_lambda
    E_lambda = float(E_lambda)
    return E_lambda

def compute_Q_lambda(lambda_, etaA, etaB, Y0_A, Y0_B):
    """
    Calculates the overall gain Q_lambda from Ma et al. (Eq. 9).
    Parameters:
        lambda_ (float): PDC parameter (lambda = sinh^2(χ), mu = 2*lambda)
        eta_A (float): Total detection efficiency for Alice
        eta_B (float): Total detection efficiency for Bob
        Y0_A (float): Background (dark) count probability for Alice
        Y0_B (float): Background (dark) count probability for Bob
    Returns:
        Q_lambda (float): Overall gain (coincidence detection probability per pulse)
    """
    term1 = (1 - Y0_A) / (1 + etaA * lambda_)**2
    term2 = (1 - Y0_B) / (1 + etaB * lambda_)**2
    term3 = ((1 - Y0_A) * (1 - Y0_B)) / (1 + (etaA + etaB) * lambda_)**2
    # print("term1:", term1)
    # print("term2:", term2)
    # print("term3:", term3)
    # print("Q_lambda:", Q_lambda)

    Q_lambda = 1 - term1 - term2 + term3
    return Q_lambda



def pauli_x_error_probability(i, e_0, p_dark, e_pol, insta_eta_alice, insta_eta_bob):
    if i == 0: # n = 0 (a0)
        current_alice_qber = 0.5
        current_bob_qber = 0.5
    elif i == 1: # n = 1 (a1)
        numerator_a = (e_0 * p_dark) + (e_pol * insta_eta_alice)
        denominator_a = p_dark + insta_eta_alice
        current_alice_qber = numerator_a / denominator_a if denominator_a != 0 else 1.0

        numerator_b = (e_0 * p_dark) + (e_pol * insta_eta_bob)
        denominator_b = p_dark + insta_eta_bob
        current_bob_qber = numerator_b / denominator_b if denominator_b != 0 else 1.0
    elif i == 2: # n = 2 (a2)
        P_detect_channel_n2_alice = (1 - (1 - insta_eta_alice)**2)
        P_detect_channel_n2_bob = (1 - (1 - insta_eta_bob)**2)

        numerator_a = (e_pol * P_detect_channel_n2_alice * (2/3) + e_0 * P_detect_channel_n2_alice * (1/3) + p_dark)
        denominator_a = (P_detect_channel_n2_alice + p_dark)
        current_alice_qber = numerator_a / denominator_a if denominator_a != 0 else 1.0

        numerator_b = (e_pol * P_detect_channel_n2_bob * (2/3) + e_0 * P_detect_channel_n2_bob * (1/3) + p_dark)
        denominator_b = (P_detect_channel_n2_bob + p_dark)
        current_bob_qber = numerator_b / denominator_b if denominator_b != 0 else 1.0
    return current_alice_qber, current_bob_qber