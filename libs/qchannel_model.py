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
    res = -p * np.log2(p) - (1-p) * np.log2(1-p)
    return res


