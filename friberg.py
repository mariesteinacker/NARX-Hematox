import numpy as np
from scipy.integrate import solve_ivp
from sklearn.preprocessing import (MinMaxScaler, StandardScaler,
                                   FunctionTransformer)

# helper function for log-transformation
log_transformer = FunctionTransformer(np.log10,
                                      inverse_func=lambda x: 10.**x,
                                      validate=True, check_inverse=True)


def E_drug_step(t, slope, therapies_tuple):
    """
    Drug effect function
    :param t: time in hours
    :param slope: E_eff individual parameter of drug effect
    :param therapies_tuple: treatment times and dose information, tuple
    :return:
    """
    therapies, dosage = therapies_tuple

    # before first treatment, treatment has no effect
    if t < np.min(therapies):
        E_drug = 0
    else:
        # create local time from last therapy start,
        # where t_loc(start) = 0
        local_start = max(np.extract(therapies <= t, therapies))
        local_dosage = dosage[np.argwhere(therapies == local_start)].flatten()[
            0]
        t_loc = t - local_start

        # on first treatment day, drug effect is assumed proportional to dose
        if t_loc <= 23:
            E_drug = slope * local_dosage

        # on days without treatment, drug effect is asssumed zero
        else:
            E_drug = 0.
    return E_drug


def friberg_drug(t, y, ktr, gamma, c0, slope, therapies):
    """
    differential eqn defining friberg model including cell loss due to
    E_drug
    :param t: time in hours
    :param y: vec of compartments (prol, t1, t2, t3, circ)
    :param ktr: rate constant
    :param gamma: exponent for feedback
    :param c0: baseline of circulating blood cells
    :param slope: slope of drug function
    :param therapies: given therapy time points and dosages
    :return: vec of dy/dt
    """
    # get vec components
    prol, t1, t2, t3, circ = y

    # get Edrug for t point
    Edrug = E_drug_step(t, slope, therapies)

    # diff eq system
    dprol = ktr * prol * (1 - Edrug) * (c0 / circ) ** gamma - ktr * prol
    dt1 = ktr * prol - ktr * t1
    dt2 = ktr * t1 - ktr * t2
    dt3 = ktr * t2 - ktr * t3
    dcirc = ktr * t3 - ktr * circ

    return dprol, dt1, dt2, dt3, dcirc


def solve_friberg(gamma, MTT, slope, c0, therapies_tuple,
                            end_fu, t_eval):
    """
    solve friberg equation for given parameters and therapy plan
    therapy starting time points given as list/array in days and
    standardized doses
    :param gamma: exponent for feedback
    :param MTT: cell maturation time in hours
    :param slope: slope for linear drug effect for drug function
    :param c0: baseline of circulating blood cells/l
    :param therapies_tuple: (therapy days, dosage in average dosage units)
    :param end_fu: follow up after last periodic treatment in days
    :param t_eval: evaluation time points in days
    :return: number of circulating blood cells/l at t_eval
    """

    therapies, dosage = therapies_tuple
    # therapy time points in hours
    therapies = therapies * 24

    # params
    ktr = 4. / MTT
    y0 = c0 * np.ones(5)        # start from steady state

    # time
    tmax = np.max(therapies) + end_fu * 24

    # solve friberg
    sol = solve_ivp(friberg_drug, (0, tmax), y0, method='LSODA',
                    t_eval=t_eval*24,
                    args=(
                    ktr, gamma, c0, slope, (therapies, dosage)),
                    rtol=1e-6, atol=1e-9, max_step=24, jac=None)
    return sol.y[-1]


def poplog_fit_friberg_mse(params, pop_params, y_arr, t_arr, therapies):
    """
    loss function to calibrate friberg model given population context,
    on logarithmic scale to emphasize nadir calibration
    :param params: model parameter guess [gamma, MTT, slope, c0/10**9]
    :param pop_params: population parameters [gamma, MTT, slope, c0/10**9]
    :param y_arr: observation to calibrate friberg model to
    :param t_arr: obsrevation times in days
    :param therapies: treatment times in days and accompanying doses
    :return: loss function, mse + population penalty
    """
    gamma, MTT, slope, c0t = params
    c0 = c0t * 10**9
    # prediction with current parameters
    y_pred = solve_friberg(gamma,  MTT, slope, c0, therapies,
                                            end_fu=t_arr[-1] +1 -
                                                   therapies[0][-1],
                           t_eval=t_arr)
    y_finite = np.log10(y_pred)
    # Mean squared error to observations
    mse = np.sum((y_finite[np.isfinite(y_finite)]
                  - np.log10(y_arr)[np.isfinite(y_finite)])**2)/ len(y_arr)
    # population penalty
    param_loss = ((np.log(params) - np.log(pop_params))**2) / (5.** 2)
    return mse + np.sum(param_loss)


def friberg(params, t_arr, therapies):
    """
    generate friberg model prediction with given parameters
    :param params: model parameter guess [gamma, MTT, slope, c0/10**9]
    :param t_arr: days to evaluate
    :param therapies: treatment in administration days and accompanying doses
    :return: predicted times series
    """
    gamma, MTT, slope, c0t = params
    c0 = c0t * 10**9
    y_pred = solve_friberg(
        gamma, MTT, slope, c0, therapies, end_fu=t_arr[-1] +1-therapies[0][-1],
        t_eval=t_arr)
    return y_pred


def test_friberg(pc, therapies, t_arr, scalery):
    """
    helper function for testing friberg model in context of transfer learning
    :param pc: friberg model parameters [gamma, MTT, slope, c0/10**9]
    :param therapies: treatment in administration days and accompanying doses
    :param t_arr: evaluation times in days
    :param scalery: scaling function for NARX model usage
    :return: treatment information for NARX model, scaled friberg model
    prediciton and unscaled friberg model prediction
    """
    # unpack friberg parameter configuration
    gamma, MTT, slope, c0t = pc
    c0 = c0t * 10 ** 9

    # get raw scenario data
    y_raw = solve_friberg(gamma, MTT, slope, c0,
                                   (therapies[0], therapies[1]),
                                   end_fu=t_arr[-1] - therapies[0][-1],
                          t_eval=t_arr)
    X_raw = np.zeros(len(y_raw))
    X_raw[therapies[0]] = therapies[1]

    # scale raw data
    y_log = log_transformer.transform(
        np.array(y_raw).flatten().reshape(-1, 1))
    y = scalery.fit_transform(y_log).flatten().reshape(
        np.array(y_raw).shape)

    return X_raw.to_numpy(), y, y_raw.to_numpy()