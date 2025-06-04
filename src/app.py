import numpy as np
from scipy.integrate import odeint
from numpy import exp as exp
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.io as sio


#Array export and creating

CK = np.arange(0, 0.110001, 0.005) #23
CNa = np.arange(0, 0.50001, 0.025) #21
psi = np.arange(-80, 261, 20) #18
flux_psi = sio.loadmat('flux_psi.mat')['flux_psi']
kflux_psi = sio.loadmat('kflux_psi.mat')['kflux_psi']
naflux_psi = sio.loadmat('naflux_psi.mat')['naflux_psi']

def interpolate_flux(K, Na, Psi, CK=CK, CNa=CNa, psi=psi, flux_psi=flux_psi):
    def find_nearest_indices(value, array):
        idx = np.searchsorted(array, value)
        idx = np.clip(idx, 1, len(array) - 1)
        return idx - 1, idx

    def interpolate_1d(value, array, idx1, idx2):
        x1, x2 = array[idx1], array[idx2]
        return (value - x1) / (x2 - x1)

    k_idx1, k_idx2 = find_nearest_indices(K, CK)
    na_idx1, na_idx2 = find_nearest_indices(Na, CNa)
    psi_idx1, psi_idx2 = find_nearest_indices(Psi, psi)

    k_ratio = interpolate_1d(K, CK, k_idx1, k_idx2)
    na_ratio = interpolate_1d(Na, CNa, na_idx1, na_idx2)
    psi_ratio = interpolate_1d(Psi, psi, psi_idx1, psi_idx2)

    c00 = flux_psi[k_idx1, na_idx1, psi_idx1] * (1 - k_ratio) + flux_psi[k_idx2, na_idx1, psi_idx1] * k_ratio
    c01 = flux_psi[k_idx1, na_idx1, psi_idx2] * (1 - k_ratio) + flux_psi[k_idx2, na_idx1, psi_idx2] * k_ratio
    c10 = flux_psi[k_idx1, na_idx2, psi_idx1] * (1 - k_ratio) + flux_psi[k_idx2, na_idx2, psi_idx1] * k_ratio
    c11 = flux_psi[k_idx1, na_idx2, psi_idx2] * (1 - k_ratio) + flux_psi[k_idx2, na_idx2, psi_idx2] * k_ratio

    c0 = c00 * (1 - na_ratio) + c10 * na_ratio
    c1 = c01 * (1 - na_ratio) + c11 * na_ratio

    return c0 * (1 - psi_ratio) + c1 * psi_ratio

# vATPase function
def J_v(pH_L, psi):
    psi_start, psi_end = -300, 300
    pH_start, pH_end = 0, 9

    if (pH_L < pH_start):
        pH_L = pH_start
    elif (pH_L > pH_end):
        pH_L = pH_end

    if (psi < psi_start):
        psi = psi_start
    elif (psi > psi_end):
        psi = psi_end

    p_a = 3.3654e-24 * psi ** 10 + -1.72718633e-21 * psi ** 9 + -1.88006118e-19 * psi ** 8 + 1.84428806e-16 * psi ** 7 + 1.14875767e-15 * psi ** 6 + -7.86180209e-12 * psi ** 5 + 8.73621142e-11 * psi ** 4 + 1.38612359e-07 * psi ** 3 + 4.02706504e-06 * psi ** 2 + -0.000921991562 * psi ** 1 + 0.892742811 * psi ** 0
    p_b = -0.01183303 * psi ** 1 + -1.74410965 * psi ** 0
    p_c = 1.97927106e-17 * psi ** 8 + -8.11163955e-15 * psi ** 7 + -1.41331765e-12 * psi ** 6 + 7.37414295e-10 * psi ** 5 + 2.69242151e-08 * psi ** 4 + -1.47727446e-05 * psi ** 3 + -0.0018809272 * psi ** 2 + 0.0294192901 * psi ** 1 + 122.27226 * psi ** 0
    p_d = 3.32575786e-13 * psi ** 6 + -3.43339599e-11 * psi ** 5 + -5.76545216e-08 * psi ** 4 + 7.89631331e-06 * psi ** 3 + 0.00138326473 * psi ** 2 + 0.11950506 * psi ** 1 + 1.50734239 * psi ** 0

    J = np.tanh(p_a * pH_L + p_b) * p_c - p_d
    return J


init_R = 0.34  # organelle radius [microns]
init_V = (4 / 3 * 3.1416 * init_R ** 3 * 1e-12) / 1000  # organelle volume [Liters]
init_S = 4 * 3.1416 * init_R ** 2 * 1e-8  # organelle {surface area [cm**2]}
init_pH = 4.2  # init lysosome pH

CAX_Ca = 1  # Calcium stoichiometry of CAX
CAX_H = 3  # Proton stoichiometry of CAX
CLC_Cl = 2  # Chloride stoichiometry of ClC-7
CLC_H = 1  # Proton stoichiometry of ClC-7
Ca_C = 1e-07  # [M]
Cl_C = 0.05  # [M]
Na_C = 0.015  # [M]
F = 96485  # [C] Na*e Faraday constant
K_C = 0.145  # [M] cytosolic  potassium
NA = 6.02e+23  # Avogadro constant

N_CAX = 0  # []
N_CLC = 50  # []

N_VATP = 550  # [numbers]
N_kNa = 13000

# N_VATP = 10      #[]
incr_coeff = 3.0
P_Ca = 1.49e-7  # [ion*cm/s]
P_Cl = 1.2e-5  # [ion*cm/s]
P_H = 6e-05  # [ion*cm/s]
P_K = 7.1e-7  # [ion*cm/s]
P_Na = 9.6e-7  # [ion*cm/s]
# R = 0.34           #[mcm]
RTF = 25.69  # RT/F [mV]} 
# S = (1.45267584e-08)*(incr_coeff**2 )     #[cm^2]
ans = 0.25  # []
beta_pH = 0.04  # [M/pH] Proton buffering capacity
cap = 1.45267584e-14  # [Farad] -  тут в оригинале -17, и в беркле тоже, это странно
cap_0 = 1e-06  # [F/cm^2] Bilayer capacitance
init_Aeff = 0.3
init_Ca_F = 0.0006      #[M]
init_Ca_T = 0.006      #[M]
init_Cl = 0.100  #[M]
init_pH = 4.7
init_H = 1e-5     #[M]
init_K = 0.060   #[M]
init_Na = 0.020   #[M]

psi_in = 0  # [mV]
psi_out = -50

B = init_K+init_Na+init_H-init_Cl  + init_Ca_T*2 - cap/F/init_V*(psi_in - psi_out)
# init_V = 1.64636595e-16      #[L]

init_psi_total = 0  # [mV]
p = 0
pH_C = 7.2
# [mV]
q = 2.2
r = 0.1  # Deactivation to activation ratio
tau_act = 1  # [s]
tau_deact = 0.25  # [s]

Pw = 0.054e-2  # water permeability
Oc = 0.291
oh = 0.73
ok = 0.73
ona = 0.73
ocl = 0.73
Q = init_V * (Oc - (oh * 10 ** (-init_pH) + ok * init_K + ona * init_Na + ocl * init_Cl))

k_fus = 1e3
k_fis = 1e-2
init_Sp = 0.1
init_HSp = 0
ohsp = 0.73


# Functions
def derivatives(X, t, water_flux=False, vATPase_stress=False,
                permabilization=False, proton_eflux=False, ca_signal=False):
    '''
    compute derivatives of X wrt time
    a = additional initial parameteres
    '''

    N_VATP = 550
    N_CAX = 16
    p = 3.88e-12

    P_Ca = 1.49e-7        #[ion*cm/s]
    P_Cl = 1.2e-5       #[ion*cm/s]
    P_H = 6e-05      #[ion*cm/s]
    P_K = 7.1e-7      #[ion*cm/s]
    P_Na = 9.6e-7      #[ion*cm/s]
    Pw = 0.054e-2

    pure_basefication = 0

    Aeff, NH, pH, NK, NNa, NCl, NCa_T, NCa_F, R, NSp = X

    if vATPase_stress:
        if (t > 300 and t < 350):
            N_VATP = 0

    if ca_signal:
        P_Ca = 1.49e-7
        N_CAX = 16
        p = 3.88e-12

    if permabilization:
        if (t > 300 and t < 350):
            n_per = 10
            P_Ca = 1.49e-7*n_per        #[ion*cm/s]
            P_Cl = 1.2e-5*n_per       #[ion*cm/s]
            P_H = 6e-05*n_per        #[ion*cm/s]
            P_K = 7.1e-7*n_per        #[ion*cm/s]
            P_Na = 9.6e-7*n_per   
            Pw = 0.054e-2*n_per  

    V = (4 / 3 * 3.1416 * abs(R) ** 3 * 1e-12) / 1000
    S = 4 * 3.1416 * abs(R) ** 2 * 1e-8

    # Luminal Concentrations
    H = NH / V / NA
    K = NK / V / NA
    Na = NNa / V / NA
    Cl = NCl / V / NA
    Ca_F = NCa_F / V / NA
    Ca_T = NCa_T / V / NA
    r = Ca_F / Ca_T
    Sp = NSp / V / NA
    HSp = init_Sp - Sp

    # membrane potential
    #     psi = (F/cap)*init_V*(H + K + Na - Cl + 2*Ca_T - B)
    psi = (F / cap) * (V * (H + K + Na - Cl + 2 * Ca_T) - B * init_V)

    #      Modified Cytoplasmic Surface Concentrations
    pH_C0 = (pH_C + psi_out / (RTF * 2.3))
    K_C0 = K_C * exp(-psi_out / RTF)
    Na_C0 = Na_C * exp(-psi_out / RTF)
    Cl_C0 = Cl_C * exp(psi_out / RTF)
    Ca_F_C0 = Ca_C * exp(-2 * psi_out / RTF)

    #     Modified Luminal Surface Concentrations
    pH_L0 = (pH + psi_in / (RTF * 2.3))
    K_L0 = K * exp(-psi_in / RTF)
    Na_L0 = Na * exp(-psi_in / RTF)
    Cl_L0 = Cl * exp(psi_in / RTF)
    Ca_F_L0 = Ca_F * exp(-2 * psi_in / RTF)

    delta_pH = pH_C0 - pH_L0;

    #     Treatment of singular terms for passive ion flux
    if (abs(psi) > 300):
        psi = np.sign(psi) * 300
    if (abs(psi) > 0.01):
        gg = psi / (1 - exp(- psi / RTF)) / RTF
        gg_Ca = 2 * psi / (1 - exp(-2 * psi / RTF)) / RTF

    else:
        gg = 1 / 1 - (psi / RTF) / 2 + (psi / RTF) ** 2 / 6 - (psi / RTF) ** 3 / 24 + (psi / RTF) ** 4 / 120
        gg_Ca = 1 / (1 - (psi / RTF) + (2 / 3) * (psi / RTF) ** 2 - (1 / 3) * (psi / RTF) ** 3 + (2 / 15) * (
                    psi / RTF) ** 4)

    # vAPTase
    J_VATPASE = N_VATP * J_v(pH, psi)

    # ClC-7 Antiporter {H out, Cl in}
    CLC_mu = (CLC_H + CLC_Cl) * psi + RTF * (CLC_H * 2.3 * delta_pH + CLC_Cl * np.log(Cl_C0 / Cl_L0))

    #     Switching function
    x = 0.5 + 0.5 * np.tanh((CLC_mu + 250) / 75);
    # Activity
    A = 0.3 * x + 1.5E-5 * (1 - x) * CLC_mu ** 2;

    if (A < Aeff):
        tau = tau_deact
    else:
        tau = tau_act

    J_CLC = N_CLC * Aeff * CLC_mu

    #      CAX Antiporter {H out, Ca in} [mV, ion/s]
    CAX_mu = (CAX_H - 2 * CAX_Ca) * psi + RTF * (CAX_H * 2.3 * delta_pH + CAX_Ca / 2 * np.log(Ca_F_L0 / Ca_F_C0))
    J_CAX = N_CAX * CAX_mu

    #      Passive flux [ion/s]
    J_H = P_H * S * (10 ** (-pH_C0) * exp(-psi / RTF) - 10 ** (-pH_L0)) * gg * NA / 1000
    J_K = P_K*S*(K_C0*exp(-psi/RTF)-K_L0)*gg*NA/1000
    J_Na = P_Na*S*(Na_C0*exp(-psi/RTF)-Na_L0)*gg*NA/1000

    J_Cl_unc = P_Cl * S * (Cl_C0 - Cl_L0 * exp(-psi / RTF)) * gg * NA / 1000
    J_Ca = P_Ca * S * (Ca_F_C0 * exp(-2 * psi / RTF) - Ca_F_L0) * gg_Ca * NA / 1000

    # water flux
    J_w = Pw * S * (oh * 10 ** (-pH) + ok * K + ona * Na + ocl * Cl + Q / V - Oc)

    #     TRPML1 channel
    y = 0.5 - 0.5*np.tanh(psi + 40)
    P_trpml1 = p*(y*abs(psi) + (1-y)*(abs(psi + 40)**3)/(pH**q)) #change p
    J_Ca_trpml1 = P_trpml1*S*(Ca_F_C0*exp(-2*psi/RTF)-Ca_F_L0)*gg_Ca*NA/1000

    J_vNa = interpolate_flux(K=K, Na = Na, Psi = psi, flux_psi = naflux_psi)
    J_vK = interpolate_flux(K=K, Na = Na, Psi = psi, flux_psi = kflux_psi)

    if proton_eflux:
        if (t > 300 and t < 350):
            pure_basefication = 3.22e4

    if NSp:
        w_fus = k_fus * np.power(10, -pH) * Sp
        w_fis = k_fis * HSp
        w = (w_fus - w_fis) * NA * V
        J_w = Pw * S * (oh * 10 ** (-pH) + ok * K + ona * Na + ocl * Cl + ohsp * HSp + Q / V - Oc)
    else:
        w = 0

    if water_flux:
        if (t > 300 and t < 350):
            J_w += 1e-13

    dxdt = [(1 / tau) * (A - Aeff),
            J_H + (J_VATPASE) - (CLC_H * J_CLC) - (CAX_H * J_CAX) - w - pure_basefication,
            (-(J_H + (J_VATPASE) - (CLC_H * J_CLC) - (CAX_H * J_CAX) - w - pure_basefication) / V / NA) / beta_pH,
            J_K + N_kNa*J_vK,
            J_Na + N_kNa*J_vNa,
            J_Cl_unc + (CLC_Cl * J_CLC),
            J_Ca + (CAX_Ca * J_CAX) + J_Ca_trpml1,
            (J_Ca + (CAX_Ca * J_CAX) + J_Ca_trpml1) * r,
            J_w / (1000 * 55) / (4 * np.pi * (R / 1e5) ** 2) * 1e5,
            -w]

    return dxdt

# Init
calcium_smth = np.arange(0, 1.1, 1, dtype=float)

# init_pH = np.append(np.arange(4, 7.5, 0.5, dtype=float), 4.7)
init_pH = np.arange(4, 7.5, 0.2, dtype = float)
init_Na = np.arange(20, 120, 10, dtype=float) / 1000 #M
init_K = np.arange(5, 100, 5 ,dtype=float) / 1000
init_Cl = np.arange(50, 120, 5, dtype=float) / 1000

init_NH = np.power(10, -init_pH)*init_V*NA;      #[ions]
init_NK = init_K*init_V*NA;
init_NNa = init_Na*init_V*NA;
init_NCl = init_Cl*init_V*NA;
init_NCa_T = init_Ca_T*init_V*NA;
init_NCa_F = init_Ca_F*init_V*NA;
init_NSp = init_Sp*init_V*NA;

X0 = [init_Aeff, init_NH[0], init_pH[0], init_NK[0], init_NNa[0], init_NCl[0], init_NCa_T, init_NCa_F, init_R, init_NSp]

a = [False, False, False, False, False]

tspan = np.arange(0, 600, 1e-2)

app = Dash(__name__)
server = app.server
app.layout = html.Div(children=[
    html.Label('Plot type:'),
    html.Div(children=[
        dcc.RadioItems(
            ['Ions concentration', 'Water flux and radius', 'pH and potential', 'single vATPase activity'],
            'Ions concentration',
            id='graph-type',
            style={'display': 'flex', 'flexDirection': 'row'}
        ),
    ]),

    html.Div(children=[dcc.Graph(id='ions', mathjax=True)]),

    html.Div(children=[

        dcc.Checklist(
            options=[
                {'label': 'Show Calcium', 'value': 'Ca'},
            ],
            value=[],
            id='Ca-check'),

        html.Br(),
        html.Label('Initial lysosome pH'),
        dcc.Slider(
            init_pH.min(),
            init_pH.max(),
            step = 0.025,
            value=4.7,
            marks={str(time): str(round(time, 1)) for time in init_pH},
            id='ph-slider'),

        dcc.Markdown(r'$\text{Initial} \ [\text{Na}^+]_L, \ \text{mM}$', mathjax=True),
        dcc.Slider(
            init_Na.min(),
            init_Na.max(),
            value=0.02,
            marks={str(time): str(int(time * 1000)) for time in init_Na},
            id='na-slider'),

        dcc.Markdown(r'$\text{Initial} \ [\text{K}^+]_L, \  \text{mM}$', mathjax=True),
        dcc.Slider(
            init_K.min(),
            init_K.max(),
            value=0.06,
            marks={str(time): str(int(time * 1000)) for time in init_K},
            id='k-slider'),

        dcc.Markdown(r'$\text{Initial} \ [\text{Cl}^-]_L, \  \text{mM}$', mathjax=True),
        dcc.Slider(
            init_Cl.min(),
            init_Cl.max(),
            value=0.1,
            marks={str(time): str(int(time * 1000)) for time in init_Cl},
            id='cl-slider'),

        html.Label('Stresses'),
        dcc.Checklist(
            options=[
                {'label': f'Water influx (S)', 'value': 'WF'},
                {'label': 'Permeabilization (S)', 'value': 'LMP'},
                {'label': 'vATPases number decrease (S)', 'value': 'ATP'},
                {'label': 'Proton efflux (S)', 'value': 'LX'},
                {'label': 'Proton sponge (L)', 'value': 'SP'},
            ],
            value=[],
            id='stress-check', inline=True

        ),
        html.Br(),
        html.Label('S for short-term (from 300 to 350s) stresses, '),
        html.Br(),
        html.Label('L for long-term (from start to end of simulation) stresses'),

    ])])


@callback(
    Output('ions', 'figure'),
    [Input('ph-slider', 'value'),
     Input('na-slider', 'value'),
     Input('k-slider', 'value'),
     Input('cl-slider', 'value'),
     Input('graph-type', 'value'),
     Input('Ca-check', 'value'),
     Input('stress-check', 'value')]
)
def update_figure(selected_ph, selected_na, selected_k,
                  selected_cl, graph_type,
                  calcium_check, stress_check):
    # initial parameters
    X0[1] = np.power(10.0, -selected_ph) * init_V * NA
    X0[2] = selected_ph
    X0[3] = selected_k * init_V * NA
    X0[4] = selected_na * init_V * NA
    X0[5] = selected_cl * init_V * NA
    X0[-1] = int('SP' in stress_check) * init_V * NA * init_Sp

    a[0] = 'WF' in stress_check
    a[1] = 'ATP' in stress_check
    a[2] = 'LMP' in stress_check
    a[3] = 'LX' in stress_check
    a[4] = 'Ca' in calcium_check

    solution = odeint(derivatives, X0, tspan, args=tuple(a))

    V_arr = (4 / 3 * 3.1416 * solution[:, 8] ** 3 * 1e-12) / 1000

    cl_conc = solution[:, 5] / V_arr / NA
    k_conc = solution[:, 3] / V_arr / NA
    na_conc = solution[:, 4] / V_arr / NA
    ca_conc = solution[:, 7] / V_arr / NA

    V_arr = (4 / 3 * 3.1416 * solution[:, 8] ** 3 * 1e-12) / 1000

    J_w_arr = (Pw * 4 * 3.1416 * abs(solution[:, 8]) ** 2 *
               1e-8 * (oh * 10 ** (-solution[:, 2]) + ok *
                       solution[:, 3] / V_arr / NA + ona *
                       solution[:, 4] / V_arr / NA + ocl *
                       solution[:, 5] / V_arr / NA + Q / V_arr - Oc))*1000*NA

    radius = solution[:, 8]

    # psi_arr = (F/cap)*init_V*(solution[:, 1]/init_V/NA +
    #                           solution[:, 3]/init_V/NA +
    #                           solution[:, 4]/init_V/NA -
    #                           solution[:, 5]/init_V/NA +
    #                           2*solution[:, 6]/init_V/NA - B)
    psi_arr = (F/cap)*(V_arr*(solution[:, 1]/V_arr/NA +
                              solution[:, 3]/V_arr/NA +
                              solution[:, 4]/V_arr/NA -
                              solution[:, 5]/V_arr/NA +
                              2*solution[:, 6]/V_arr/NA )- B*init_V)

    ph = solution[:, 2]

    v_arr = np.vectorize(J_v)(solution[:, 2], psi_arr)

    if graph_type == 'Ions concentration':
        if 'Ca' in calcium_check:
            times = np.tile(tspan, 4)
            concentrations = np.concatenate((cl_conc * 1e3, k_conc * 1e3, na_conc * 1e3, ca_conc * 1e3), axis=None)
            ions = np.concatenate((np.repeat('Cl', cl_conc.shape[0]), np.repeat('K', k_conc.shape[0])
                                   , np.repeat('Na', na_conc.shape[0]), np.repeat('Ca', ca_conc.shape[0])), axis=None)

            df = pd.DataFrame({'Time': times,
                               'Concentrations': concentrations,
                               'Ion': ions})

        else:
            times = np.tile(tspan, 3)
            concentrations = np.concatenate((cl_conc * 1e3, k_conc * 1e3, na_conc * 1e3), axis=None)
            ions = np.concatenate((np.repeat('Cl', cl_conc.shape[0]), np.repeat('K', k_conc.shape[0])
                                   , np.repeat('Na', na_conc.shape[0])), axis=None)

            df = pd.DataFrame({'Time': times,
                               'Concentrations': concentrations,
                               'Ion': ions})

        fig = px.line(df, x="Time", y="Concentrations", color="Ion", render_mode='webgl')

        fig.update_layout(font_family='Rockwell', font_size=18)
        fig.update_xaxes(title_text="Time, s")
        fig.update_yaxes(title_text='Concentration, mM')

        return fig

    if graph_type == 'Water flux and radius':
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x=tspan, y=radius, name="R", mode='lines'),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=tspan, y=J_w_arr, name="Water Flux", mode='lines'),
            secondary_y=True
        )

        fig.update_xaxes(title_text="Time, s")
        fig.update_yaxes(title_text=r"$\text{Radius}, \  \mu m$", secondary_y=False)
        fig.update_yaxes(title_text=r"$\text{Water flux}, \ \text{H}_2\text{O}/s$", secondary_y=True, showgrid=False, ticks="outside", tickwidth=0.5, tickcolor='black', ticklen=10, exponentformat = "E")

        fig.update_layout(font_family='Rockwell', font_size=18)


        return fig

    if graph_type == 'pH and potential':
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=tspan, y=ph, name="pH", mode='lines'),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=tspan, y=psi_arr, name=r'$\Psi$', mode='lines'),
            secondary_y=True
        )
        fig.update_xaxes(title_text='Time, s')
        fig.update_yaxes(title_text='pH', secondary_y=False)
        fig.update_yaxes(title_text=r'$\Psi, \text{mV}$', secondary_y=True, showgrid=False, ticks="outside", tickwidth=0.5, tickcolor='black', ticklen=10)

        fig.update_layout(font_family='Rockwell', font_size=18)
        return fig

    if graph_type == 'single vATPase activity':
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=tspan, y=v_arr*550/6, name="J_v", mode='lines')
        )

        fig.update_xaxes(title_text="Time, s")
        fig.update_yaxes(title=dict(text=r"$\text{single vATPase activity}, \  \text{H}^+ /s$"))

        fig.update_layout(font_family='Rockwell', font_size=18)
        return fig


app.title = 'Lysosome model'




if __name__ == '__main__':
    print('Please copy the link and paste it into your browser.')
    app.run(debug=True)





