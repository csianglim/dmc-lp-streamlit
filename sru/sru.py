import streamlit as st
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from pulp import LpVariable, LpStatus, LpProblem, LpMinimize, value, PULP_CBC_CMD
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.ticker as mtick
from matplotlib.colors import TwoSlopeNorm, Normalize

st.set_page_config(layout="wide")

# variable names and engineering units
varMV1 = {'Name': 'O2', 'UOM': '%', 'Cost': 10, 'SSVal':30.0, 'Limits': (21, 37), 'EngLimits': (21.0, 70.0), 'TYPMOV': 0.1}
varMV2 = {'Name': 'FG', 'UOM': 'SCFH', 'Cost': 0.1, 'SSVal':900.0, 'Limits': (400, 1000), 'EngLimits': (290, 1500), 'TYPMOV': 10}

varCV1 = {'Name': 'P', 'UOM': 'psi', 'SSVal':8.6, 'Limits': (7.0, 9.0), 'EngLimits': (6.0, 10.0), 'TYPMOV': 0.1}
varCV2 = {'Name': 'dP', 'UOM': 'psi', 'SSVal':10.5, 'Limits': (8.0, 14.0), 'EngLimits': (0.0, 20.0), 'TYPMOV': 0.1}
varCV3 = {'Name': 'Temp', 'UOM': "°F", 'SSVal':2120, 'Limits': (2100, 2150), 'EngLimits': (2050, 2200), 'TYPMOV': 1}

varFF1 = {'Name':'AC155.CO % (H2S-2*SO2)' ,'UOM':'%', 'SSVal': 50}
varFF2 = {'Name':'Amine Acid Gas' ,'UOM':'MSCFH', 'SSVal': 27.5}
varFF3 = {'Name':'SWS Gas' ,'UOM':'MSCFH', 'SSVal': 8.0}

if "init" not in st.session_state:
    st.session_state["init"] = True
    st.session_state["MV1Limits"] = varMV1['Limits']
    st.session_state["MV2Limits"] = varMV2['Limits']
    st.session_state["MV1Cost"] = varMV1['Cost']
    st.session_state["MV2Cost"] = varMV2['Cost']
    st.session_state["CV1Limits"] = varCV1['Limits']
    st.session_state["CV2Limits"] = varCV2['Limits']
    st.session_state["CV3Limits"] = varCV3['Limits']

    st.session_state['MV1SSVal'] = varMV1['SSVal']
    st.session_state['MV2SSVal'] = varMV2['SSVal']

    st.session_state['CV1SSVal'] = varCV1['SSVal']
    st.session_state['CV2SSVal'] = varCV2['SSVal']
    st.session_state['CV3SSVal'] = varCV3['SSVal']

    st.session_state['FF1SSVal'] = varFF1['SSVal']
    st.session_state['FF2SSVal'] = varFF2['SSVal']
    st.session_state['FF3SSVal'] = varFF3['SSVal']

    st.session_state["shade_MV1"] = False
    st.session_state["shade_MV2"] = False
    st.session_state["shade_CV1"] = False
    st.session_state["shade_CV2"] = False
    st.session_state["shade_CV3"] = False
    st.session_state["shade_feasible"] = False
    st.session_state["show_optimum"] = False    
    st.session_state["show_vectors"] = False
    st.session_state["show_isoprofit"] = False

with st.sidebar:
    st.title('LP-DMC Simulation')
    st.text("Model & Tuning Parameters")
    with st.expander('Gain Matrix (MV-CV)'):
        st.number_input(f"$G_{{11}}$: {varMV1['Name']} vs. {varCV1['Name']}", key="G11", value=-0.1131, step=0.02, format="%.4f")
        st.number_input(f"$G_{{12}}$: {varMV2['Name']} vs. {varCV1['Name']}", key="G12", value=0.00122, step=0.001, format="%.4f")
        st.number_input(f"$G_{{21}}$: {varMV1['Name']} vs. {varCV2['Name']}", key="G21", value=-0.5649, step=0.10, format="%.4f")
        st.number_input(f"$G_{{22}}$: {varMV2['Name']} vs. {varCV2['Name']}", key="G22", value=0.00188, step=0.001, format="%.4f")
        st.number_input(f"$G_{{31}}$: {varMV1['Name']} vs. {varCV3['Name']}", key="G31", value=3.0882, step=0.5, format="%.4f")
        st.number_input(f"$G_{{32}}$: {varMV2['Name']} vs. {varCV3['Name']}", key="G32", value=0.0782, step=0.01, format="%.4f")
    with st.expander('LP Costs'):
        st.number_input(rf"{varMV1['Name']} Cost", step=0.5, key="MV1Cost")
        st.number_input(rf"{varMV2['Name']} Cost", step=0.02, key="MV2Cost")
    with st.expander('Feedforward Gains'):
        st.number_input(f"$G_{{13}}$: {varFF1['Name']} vs. {varCV1['Name']}", key="G13", value=0.08096, step=0.1, format="%.4f")
        st.number_input(f"$G_{{14}}$: {varFF2['Name']} vs. {varCV1['Name']}", key="G14", value=0.19803, step=0.1, format="%.4f")
        st.number_input(f"$G_{{15}}$: {varFF3['Name']} vs. {varCV1['Name']}", key="G15", value=0.31035, step=0.1, format="%.4f")
        st.number_input(f"$G_{{23}}$: {varFF1['Name']} vs. {varCV2['Name']}", key="G23", value=0.20909, step=0.1, format="%.4f")
        st.number_input(f"$G_{{24}}$: {varFF2['Name']} vs. {varCV2['Name']}", key="G24", value=0.26042, step=0.1, format="%.4f")
        st.number_input(f"$G_{{25}}$: {varFF3['Name']} vs. {varCV2['Name']}", key="G25", value=0.14858, step=0.1, format="%.4f")
        st.number_input(f"$G_{{33}}$: {varFF1['Name']} vs. {varCV3['Name']}", key="G33", value=9.97798, step=1.0, format="%.4f")
        st.number_input(f"$G_{{34}}$: {varFF2['Name']} vs. {varCV3['Name']}", key="G34", value=1.42339, step=1.0, format="%.4f")
        st.number_input(f"$G_{{25}}$: {varFF3['Name']} vs. {varCV3['Name']}", key="G35", value=4.38991, step=1.0, format="%.4f")

    st.subheader('MV Limits')
    st.slider(rf"{varMV1['Name']} Limits ({varMV1['UOM']})", varMV1['EngLimits'][0], varMV1['EngLimits'][1], step=varMV1['TYPMOV'], key="MV1Limits")
    st.number_input(r'Current Value', step=varMV1['TYPMOV'], key="MV1SSVal")
    st.divider()

    st.slider(rf"{varMV2['Name']} Limits ({varMV2['UOM']})", varMV2['EngLimits'][0], varMV2['EngLimits'][1], step=varMV2['TYPMOV'], key="MV2Limits")
    st.number_input(r'Current Value', step=varMV2['TYPMOV'], key="MV2SSVal")
    st.divider()

    st.subheader('CV Limits')
    st.slider(rf"{varCV1['Name']} ({varCV1['UOM']})", varCV1['EngLimits'][0], varCV1['EngLimits'][1], step=varCV1['TYPMOV'], key="CV1Limits")
    st.number_input(r'Current Value', step=varCV1['TYPMOV'], key="CV1SSVal")
    st.divider()

    st.slider(rf"{varCV2['Name']} ({varCV2['UOM']})", varCV2['EngLimits'][0], varCV2['EngLimits'][1], step=varCV2['TYPMOV'], key="CV2Limits")
    st.number_input(r'Current Value', step=varCV2['TYPMOV'], key="CV2SSVal")
    st.divider()

    st.slider(rf"{varCV3['Name']} ({varCV3['UOM']})", varCV3['EngLimits'][0], varCV3['EngLimits'][1], step=varCV3['TYPMOV'], key="CV3Limits")
    st.number_input(r'Current Value', step=varCV3['TYPMOV'], key="CV3SSVal")

    st.divider()

    st.subheader('FF Disturbances')
    st.number_input(r'AC155.CO % (H2S-2*SO2)', step=0.1, key="FF1SSVal")
    st.caption(f"{varFF1['Name']} Baseline: {varFF1['SSVal']}{varFF1['UOM']}")
    st.number_input(r'Amine Acid Gas (MSCFH)', step=0.1, key="FF2SSVal")
    st.caption(f"{varFF2['Name']} Baseline: {varFF2['SSVal']} {varFF2['UOM']}")
    st.number_input(r'SWS Gas (MSCFH)', step=0.1, key="FF3SSVal")
    st.caption(f"{varFF3['Name']} Baseline: {varFF3['SSVal']} {varFF3['UOM']}")

    # st.markdown('Last Updated: Siang Lim (Sept 2025). Source on [GitHub](https://github.com/csianglim/dmc-lp-streamlit).')



def plotSVD(G):
    # Conditioning plots
    t = np.linspace(0,np.pi*2,200)
    X = np.array(list(zip(np.sin(t), np.cos(t))))
    Y = (G@X.T)
    u, s, Vt = np.linalg.svd(G, full_matrices=True)
    u_scaled = u @ np.diag(s)

    fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(5,10))

    ax1.plot(np.cos(t), np.sin(t), '-k', linewidth=1.5)
    ax1.plot(0,0,'ok')
    ax1.axvline(x=0, color='black', lw=0.5, linestyle='--')
    ax1.axhline(y=0, color='black', lw=0.5, linestyle='--')
    ax1.annotate("", xy=Vt[0,:], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='r', lw=1.5))
    ax1.annotate("", xy=Vt[1,:], xytext=(0, 0), arrowprops=dict(arrowstyle="->", linestyle="--",  color='b', lw=1.5))
    ax1.annotate("$v_1$", xy=Vt[0,:]/2, xycoords="data", color="red",
                      va="center", ha="center", bbox=dict(boxstyle="round", fc="w", edgecolor='none', pad=0.1))
    ax1.annotate("$v_2$", xy=Vt[1,:]/2, xycoords="data", color='blue',
                      va="center", ha="center", bbox=dict(boxstyle="round", fc="w", edgecolor='none', pad=0.1))

    ax2.plot(Y[0,:],Y[1,:], '-k', linewidth=1.5)
    ax2.plot(0,0,'ok')
    ax2.axvline(x=0, color='black', lw=0.5, linestyle='--')
    ax2.axhline(y=0, color='black', lw=0.5, linestyle='--')
    ax2.annotate("", xy=u_scaled[:,0], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='r', lw=1.5))
    ax2.annotate("", xy=u_scaled[:,1], xytext=(0, 0), arrowprops=dict(arrowstyle="->", linestyle="--",  color='b', lw=1.5))

    # ax2.annotate("$\sigma_1 u_1$", xy=u_scaled[:,0]/2, xycoords="data", color="red",
    #                   va="center", ha="center", bbox=dict(boxstyle="round", fc="w", edgecolor='none', pad=0.1))
    # ax2.annotate("$\sigma_2 u_2$", xy=u_scaled[:,1]/2, xycoords="data", color='blue',
    #                   va="center", ha="center", bbox=dict(boxstyle="round", fc="w", edgecolor='none', pad=0.1))

    labels = ['$A$', '$B$', '$C$', '$D$']
    points = [[0,1], [1,0], [0,-1], [-1,0]]

    for i, p1 in enumerate(points):
        ax1.plot(p1[0], p1[1], color='k', marker='o', markersize=8, markerfacecolor='y', alpha=0.65)
        # ax2.plot((G@p1)[0], (G@p1)[1], color='k', marker='o', markersize=8, markerfacecolor='y', alpha=0.65)

        if (G@p1)[0] <= 0 and (G@p1)[1] >= 0:
            offset = (-1,1)
        elif (G@p1)[0] >=0 and (G@p1)[1] >= 0:
            offset = (1,1)
        elif (G@p1)[0] >=0 and (G@p1)[1] <= 0:
            offset = (1,-1)
        else:
            offset = (-1,-1)

        offset = [o*10 for o in offset]

        ax1.annotate(labels[i], # this is the text
                     (p1[0]+np.sign(p1[0])*0.11,p1[1]+np.sign(p1[1])*0.11), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     # xytext=(2,2),
                     bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                     ha='center')

        # ax2.annotate(labels[i], # this is the text
        #              ((G@p1)[0]+(np.sign((G@p1)[0])*0.1), (G@p1)[1]+(np.sign((G@p1)[1])*0.1)), # these are the coordinates to position the label
        #              textcoords="offset points", # how to position the text
        #              xytext=((G@p1)[0],(G@p1)[1]), # distance from text to points (x,y)
        #              bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
        #              ha='center')    

    ax1.set_xlabel("$\Delta MV_1$")
    ax1.set_ylabel("$\Delta MV_2$")
    ax2.set_xlabel("$\Delta CV_1$")
    ax2.set_ylabel("$\Delta CV_2$")

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'));
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'));
        ax.set_box_aspect(1);

    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax2.set_xlim([-0.0125, 0.0125])
    ax2.set_ylim([-0.5, 0.5])

    return fig

# Grab vars
def state_snapshot(keys):
    """Return a hashable snapshot of selected session state keys."""
    return tuple((k, st.session_state[k]) for k in sorted(keys))

# MV-CV
G11 = st.session_state["G11"]
G12 = st.session_state["G12"]
G21 = st.session_state["G21"]
G22 = st.session_state["G22"]
G31 = st.session_state["G31"]
G32 = st.session_state["G32"]

# FF-CV
G13 = st.session_state["G13"]
G14 = st.session_state["G14"]
G15 = st.session_state["G15"]
G23 = st.session_state["G23"]
G24 = st.session_state["G24"]
G25 = st.session_state["G25"]
G33 = st.session_state["G33"]
G34 = st.session_state["G34"]
G35 = st.session_state["G35"]

CV1LoSS, CV1HiSS = st.session_state["CV1Limits"]
CV2LoSS, CV2HiSS = st.session_state["CV2Limits"]
CV3LoSS, CV3HiSS = st.session_state["CV3Limits"]
MV1LoSS, MV1HiSS = st.session_state["MV1Limits"]
MV2LoSS, MV2HiSS = st.session_state["MV2Limits"]

MV1Cost = st.session_state["MV1Cost"]
MV2Cost = st.session_state["MV2Cost"]

xlimits = 10
ylimits = 600

xspace = np.linspace(-xlimits, xlimits, 1000)
yspace = np.linspace(-ylimits, ylimits, 1000)
x,y = np.meshgrid(xspace,yspace)
v_new = 0

# Disturbances vs. initial SSVal
FF1PV = st.session_state['FF1SSVal'] - varFF1['SSVal']
FF2PV = st.session_state['FF2SSVal'] - varFF2['SSVal']
FF3PV = st.session_state['FF3SSVal'] - varFF3['SSVal']
D1 = G13*FF1PV - G14*FF2PV - G15*FF3PV
D2 = G23*FF1PV - G24*FF2PV - G25*FF3PV
D3 = G33*FF1PV - G34*FF2PV - G35*FF3PV

CV1Hi = CV1HiSS - (st.session_state['CV1SSVal'] + D1)
CV1Lo = CV1LoSS - (st.session_state['CV1SSVal'] + D1)
CV2Hi = CV2HiSS - (st.session_state['CV2SSVal'] + D2)
CV2Lo = CV2LoSS - (st.session_state['CV2SSVal'] + D2)
CV3Hi = CV3HiSS - (st.session_state['CV3SSVal'] + D3)
CV3Lo = CV3LoSS - (st.session_state['CV3SSVal'] + D3)

MV1Hi = MV1HiSS - st.session_state['MV1SSVal']
MV1Lo = MV1LoSS - st.session_state['MV1SSVal']
MV2Hi = MV2HiSS - st.session_state['MV2SSVal']
MV2Lo = MV2LoSS - st.session_state['MV2SSVal']

# CV constraints
c1 = G11*x+G12*y <= CV1Hi
c2 = G11*x+G12*y >= CV1Lo
c3 = G21*x+G22*y <= CV2Hi
c4 = G21*x+G22*y >= CV2Lo
c5 = G31*x+G32*y <= CV3Hi
c6 = G31*x+G32*y >= CV3Lo

# MV constraints
m1 = (x >= MV1Lo) & (x <= MV1Hi)
m2 = (y >= MV2Lo) & (y <= MV2Hi)

# equation of a line, y = mx + c
y_c1 = (CV1Hi - G11*xspace)/G12
y_c2 = (CV1Lo - G11*xspace)/G12
y_c3 = (CV2Hi - G21*xspace)/G22
y_c4 = (CV2Lo - G21*xspace)/G22
y_c5 = (CV3Hi - G31*xspace)/G32
y_c6 = (CV3Lo - G31*xspace)/G32

# the lp problem is formulated in terms of deviation variables - CV1Hi = CV1HiSS - CV1SSVal
prob = LpProblem("DMC", LpMinimize)

MV1 = LpVariable("MV1", MV1Lo)
MV2 = LpVariable("MV2", MV2Lo)

prob += MV1Cost * MV1 + MV2Cost * MV2

# constraint formulation in terms of MV1 and MV2
prob += G11*MV1+G12*MV2 <= CV1Hi, "CV1 High Limit"
prob += G11*MV1+G12*MV2 >= CV1Lo, "CV1 Low Limit"
prob += G21*MV1+G22*MV2 <= CV2Hi, "CV2 High Limit"
prob += G21*MV1+G22*MV2 >= CV2Lo, "CV2 Low Limit"
prob += G31*MV1+G32*MV2 <= CV3Hi, "CV3 High Limit"
prob += G31*MV1+G32*MV2 >= CV3Lo, "CV3 Low Limit"
prob += MV1 >= MV1Lo, "MV1 Low Limit"
prob += MV1 <= MV1Hi, "MV1 High Limit"
prob += MV2 >= MV2Lo, "MV2 Low Limit"
prob += MV2 <= MV2Hi, "MV2 High Limit"

# The problem is solved using PuLP's choice of Solver
status = prob.solve(PULP_CBC_CMD(msg=0))

if LpStatus[status] != "Optimal":
    soln = [0,0]
    V = 0
else:
    soln = [v.varValue for v in prob.variables()]
    V = prob.objective.value()

# Iso-profit line passing through the optimum solution V = Cost1*X_opt + Cost2*Y_opt
xspace_line = np.linspace(-xlimits, xlimits, 10)
y_obj = (1/MV2Cost)*V - (MV1Cost/MV2Cost)*xspace_line

# The vector field
dvecx = np.linspace(-xlimits, xlimits, 50)
dvecy = np.linspace(-ylimits, ylimits, 50)
xv, yv = np.meshgrid(dvecx, dvecy)

def dir_text(soln):
    return "<span style='color:blue'>⬆</span> Up" if soln > 0 else "<span style='color:red'>⬇</span> Down"

def dollar_formatter(x, pos):
    if x < 0:
        return f"\u2212${abs(x):,.0f}"
    else:
        return f"${x:,.0f}"

o = {}
# st.text(prob.constraints.items())
for name, c in prob.constraints.items():
    o[name] = {'shadow price':c.pi, 'slack': abs(c.slack)}
    # st.write(name, c.slack)
# st.text(o)

constrained = {}
for var in ['MV1', 'MV2', 'CV1', 'CV2', 'CV3']:
    loslack = o[var+'_Low_Limit']['slack']
    hislack = o[var+'_High_Limit']['slack']
    
    if np.isclose(abs(loslack), 0, rtol=1e-20):
        constrained[var] = "Lo Limit"
    elif np.isclose(abs(hislack), 0, rtol=1e-20):
        constrained[var] = "Hi Limit"
    else:
        constrained[var] = "Normal"

def color_constraint(val):
    color = 'lightblue' if 'Limit' in val else ''
    return f"background-color: {color}"

def highlight_lo(x):
    df2 = pd.DataFrame('', index=x.index, columns=x.columns)
    mask = x['Status'] == 'Lo Limit'
    df2.loc[mask, 'LoLim'] = 'background-color: lightblue'
    return df2

def highlight_hi(x):
    df2 = pd.DataFrame('', index=x.index, columns=x.columns)
    mask = x['Status'] == 'Hi Limit'
    df2.loc[mask, 'HiLim'] = 'background-color: lightblue'
    return df2  

df = pd.DataFrame.from_dict(constrained, orient='index', columns=["Status"])

df.loc['MV1', 'LoLim'] = MV1LoSS
df.loc['MV2', 'LoLim'] = MV2LoSS
df.loc['CV1', 'LoLim'] = CV1LoSS
df.loc['CV2', 'LoLim'] = CV2LoSS
df.loc['CV3', 'LoLim'] = CV3LoSS
df.loc['MV1', 'PV'] = st.session_state['MV1SSVal']
df.loc['MV2', 'PV'] = st.session_state['MV2SSVal']
df.loc['CV1', 'PV'] = st.session_state['CV1SSVal']
df.loc['CV2', 'PV'] = st.session_state['CV2SSVal']
df.loc['CV3', 'PV'] = st.session_state['CV3SSVal']
df.loc['MV1', 'SSTarget'] = soln[0] + st.session_state['MV1SSVal']
df.loc['MV2', 'SSTarget'] = soln[1] + st.session_state['MV2SSVal']
df.loc['CV1', 'SSTarget'] = G11*soln[0]+G12*soln[1] + st.session_state['CV1SSVal']
df.loc['CV2', 'SSTarget'] = G21*soln[0]+G22*soln[1] + st.session_state['CV2SSVal']
df.loc['CV3', 'SSTarget'] = G31*soln[0]+G32*soln[1] + st.session_state['CV3SSVal']
df.loc['MV1', 'HiLim'] = MV1HiSS
df.loc['MV2', 'HiLim'] = MV2HiSS
df.loc['CV1', 'HiLim'] = CV1HiSS
df.loc['CV2', 'HiLim'] = CV2HiSS
df.loc['CV3', 'HiLim'] = CV3HiSS
df.loc['MV1', 'Move'] = dir_text(soln[0])
df.loc['MV2', 'Move'] = dir_text(soln[1])
df.loc['CV1', 'Move'] = dir_text(G11*soln[0]+G12*soln[1])
df.loc['CV2', 'Move'] = dir_text(G21*soln[0]+G22*soln[1])
df.loc['CV3', 'Move'] = dir_text(G31*soln[0]+G32*soln[1])
df.loc['MV1', 'Delta'] = soln[0]
df.loc['MV2', 'Delta'] = soln[1]
df.loc['CV1', 'Delta'] = G11*soln[0]+G12*soln[1]
df.loc['CV2', 'Delta'] = G21*soln[0]+G22*soln[1]
df.loc['CV3', 'Delta'] = G31*soln[0]+G32*soln[1]

df.rename(index={
    'MV1': f"MV - {varMV1['Name']} ({varMV1['UOM']})", 
    'MV2': f"MV - {varMV2['Name']} ({varMV2['UOM']})", 
    'CV1': f"CV - {varCV1['Name']} ({varCV1['UOM']})", 
    'CV2': f"CV - {varCV2['Name']} ({varCV2['UOM']})", 
    'CV3': f"CV - {varCV3['Name']} ({varCV3['UOM']})"}, 
    inplace=True)

df = df.style.apply(highlight_lo, axis=None, subset=['Status', 'LoLim'])\
             .apply(highlight_hi, axis=None, subset=['Status', 'HiLim'])\
             .map(color_constraint, subset=['Status'])\
             .format(dict.fromkeys(df.select_dtypes('float').columns, "{:.2f}"))\
             .to_html()

def plotLP(showVector=True, showOptimum=True):
    fig, ax = plt.subplots(figsize=(5,5))
    
    if st.session_state["shade_MV1"]:
        ax.imshow(m1, extent=(x.min(),x.max(),y.min(),y.max()), aspect='auto', origin="lower", cmap=mcolors.ListedColormap(['none', 'yellow']), alpha=0.10)
    if st.session_state["shade_MV2"]:
        ax.imshow(m2, extent=(x.min(),x.max(),y.min(),y.max()), aspect='auto', origin="lower", cmap=mcolors.ListedColormap(['none', 'yellow']), alpha=0.10)
    if st.session_state["shade_CV1"]:
        ax.imshow((c1 & c2).astype(int), extent=(x.min(),x.max(),y.min(),y.max()), aspect='auto', origin="lower", cmap="Reds", alpha=0.15)
    if st.session_state["shade_CV2"]:
        ax.imshow((c3 & c4).astype(int), extent=(x.min(),x.max(),y.min(),y.max()), aspect='auto', origin="lower", cmap="Blues", alpha=0.15)
    if st.session_state["shade_CV3"]:
        ax.imshow((c5 & c6).astype(int), extent=(x.min(),x.max(),y.min(),y.max()), aspect='auto', origin="lower", cmap="Greens", alpha=0.15)
    c1h, = ax.plot(xspace, y_c1, '-r', label='CV1_Hi');
    c1l, = ax.plot(xspace, y_c2, '--r', label='CV1_Lo');
    c2h, = ax.plot(xspace, y_c3, '-b', label='CV2_Hi');
    c2l, = ax.plot(xspace, y_c4, '--b', label='CV2_Lo');
    c3h, = ax.plot(xspace, y_c5, '-g', label='CV3_Hi');
    c3l, = ax.plot(xspace, y_c6, '--g', label='CV3_Lo');    

    ax.axvline(x=0, color='black', lw=0.2, linestyle='-')
    ax.axhline(y=0, color='black', lw=0.2, linestyle='-')

    m1l = ax.axvline(x=MV1Lo, color='olive', lw=1, linestyle='--', label=f"MV1_Lo")
    m1h = ax.axvline(x=MV1Hi, color='olive', lw=1, linestyle='-', label=f"MV1_Hi")
    m2l = ax.axhline(y=MV2Lo, color='olive', lw=1, linestyle='--', label=f"MV2_Lo")
    m2h = ax.axhline(y=MV2Hi, color='olive', lw=1, linestyle='-', label=f"MV2_Hi")

    ax.plot(0,0,'kx');

    # the obj func
    if not LpStatus[status] != "Optimal":
        if st.session_state["show_optimum"]:
            ax.plot(soln[0], soln[1], 'Dk', ms=6);
        if st.session_state["show_isoprofit"]:
            ax.plot(xspace_line, y_obj, ':k', lw=2);
            ax.quiver(soln[0], soln[1], -MV1Cost, -MV2Cost*((ylimits / xlimits)),
                headwidth=6, width=0.005, alpha=0.95, scale_units='xy', scale=6)

        # plot vector field to show the direction of optimization (direction of decreasing cost)
        if st.session_state["show_vectors"]:
            # countour_obj = -((MV1Cost * x) + (MV2Cost * y))
            # z_obj = -z_obj * abs(-V/np.max(abs(z_obj)))

            mask_threshold = 4
            mask = (MV1Cost * xv) + (MV2Cost * yv) - mask_threshold <= V
            x_masked = xv[~mask]
            y_masked = yv[~mask]
            z_obj        = -((MV1Cost * x_masked) + (MV2Cost * y_masked))

            # center the cmap around $0
            vmin = np.min(z_obj)
            vmax = np.max([50, np.max(z_obj)]) # set to 50 for reasonable high range on cmap when z_obj val is low
            if vmin < 0 and vmax > 0:
                # Data spans negative and positive → center at 0
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            else:
                # Data is all positive or all negative → linear scaling
                norm = Normalize(vmin=vmin, vmax=vmax)

            q = ax.quiver(x_masked, y_masked, -MV1Cost, -MV2Cost*((ylimits / xlimits)), z_obj, norm=norm, cmap='RdYlGn',
                headwidth=4, width=0.0025, alpha=0.45, scale_units='xy', scale=20)
            pos = ax.get_position()  # [left, bottom, width, height]

            # Create a new axes below the main plot
            cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.13, pos.width, 0.03])  # 0.05 below 
            cbar = fig.colorbar(q, cax=cbar_ax, orientation='horizontal', label="Profit ($) for this move")
            if vmin < 0 and vmax > 0:
                cbar.set_ticks([vmin, vmin*3/4, vmin/2, vmin/4, 0, vmax/4, vmax/2, vmax*3/4, vmax])
            cbar_ax.xaxis.set_major_formatter(dollar_formatter)

    # must be on top layer
    feasible_mask = (c1 & c2 & c3 & c4 & c5 & c6 & m1 & m2).astype(int)
    ax.imshow(feasible_mask, extent=(x.min(),x.max(),y.min(),y.max()), aspect='auto', origin="lower", cmap="binary", alpha=0.10);
    if st.session_state["shade_feasible"]:
        ax.contourf(x, y, feasible_mask, levels=[0.5, 1], colors=['none'], hatches=["///"], alpha=0)

    ax.set_xlim((-xlimits, xlimits))
    ax.set_ylim((-ylimits, ylimits))
    ax.set_xlabel(f"MV1 Move: $\Delta${varMV1['Name']} ({varMV1['UOM']})")
    ax.set_ylabel(f"MV2 Move: $\Delta${varMV2['Name']} ({varMV2['UOM']})")
    ax.set_aspect('auto')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    ax.legend(
        [(c1h, c1l), (c2h, c2l), (c3h, c3l), m1l], 
        [f"CV1: {varCV1['Name']}",
         f"CV2: {varCV2['Name']}",
         f"CV3: {varCV3['Name']}",
         'MV Limits'],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc='lower center', 
        bbox_to_anchor=(0.5, 1.02), 
        ncol=5
    )

    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=5)
    return fig

tab1, tab2 = st.tabs(["📈 Simulation", "🕵 Gain Matrix"]) #, "🕵 Shadow Prices"]) #, "🕵 Scenarios"])

with tab2:
    # For SVD plot
    # G = np.array([[G11,G12], [G31,G32]])
    # t = np.linspace(0,np.pi*2,200)
    # u, s, Vt = np.linalg.svd(G, full_matrices=True)

    cols=st.columns([0.5, 0.5])
    with cols[0]:
        st.subheader("Gain Matrix")
        st.markdown(r"The $3\times2$ gain matrix, $G$ with 2 MVs and 3 CVs, has 6 elements, where each element $G_{ij}$ describes the *steady-state* relationship between $\text{CV}_{i}$ and $\text{MV}_{j}$.")   
        G11 = st.session_state["G11"]
        G12 = st.session_state["G12"]
        G21 = st.session_state["G21"]
        G22 = st.session_state["G22"]
        G31 = st.session_state["G31"]
        G32 = st.session_state["G32"]

        varvals = {f"{varMV1['Name']}": [G11, G21, G31], f"{varMV2['Name']}": [G12, G22, G32]}
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.dataframe(pd.DataFrame(varvals, index=[f"{varCV1['Name']}", f"{varCV2['Name']}", f"{varCV3['Name']}"]).style.format("{:.3f}"))

        st.markdown("The equation relating the CVs and MVs through the gain matrix is given by:")
        st.latex(r"\Delta CV = G \cdot \Delta MV")  
        st.latex(rf"G = \begin{{bmatrix}} \
                G_{{11}} & G_{{12}} \\ \
                G_{{21}} & G_{{22}} \\ \
                G_{{31}} & G_{{32}} \
            \end{{bmatrix}} = \begin{{bmatrix}} \
                {G11:.3f} & {G12:.3f} \\ \
                {G21:.3f} & {G22:.3f} \\ \
                {G31:.3f} & {G32:.3f} \
            \end{{bmatrix}}")
        st.markdown("Using the gain matrix, the CV relationships can be written in terms of its MVs:")
        st.latex(rf"""
            \begin{{align}}
                \Delta \text{{{varCV1['Name']}}} &= {G11:.3f} \cdot \Delta \text{{{varMV1['Name']}}} + {G12:.3f} \cdot \Delta \text{{{varMV2['Name']}}}\\ 
                \Delta \text{{{varCV2['Name']}}} &= {G21:.3f} \cdot \Delta \text{{{varMV1['Name']}}} + {G22:.3f} \cdot \Delta \text{{{varMV2['Name']}}}\\
                \Delta \text{{{varCV3['Name']}}} &= {G31:.3f} \cdot \Delta \text{{{varMV1['Name']}}} + {G32:.3f} \cdot \Delta \text{{{varMV2['Name']}}}            
            \end{{align}}
            """)


    # cols=st.columns([0.65, 0.35])
    # with cols[0]:
    #     st.subheader("Matrix Conditioning")
    #     st.markdown("In this example, let's consider a 2x2 gain matrix for simplicity.")
    #     st.latex(rf"G_A = \begin{{bmatrix}} \
    #             G_{{11}} & G_{{12}} \\ \
    #             G_{{31}} & G_{{32}} \
    #         \end{{bmatrix}} = \begin{{bmatrix}} \
    #             {G11:.3f} & {G12:.3f} \\ \
    #             {G31:.3f} & {G32:.3f} \
    #         \end{{bmatrix}}")        
    #     st.info("Use the controls on the left to change the gain matrix and see the impact on the CVs and the shape of the CV response ellipse.")    
    #     st.markdown("Singular Value Decomposition (SVD) breaks down the gain matrix $G$ into 3 matrices which have some nice mathematical properties [[Moore (1986)](https://ieeexplore.ieee.org/abstract/document/4789019)] for process control.")
    #     st.latex(r"G = U \Sigma V^T")

    #     st.markdown("""
    #         - In $U$, the first column vector tells the strongest direction of CV movements possible for this system, and the second column gives the second strongest direction, and so on.
    #         - In $V^T$, the first row vector tells us the MV movements required to achieve the strongest direction of CV movement, and the second row vector is the MV movements needed for the second strongest direction, and so on.
    #         - In $\Sigma$, the singular values tell us the magnitude of these directions.

    #     """)
    #     st.markdown(f"A geometric interpretation of the SVD is shown below. The left subplot shows a unit circle of MV movements. The right subplot shows the impact of the corresponding MV movements on the CVs. For example, point A is a unit increase in Reflux FC with no change to Reboiler TC, which causes RVP to decrease by {G12:.2f} and C5 % to decrease by {G22:.2f}.")
    #     st.markdown("The shape of the CV response ellipse tells us that there are strong and weak control directions. For this particular system, it is much more difficult to control the RVP compared to the C5. To get a 1 unit change in C5, we need about 10 times the magnitude of MV movement compared to making a 1 unit change in RVP.")

    #     st.latex(rf"\begin{{bmatrix}} \
    #             {G11:.3f} & {G12:.3f} \\ \
    #             {G31:.3f} & {G32:.3f} \
    #         \end{{bmatrix}} = \
    #         \left.\vphantom{{\begin{{matrix}} {u[0][0]:.3f} \\ {u[1][0]:.3f} \end{{matrix}}}}\right[\
    #         \overbrace{{ \begin{{matrix}} {u[0][0]:.3f} \\ {u[1][0]:.3f} \end{{matrix}}}}^{{u_1}}\
    #         \overbrace{{ \begin{{matrix}} {u[0][1]:.3f} \\ {u[1][1]:.3f} \end{{matrix}}}}^{{u_2}}\
    #         \left.\vphantom{{\begin{{matrix}} {u[0][1]:.3f} \\ {u[1][1]:.3f} \end{{matrix}}}}\right]\
    #         \
    #         \left.\vphantom{{\begin{{matrix}} {s[0]:.3f} \\ {0:d} \end{{matrix}}}}\right[\
    #         \overbrace{{ \begin{{matrix}} {s[0]:.3f} \\ {0:d} \end{{matrix}}}}^{{\sigma_1}}\
    #         \underbrace{{ \begin{{matrix}} {0:d} \\ {s[1]:.3f} \end{{matrix}}}}_{{\sigma_2}}\
    #         \left.\vphantom{{\begin{{matrix}} {0:d} \\ {s[1]:.3f} \end{{matrix}}}}\right]\
    #         \
    #         \begin{{bmatrix}} \
    #             {Vt[0][0]:.3f} & {Vt[0][1]:.3f} \\ \
    #             {Vt[1][0]:.3f} & {Vt[1][1]:.3f} \
    #         \end{{bmatrix}}\!\!\
    #         \begin{{matrix}} v_1 \\ v_2 \end{{matrix}}")    

    # with cols[1]:
    #     fig = plotSVD(G)
    #     st.pyplot(fig)

    # st.info(f"The gain matrix tells us that: Every unit increase in reboiler Temp reduces RVP by {G11:.2f} units and increases C5 by {G21:.2f} units. This makes sense, if we fire the reboiler harder, we boil stuff up the top so the bottom RVP decreases (less volatile, more heavier components), and some of the heavier components go up the column, so overhead C5 increases. Every unit increase in reflux reduces RVP by {G12:.2f} units and reduces C5 by {G22:.2f} units. This also makes sense, because increasing reflux improves separation and washes down the heavier materials from the top. Separation is improved up to a certain point, considering flooding limits etc. **This time, our controller is also constrained by the dP limits.**") 

with tab1:
    # st.badge("Settings")
    # st.header("Interactive LP Simulation")
    # st.markdown("In this module, you can adjust the operating limits, gain matrix and LP costs to study the response of the LP solution. We will explore the effects of **'clamping'** the limits of the dP and how it impacts the LP solution and optimization directions. Initially, the dP limits are wide open, and does not impact the optimum point or the feasible region.")
    # st.info("**Case Study**: What happens when you reduce the dP upper limit? Reducing the upper limit simulates a possible operational scenario where the tower is flooded or overloaded.")
    cols=st.columns([0.55, 0.45])
    with cols[1]:
        # runLP()
        fig = plotLP()
        st.pyplot(fig)

        if st.session_state["shade_MV1"] or st.session_state["shade_MV2"]:
            st.info("MV limits are *hard constraints* that cannot be violated.")

        if st.session_state["shade_CV1"] or st.session_state["shade_CV2"] or st.session_state["shade_CV3"]:
            st.info("CV limits are *soft constraints* that can be **given up** if the LP problem is infeasible.")

        if st.session_state["shade_feasible"]:
            st.info("""
                #### Feasible Region
                Each MV/CV high limit and low limit can be plotted as a shaded region between 2 straight lines.\
                If we take the intersection of all shaded regions formed by each constraint,\
                we get the feasible region or solution space.\
                The feasible region tells us that DMC is allowed to make MV moves on $\Delta MV_1$ and $\Delta MV_2$ within that space without violating any limits.                
            """)

    with cols[0]:
        if LpStatus[status] != "Optimal":
            st.error(f"⚠ Status: {LpStatus[status]}. No feasible LP solution found.")
        else:        
            st.badge("LP Solution Table")
            # st.markdown("**LP Solution Table**")
            st.write(df, unsafe_allow_html=True)
            # st.markdown(f"""
            #     - DMC LP will move MV1 **$\Delta${varMV1['Name']}:** {dir_text(soln[0])} by {soln[0]:.1f}{varMV1['UOM']}
            #     - DMC LP will move MV2 **$\Delta${varMV2['Name']}:** {dir_text(soln[1])} by {soln[1]:.0f} {varMV2['UOM']}
            # """, unsafe_allow_html=True)

        # st.divider()
        st.html("<hr><b>LP Visualization Details</b><hr>")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            shade_feasible = st.checkbox(f"Feasible Region", key='shade_feasible')
        with col2:
            show_optimum = st.checkbox(f"Optimum Point", key='show_optimum')
        with col3:
            show_vectors = st.checkbox(f"Profit Direction", key='show_vectors')
        with col4:
            show_isoprofit = st.checkbox(f"Iso-Profit Line", key='show_isoprofit')        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            shade_MV1 = st.checkbox(f"MV1: {varMV1['Name']}", key='shade_MV1')

        with col2:
            shade_MV2 = st.checkbox(f"MV2: {varMV2['Name']}", key='shade_MV2')
        with col3:
            shade_CV1 = st.checkbox(f"CV1: {varCV1['Name']}", key='shade_CV1')
        with col4:        
            shade_CV2 = st.checkbox(f"CV2: {varCV2['Name']}", key='shade_CV2')
        with col5:          
            shade_CV3 = st.checkbox(f"CV3: {varCV3['Name']}", key='shade_CV3')

        if st.session_state["shade_MV1"]:
            st.latex(rf"""
                \text{{{varMV1['Name']} Constraint: }} {MV1LoSS} \leq {{{st.session_state['MV1SSVal']:.2f}}} + \Delta\text{{{varMV1['Name']}}} \leq {MV1HiSS}
                """)
        if st.session_state["shade_MV2"]:
            st.latex(rf"""
                \text{{{varMV2['Name']} Constraint: }} {MV2LoSS} \leq {{{st.session_state['MV2SSVal']:.2f}}} + \Delta\text{{{varMV2['Name']}}} \leq {MV2HiSS}
                """)
        if st.session_state["shade_CV1"]:
            st.latex(rf"""
                \text{{{varCV1['Name']} Constraint: }} {CV1LoSS} \leq {{{st.session_state['CV1SSVal']:.2f}}} + \Delta\text{{{varCV1['Name']}}} \leq {CV1HiSS}
                """)
        if st.session_state["shade_CV2"]:
            st.latex(rf"""
                \text{{{varCV2['Name']} Constraint: }} {CV2LoSS} \leq {{{st.session_state['CV2SSVal']:.2f}}} + \Delta\text{{{varCV2['Name']}}} \leq {CV2HiSS}
                """)
        if st.session_state["shade_CV3"]:
            st.latex(rf"""
                \text{{{varCV3['Name']} Constraint: }} {CV3LoSS} \leq {{{st.session_state['CV3SSVal']:.2f}}} + \Delta\text{{{varCV3['Name']}}} \leq {CV3HiSS}
                """)            

        if st.session_state["show_isoprofit"]:
            st.markdown(f"""
                #### Isoprofit Line
                The objective function is linear,\
                so its slope will always be constant along any given direction in the xy plane.\
                Each value of the objective function will give a different straight line. These lines are all parallel to each other. They are isoprofit lines, meaning the value of the objective function is the same anywhere on this line.\
                The LP will drive the system in the direction of maximum cost reduction, which is orthogonal to the isoprofit lines (see the vector field arrows).\
            """)

        if st.session_state["show_vectors"]:
            st.markdown(fr"""
                #### Direction of increasing profits
                The vector field (arrows) show the direction of increasing profits which are better solutions.

                The LP costs set the preferred direction of optimization. 
                As a rule of thumb, a negative LP cost would incentivize the DMC LP to maximize that variable *(preferred, but not guaranteed!)*, 
                and likewise, a positive cost would incentivize the DMC LP to minimize that variable.
            """)

        if st.session_state["show_optimum"]:
            st.markdown(fr"""
                #### LP Optimum Point
                Out of all possible points in the feasible region, which one should the optimizer pick?
                DMC solves an **optimization problem** called a **Linear Program (LP)** to decide. The objective function $f$ is to minimize the overall cost of MV movements $\Delta MV$ where the cost for each unit of movement of an MV is $c$.

                $$\min f= c_1 \Delta MV_1 + c_2 \Delta MV_2$$

                Optimum $${varMV1['Name']}^{{OPT}} = {st.session_state['MV1SSVal'] + soln[0]:.2f}$$ {varMV1['UOM']}

                Optimum $${varMV2['Name']}^{{OPT}} = {st.session_state['MV2SSVal'] + soln[1]:.2f}$$ {varMV2['UOM']}
            """)

        # st.header("Optimization Case Studies")
        # st.subheader("Case Study 1: Clamp dP Upper Limit")
        # st.markdown("What happens to the optimization direction when the upper dP limit is clamped? Use the buttons below or the control panel on the left sidebar to explore this scenario.")
        # cols=st.columns([0.5, 0.5])
        # with cols[0]:
        #     st.button('Clamp dP Limit', on_click=clamp_dp, use_container_width=True)
        # with cols[1]:
        #     st.button('Reset dP Limit', on_click=unclamp_dp, use_container_width=True)
        # st.markdown("We can see that, when the dP is clamped:")
        # st.markdown("""
        #     - The C5 and RVP are no longer controlling to the upper limit.
        #     - The controller reduces reflux to offload the column. 
        #     """)

def change_ref_lp():
    st.session_state["MV2Cost"] = -0.1

def reverse_ref_lp():
    st.session_state["MV2Cost"] = -1.0    

# with tab3:
#     cols=st.columns([0.5, 0.5])
#     with cols[0]:
#         st.header("Shadow Prices")
#         st.markdown(r"In Linear Programming theory, the shadow price of a constraint is defined as the change in objective function for each engineering unit of moving a limit.")
#         st.latex(r"\text{Shadow Price} = \Delta\text{Obj}/\Delta\text{Limit}")
#         st.markdown("By definition, the shadow price of a **non-binding** constraint, which is a variable not at its limit, is equal to 0.")

#         st.header("Case Study 2: Shadow Price Calculation")
#         st.markdown("Consider the default simulation limits, but now with a new LP cost of -0.1 for the FG. How does that impact the LP solution? Did the constraints change? What are the new binding constraints (i.e. variables at their limits), and what is the shadow price of this new variable? Use the diagram and tables on the right as a reference.")
#         innercols=st.columns([0.5, 0.5])
#         with innercols[0]:
#             st.button('Set Reflux LP Cost to -0.1', on_click=change_ref_lp, use_container_width=True)
#         with innercols[1]:
#             st.button('Reset Reflux LP Cost to -1.0', on_click=reverse_ref_lp, use_container_width=True)      
#         st.subheader("Discussion")
#         st.markdown("Since we know the LP costs, we can actually calculate the shadow price by hand, using the change in coordinates of the LP solution.")
#         st.latex(r"\text{Shadow Price} = \Delta\text{Obj}/\Delta\text{Limit}")       

#     with cols[1]:
#         innercols=st.columns([0.5, 0.5])
#         # with innercols[0]:
#         fig = plotLP()
#         st.pyplot(fig)
#         st.write(df, unsafe_allow_html=True)
#         st.markdown("#### Optimization Directions")
#         st.markdown(f"""
#             - **O2:** {dir_text(soln[0])} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **FG:** {dir_text(soln[1])}
#             - Value of Objective Function (Profit): ${-(V):.2f}
#             - Coodinates of Optimum Point: ({soln[0]:.3f}, {soln[1]:.3f})
#         """, unsafe_allow_html=True)        
#         st.text("\n")
#         st.markdown("Shadow Prices")
#         st.dataframe(pd.DataFrame(o).T)         