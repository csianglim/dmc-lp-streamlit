import streamlit as st
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from pulp import LpVariable, LpProblem, LpMinimize, value, PULP_CBC_CMD
st.set_page_config(layout="wide")

CV1SSVal = 9.0
CV2SSVal = 2.5
CV3SSVal = 4.0
MV1SSVal = 300
MV2SSVal = 11.0

if "CV3Limits" not in st.session_state:
    st.session_state["CV3Limits"] = (3.6, 4.4)

CVStep = 0.05
with st.sidebar:
    st.header("Model & Tuning Parameters")
    with st.expander('Steady-State Gains'):
        cols=st.columns(2)
        with cols[0]:
            st.number_input('$G_{11}$: Reboiler TC-RVP', key="G11", value=-0.200, step=0.10, format="%.3f")
        with cols[1]:
            st.number_input('$G_{12}$: Reflux FC-RVP', key="G12", value=-0.072, step=0.10, format="%.3f")
        cols=st.columns(2)
        with cols[0]:
            st.number_input('$G_{21}$: Reboiler TC-C5', key="G21", value=0.125, step=0.10, format="%.3f")
        with cols[1]:
            st.number_input('$G_{22}$: Reflux FC-C5', key="G22", value=-0.954, step=0.10, format="%.3f")
        cols=st.columns(2)
        with cols[0]:
            st.number_input('$G_{31}$: Reboiler TC-dP', key="G31", value=0.025, step=0.01, format="%.3f")
        with cols[1]:
            st.number_input('$G_{32}$: Reflux FC-dP', key="G32", value=0.101, step=0.01, format="%.3f")

    with st.expander('LP Costs'):
        cols=st.columns(2)
        with cols[0]:
            MV1Cost = st.number_input(r'Reboiler TC Cost', value=-1.0, step=0.1)
        with cols[1]:
            MV2Cost = st.number_input(r'Reflux FC Cost', value=-1.0, step=0.1)  


    st.subheader('Manipulated Variables')
    with st.columns(1)[0]:
        MV1LoSS, MV1HiSS = st.slider(r'Reboiler TC Limits (°F)', 290.0, 310.0, (291.0, 309.0))
    with st.columns(1)[0]:
        MV2LoSS, MV2HiSS = st.slider(r'Reflux FC Limits (MBPD)', 1.0, 21.0, (2.0, 20.0))        
    
    st.subheader('Controlled Variables')
    with st.columns(1)[0]:
        CV1LoSS, CV1HiSS = st.slider(r'RVP Limits (psi)', 7.5, 10.0, (8.0, 9.5), step=CVStep)
    with st.columns(1)[0]:
        CV2LoSS, CV2HiSS = st.slider(r'C5 Limits (%)', 0.5, 5.0, (1.0, 4.5), step=CVStep)
    with st.columns(1)[0]:
        st.slider(r'dP Limits (psig)', 3.5, 4.5, step=0.01, key="CV3Limits")

    st.divider()
    st.markdown('Last Updated: Siang Lim (August 2023). Source on [GitHub](https://github.com/csianglim/dmc-lp-streamlit).')

# Grab vars
G11 = st.session_state["G11"]
G12 = st.session_state["G12"]
G21 = st.session_state["G21"]
G22 = st.session_state["G22"]
G31 = st.session_state["G31"]
G32 = st.session_state["G32"]
CV3LoSS, CV3HiSS = st.session_state["CV3Limits"]

limits = 10
d = np.linspace(-limits, limits, 100)
x,y = np.meshgrid(d,d)
stepsize = 0.1
v_new = 0

CV1Hi = CV1HiSS - CV1SSVal
CV1Lo = CV1LoSS - CV1SSVal
CV2Hi = CV2HiSS - CV2SSVal
CV2Lo = CV2LoSS - CV2SSVal
CV3Hi = CV3HiSS - CV3SSVal
CV3Lo = CV3LoSS - CV3SSVal

MV1Hi = MV1HiSS - MV1SSVal
MV1Lo = MV1LoSS - MV1SSVal
MV2Hi = MV2HiSS - MV2SSVal
MV2Lo = MV2LoSS - MV2SSVal

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
y_c1 = (CV1Hi - G11*d)/G12
y_c2 = (CV1Lo - G11*d)/G12
y_c3 = (CV2Hi - G21*d)/G22
y_c4 = (CV2Lo - G21*d)/G22
y_c5 = (CV3Hi - G31*d)/G32
y_c6 = (CV3Lo - G31*d)/G32

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
prob.solve(PULP_CBC_CMD(msg=0))
soln = [v.varValue for v in prob.variables()]
V = prob.objective.value()

# Each of the variables is printed with it's resolved optimum value
#     for v in prob.variables():
#         print(v.name, "=", v.varValue)

dvec = np.linspace(-limits + (0.1*limits), limits - (0.1*limits), 12)
xv, yv = np.meshgrid(dvec, dvec)
y_obj = (1/MV2Cost)*V - (MV1Cost/MV2Cost)*d
z_obj = (MV1Cost * xv) + (MV2Cost * yv)

def plotLP(showVector=True, showOptimum=True):
    fig, ax = plt.subplots()
    ax.imshow((c1 & c2 & c3 & c4 & c5 & c6 & m1 & m2).astype(int), extent=(x.min(),x.max(),y.min(),y.max()), origin="lower", cmap="binary", alpha=0.1);
    ax.plot(d, y_c1, '-r', label='CV1_Hi');
    ax.plot(d, y_c2, '--r', label='CV1_Lo');
    ax.plot(d, y_c3, '-b', label='CV2_Hi');
    ax.plot(d, y_c4, '--b', label='CV2_Lo');
    ax.plot(d, y_c5, '-y', label='CV3_Hi');
    ax.plot(d, y_c6, '--y', label='CV3_Lo');    

    ax.axvline(x=0, color='black', lw=0.1, linestyle='--')
    ax.axhline(y=0, color='black', lw=0.1, linestyle='--')

    ax.axvline(x=MV1Lo, color='green', lw=1, linestyle=':', label=f'MV1_Lo')
    ax.axvline(x=MV1Hi, color='green', lw=1, linestyle=':', label=f'MV1_Hi')
    ax.axhline(y=MV2Lo, color='green', lw=1, linestyle=':', label=f'MV2_Lo')
    ax.axhline(y=MV2Hi, color='green', lw=1, linestyle=':', label=f'MV2_Hi')

    ax.plot(0,0,'kx');

    # the obj func
    if showOptimum:
        ax.plot(soln[0], soln[1], 'ok');
        ax.plot(d, y_obj, '-.k');

    # plot vector field to show the direction of optimization (direction of decreasing cost)
    if showVector:
        ax.quiver(xv, yv,-MV1Cost,-MV2Cost, z_obj, cmap='gray', headwidth=4, width=0.003, scale=40, alpha=0.35)

    ax.set_xlim((-limits, limits))
    ax.set_ylim((-limits, limits))
    ax.set_xlabel('Reboiler TC Moves')
    ax.set_ylabel('Reflux FC Moves')
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')); # No decimal places
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')); # No decimal places

    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=5);
    return fig

# Conditioning plots
G = np.array([[G11,G12], [G21,G22]])
t = np.linspace(0,np.pi*2,200)
X = np.array(list(zip(np.sin(t), np.cos(t))))
Y = (G@X.T)
u, s, Vt = np.linalg.svd(G, full_matrices=True)
u_scaled = u @ np.diag(s)
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(10,5))

ax1.plot(np.cos(t), np.sin(t), '-k', linewidth=1.5)
ax1.plot(0,0,'ok')
ax1.axvline(x=0, color='black', lw=0.5, linestyle='--')
ax1.axhline(y=0, color='black', lw=0.5, linestyle='--')

ax2.plot(Y[0,:],Y[1,:], '-k', linewidth=1.5)
ax2.plot(0,0,'ok')
ax2.axvline(x=0, color='black', lw=0.5, linestyle='--')
ax2.axhline(y=0, color='black', lw=0.5, linestyle='--')

ax1.annotate("", xy=Vt[0,:], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='r', lw=1.5))
ax1.annotate("", xy=Vt[1,:], xytext=(0, 0), arrowprops=dict(arrowstyle="->", linestyle="--",  color='b', lw=1.5))
ax2.annotate("", xy=u_scaled[:,0], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color='r', lw=1.5))
ax2.annotate("", xy=u_scaled[:,1], xytext=(0, 0), arrowprops=dict(arrowstyle="->", linestyle="--",  color='b', lw=1.5))

ax1.annotate("$v_1$", xy=Vt[0,:]/2, xycoords="data", color="red",
                  va="center", ha="center", bbox=dict(boxstyle="round", fc="w", edgecolor='none', pad=0.1))
ax1.annotate("$v_2$", xy=Vt[1,:]/2, xycoords="data", color='blue',
                  va="center", ha="center", bbox=dict(boxstyle="round", fc="w", edgecolor='none', pad=0.1))
ax2.annotate("$\sigma_1 u_1$", xy=u_scaled[:,0]/2, xycoords="data", color="red",
                  va="center", ha="center", bbox=dict(boxstyle="round", fc="w", edgecolor='none', pad=0.1))
ax2.annotate("$\sigma_2 u_2$", xy=u_scaled[:,1]/2, xycoords="data", color='blue',
                  va="center", ha="center", bbox=dict(boxstyle="round", fc="w", edgecolor='none', pad=0.1))

labels = ['$A$', '$B$', '$C$', '$D$']
points = [[0,1], [1,0], [0,-1], [-1,0]]

for i, p1 in enumerate(points):
    ax1.plot(p1[0], p1[1], color='k', marker='o', markersize=8, markerfacecolor='y', alpha=0.65)
    ax2.plot((G@p1)[0], (G@p1)[1], color='k', marker='o', markersize=8, markerfacecolor='y', alpha=0.65)

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
                 # textcoords="offset points", # how to position the text
                 # xytext=(2,2),
                 bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                 ha='center')

    ax2.annotate(labels[i], # this is the text
                 ((G@p1)[0]+(np.sign((G@p1)[0])*0.1), (G@p1)[1]+(np.sign((G@p1)[1])*0.1)), # these are the coordinates to position the label
                 # textcoords="offset points", # how to position the text
                 # xytext=((G@p1)[0],(G@p1)[1]), # distance from text to points (x,y)
                 bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                 ha='center')    

ax1.set_xlabel("$\Delta MV_1$")
ax1.set_ylabel("$\Delta MV_2$")
ax2.set_xlabel("$\Delta CV_1$")
ax2.set_ylabel("$\Delta CV_2$")

for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'));
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'));
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_box_aspect(1);

st.title('Multivariable Controllers - LP-DMC (Part 2: Constraint Relief)')
st.subheader('Interpreting multivariable controller actions: an interactive tutorial.')


tab1, tab4 = st.tabs(["🎓 Introduction", "📈 Simulation"]) #, "🕵 Scenarios"])

with tab1: #st.expander('DMC', expanded=True):
    row1=st.columns([0.55, 0.45])

with row1[1]:
    st.image('column2.png', caption='Figure 1: Simple debutanizer column. Details see: https://arxiv.org/abs/2202.13291')
with row1[0]:
    st.header("Debutanizer Column - now with 3 CVs")
    st.markdown("""
        In the [previous tutorial](https://debutanizer.streamlit.app/), we looked at a simple debutanizer column as shown in Figure 1 with 2 manipulated variables (MVs) and 2 controlled variables (CVs). We now add another CV to the system, the pressure drop (dP) across the tower.
        """)
    st.subheader("Gain Matrix")
    st.markdown(r"Consider a $3\times2$ gain matrix, $G$ with 2 MVs and 3 CVs. There will be 6 elements, where each $G_{ij}$ describes the *steady-state* relationship between $\text{CV}_{i}$ and $\text{MV}_{j}$.")   
    st.markdown(r"For this debutanizer column, the gains are:") 

    varvals = {'Reboiler TC': [G11, G21, G31], 'Reflux FC': [G12, G22, G32]}
    st.dataframe(pd.DataFrame(varvals, index=['RVP', 'C5', 'dP']).style.format("{:.3f}"))

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
            \Delta \text{{RVP}} &= {G11:.3f} \cdot \Delta \text{{TC}}_\text{{Reboiler}} + {G12:.3f} \cdot \Delta \text{{FC}}_{{Reflux}}\\ 
            \Delta \text{{C5}} &= {G21:.3f} \cdot \Delta \text{{TC}}_{{Reboiler}} + {G22:.3f} \cdot \Delta \text{{FC}}_{{Reflux}}\\
            \Delta \text{{dP}} &= {G31:.3f} \cdot \Delta \text{{TC}}_{{Reboiler}} + {G32:.3f} \cdot \Delta \text{{FC}}_{{Reflux}}            
        \end{{align}}
        """)

    st.info(f"The gain matrix tells us that: Every unit increase in reboiler temperature reduces RVP by {G11:.2f} units and increases C5 by {G21:.2f} units. This makes sense, if we fire the reboiler harder, we boil stuff up the top so the bottom RVP decreases (less volatile, more heavier components), and some of the heavier components go up the column, so overhead C5 increases. Every unit increase in reflux reduces RVP by {G12:.2f} units and reduces C5 by {G22:.2f} units. This also makes sense, because increasing reflux improves separation and washes down the heavier materials from the top. Separation is improved up to a certain point, considering flooding limits etc. **This time, our controller is also constrained by the dP limits.**")

def dir_text(soln):
    return "<span style='color:blue'>🠉</span> Up" if soln > 0 else "<span style='color:red'>🠋</span> Down"

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
    return f'background-color: {color}'

def highlight_lo(x):
    df2 = pd.DataFrame('', index=x.index, columns=x.columns)
    mask = x['Constraint'] == 'Lo Limit'
    df2.loc[mask, 'OpLo'] = 'background-color: lightblue'
    return df2

def highlight_hi(x):
    df2 = pd.DataFrame('', index=x.index, columns=x.columns)
    mask = x['Constraint'] == 'Hi Limit'
    df2.loc[mask, 'OpHi'] = 'background-color: lightblue'
    return df2  

# df.style.apply(highlight_lo, axis=None, subset=['Constraint', 'OpLo'])
# df.style.apply(highlight_lo, axis=None, subset=['Constraint', 'OpHi'])

df = pd.DataFrame.from_dict(constrained, orient='index', columns=["Constraint"])

df.loc['MV1', 'OpLo'] = MV1LoSS
df.loc['MV2', 'OpLo'] = MV2LoSS
df.loc['CV1', 'OpLo'] = CV1LoSS
df.loc['CV2', 'OpLo'] = CV2LoSS
df.loc['CV3', 'OpLo'] = CV3LoSS
df.loc['MV1', 'Value'] = MV1SSVal
df.loc['MV2', 'Value'] = MV2SSVal
df.loc['CV1', 'Value'] = CV1SSVal
df.loc['CV2', 'Value'] = CV2SSVal
df.loc['CV3', 'Value'] = CV3SSVal
df.loc['MV1', 'Target'] = soln[0] + MV1SSVal
df.loc['MV2', 'Target'] = soln[1] + MV2SSVal
df.loc['CV1', 'Target'] = G11*soln[0]+G12*soln[1] + CV1SSVal
df.loc['CV2', 'Target'] = G21*soln[0]+G22*soln[1] + CV2SSVal
df.loc['CV3', 'Target'] = G31*soln[0]+G32*soln[1] + CV3SSVal
df.loc['MV1', 'OpHi'] = MV1HiSS
df.loc['MV2', 'OpHi'] = MV2HiSS
df.loc['CV1', 'OpHi'] = CV1HiSS
df.loc['CV2', 'OpHi'] = CV2HiSS
df.loc['CV3', 'OpHi'] = CV3HiSS
df.loc['MV1', 'Direction'] = dir_text(soln[0])
df.loc['MV2', 'Direction'] = dir_text(soln[1])
df.loc['CV1', 'Direction'] = dir_text(G11*soln[0]+G12*soln[1])
df.loc['CV2', 'Direction'] = dir_text(G21*soln[0]+G22*soln[1])
df.loc['CV3', 'Direction'] = dir_text(G31*soln[0]+G32*soln[1])
df.loc['MV1', 'Delta'] = soln[0]
df.loc['MV2', 'Delta'] = soln[1]
df.loc['CV1', 'Delta'] = G11*soln[0]+G12*soln[1]
df.loc['CV2', 'Delta'] = G21*soln[0]+G22*soln[1]
df.loc['CV3', 'Delta'] = G31*soln[0]+G32*soln[1]

df.rename(index={'MV1': 'Reboiler TC (°F)', 'MV2': 'Reflux FC (MBPD)', 'CV1': 'RVP (psi)', 'CV2': 'C5 (%)', 'CV3': 'dP (psig)'}, inplace=True)
# df.style.format("{:.3f}")
df = df.style.apply(highlight_lo, axis=None, subset=['Constraint', 'OpLo'])\
             .apply(highlight_hi, axis=None, subset=['Constraint', 'OpHi'])\
             .applymap(color_constraint, subset=['Constraint'])\
             .format(dict.fromkeys(df.select_dtypes('float').columns, "{:.2f}"))
df = df.to_html()

def clamp_dp():
    st.session_state["CV3Limits"] = (3.6, 3.9)

def unclamp_dp():
    st.session_state["CV3Limits"] = (3.6, 4.4)    

with tab4:
    st.header("Interactive LP Simulation")
    st.markdown("In this module, you can adjust the operating limits, gain matrix and LP costs to study the response of the LP solution. We will explore the effects of **'clamping'** the limits of the dP and how it impacts the LP solution and optimization directions. Initially, the dP limits are wide open, and does not impact the optimum point or the feasible region.")
    st.info("**Case Study**: What happens when you reduce the dP upper limit? Reducing the upper limit simulates a possible operational scenario where the tower is flooded or overloaded.")
    cols=st.columns([0.5, 0.5])
    with cols[1]:
        fig = plotLP()
        st.pyplot(fig)
    with cols[0]:
        st.subheader('LP Solution')
        st.write(df, unsafe_allow_html=True)
        st.text("\n")

        st.markdown("#### Optimization Directions")
        st.markdown(f"""
            - **Reboiler TC:** {dir_text(soln[0])} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Reflux FC:** {dir_text(soln[1])}
            - Value of Objective Function (Profit): ${-(V):.2f}
        """, unsafe_allow_html=True)

        st.divider()

        st.header("Optimization Case Studies")
        st.subheader("Case Study 1: Clamp dP Upper Limit")
        st.markdown("What happens to the optimization direction when the upper dP limit is clamped? Use the buttons below or the control panel on the left sidebar to explore this scenario.")
        cols=st.columns([0.5, 0.5])
        with cols[0]:
            st.button('Clamp dP Limit', on_click=clamp_dp, use_container_width=True)
        with cols[1]:
            st.button('Reset dP Limit', on_click=unclamp_dp, use_container_width=True)
        st.markdown("We can see that, when the dP is clamped:")
        st.markdown("""
            - The C5 and RVP are no longer controlling to the upper limit.
            - The controller reduces reflux to offload the column. 
            """)