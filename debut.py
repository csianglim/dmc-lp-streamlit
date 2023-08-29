import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from pulp import LpVariable, LpProblem, LpMinimize, value, PULP_CBC_CMD
st.set_page_config(layout="wide")

CV1SSVal = 9.0
CV2SSVal = 2.5
MV1SSVal = 300
MV2SSVal = 11.0

CVStep = 0.05
with st.sidebar:
    st.header("Model & Controls")
    with st.expander('Steady-State Gains'):
        cols=st.columns(2)
        with cols[0]:
            st.number_input('$G_{11}$: Reboiler TC-RVP', key="G11", value=-0.200, step=0.10, format="%.3f")
        with cols[1]:
            G12 = st.number_input('$G_{12}$: Reflux FC-RVP', value=-0.072, step=0.10, format="%.3f")
        cols=st.columns(2)
        with cols[0]:
            G21 = st.number_input('$G_{21}$: Reboiler TC-C5', value=0.125, step=0.10, format="%.3f")
        with cols[1]:
            G22 = st.number_input('$G_{22}$: Reflux FC-C5', value=-0.954, step=0.10, format="%.3f")
    st.subheader('Controlled Variables')
    with st.columns(1)[0]:
        CV1LoSS, CV1HiSS = st.slider(r'RVP Limits (psi)', 7.5, 10.0, (8.0, 9.5), step=CVStep)
    with st.columns(1)[0]:
        CV2LoSS, CV2HiSS = st.slider(r'C5 Limits (%)', 0.5, 5.0, (1.0, 4.5), step=CVStep)

    st.subheader('Manipulated Variables')

    with st.columns(1)[0]:
        MV1LoSS, MV1HiSS = st.slider(r'Reboiler TC Limits (°F)', 290.0, 310.0, (291.0, 309.0))

    with st.columns(1)[0]:
        MV2LoSS, MV2HiSS = st.slider(r'Reflux FC Limits (MBPD)', 1.0, 21.0, (2.0, 20.0))        
        
    st.subheader('LP Costs')
    cols=st.columns(2)
    with cols[0]:
        MV1Cost = st.number_input(r'Reboiler TC Cost', value=-1.0, step=0.1)
    with cols[1]:
        MV2Cost = st.number_input(r'Reflux FC Cost', value=-1.0, step=0.1)  

# Grab vars
G11 = st.session_state["G11"]

limits = 10
d = np.linspace(-limits, limits, 100)
x,y = np.meshgrid(d,d)
stepsize = 0.1
v_new = 0

CV1Hi = CV1HiSS - CV1SSVal
CV1Lo = CV1LoSS - CV1SSVal
CV2Hi = CV2HiSS - CV2SSVal
CV2Lo = CV2LoSS - CV2SSVal

MV1Hi = MV1HiSS - MV1SSVal
MV1Lo = MV1LoSS - MV1SSVal
MV2Hi = MV2HiSS - MV2SSVal
MV2Lo = MV2LoSS - MV2SSVal

# CV constraints
c1 = G11*x+G12*y <= CV1Hi
c2 = G11*x+G12*y >= CV1Lo
c3 = G21*x+G22*y <= CV2Hi
c4 = G21*x+G22*y >= CV2Lo

# MV constraints
m1 = (x >= MV1Lo) & (x <= MV1Hi)
m2 = (y >= MV2Lo) & (y <= MV2Hi)

# equation of a line, y = mx + c
y_c1 = (CV1Hi - G11*d)/G12
y_c2 = (CV1Lo - G11*d)/G12
y_c3 = (CV2Hi - G21*d)/G22
y_c4 = (CV2Lo - G21*d)/G22

prob = LpProblem("DMC", LpMinimize)

MV1 = LpVariable("MV1", MV1Lo)
MV2 = LpVariable("MV2", MV2Lo)

prob += MV1Cost * MV1 + MV2Cost * MV2

# constraint formulation in terms of MV1 and MV2
prob += G11*MV1+G12*MV2 <= CV1Hi, "CV1 High Limit"
prob += G11*MV1+G12*MV2 >= CV1Lo, "CV1 Low Limit"
prob += G21*MV1+G22*MV2 <= CV2Hi, "CV2 High Limit"
prob += G21*MV1+G22*MV2 >= CV2Lo, "CV2 Low Limit"
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
    ax.imshow((c1 & c2 & c3 & c4 & m1 & m2).astype(int), extent=(x.min(),x.max(),y.min(),y.max()), origin="lower", cmap="binary", alpha=0.1);
    ax.plot(d, y_c1, '-r', label='CV1_Hi');
    ax.plot(d, y_c2, '--r', label='CV1_Lo');
    ax.plot(d, y_c3, '-b', label='CV2_Hi');
    ax.plot(d, y_c4, '--b', label='CV2_Lo');

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

st.title('DMC LP Intro')
st.subheader('An interactive tutorial on interpreting the DMC linear program.')
st.markdown('Last Updated: Siang Lim (August 2023). Source on [GitHub](https://github.com/csianglim/dmc-lp-streamlit).')

with st.expander('Model', expanded=True):
    row1=st.columns([0.55, 0.45])
with st.expander('Linear Program', expanded=True):  
    row2=st.columns([0.55, 0.45])
    row3=st.columns([0.55, 0.45])
with row1[1]:
    st.image('column.png', caption='Figure 1: Simple debutanizer column. Details see: https://arxiv.org/abs/2202.13291')
with row1[0]:
    st.header("Debutanizer Column")
    st.markdown("Consider a simple debutanizer column as shown in Figure 1 that's configured with an APC (Advanced Process Control) system with 2 manipulated variables (MVs) and 2 controlled variables (CVs). The 2 MVs are the reflux flow rate in the overhead, and the reboiler temperature in the bottoms. The 2 CVs are the bottoms RVP and the overhead C5. The control objective for this column is to maintain a maximum RVP target required by product specs in the bottom stream and minimize C5s in the overhead stream.")
    st.subheader("Gain Matrix")
    st.markdown(r"Consider a $2\times2$ gain matrix, $G$ with 2 MVs and 2 CVs. There will be 4 elements, where each $G_{ij}$ describes the *steady-state* relationship between $\text{CV}_{i}$ and $\text{MV}_{j}$.")   
    st.markdown(r"For this debutanizer column, the gains are:") 

    varvals = {'Reboiler TC': [G11, G21], 'Reflux FC': [G12, G22]}
    st.dataframe(pd.DataFrame(varvals, index=['RVP', 'C5']).style.format("{:.3f}"))

    st.markdown("The equation relating the CVs and MVs through the gain matrix is given by:")
    st.latex(r"\Delta CV = G \cdot \Delta MV")  
    st.latex(rf"G = \begin{{bmatrix}} \
            G_{{11}} & G_{{12}} \\ \
            G_{{21}} & G_{{22}} \
        \end{{bmatrix}} = \begin{{bmatrix}} \
            {G11:.3f} & {G12:.3f} \\ \
            {G21:.3f} & {G22:.3f} \
        \end{{bmatrix}}")
    st.markdown("Using the gain matrix, the CV relationships can be written in terms of its MVs:")
    st.latex(rF"\begin{{align}}\Delta \text{{RVP}} &= {G11:.3f} \cdot \Delta \text{{TC}}_\text{{Reboiler}} + {G12:.3f} \cdot \Delta \text{{FC}}_{{Reflux}}\\ \Delta \text{{C5}} &= {G21:.3f} \cdot \Delta \text{{TC}}_{{Reboiler}} + {G22:.3f} \cdot \Delta \text{{FC}}_{{Reflux}} \end{{align}}")

    st.info(f"The gain matrix tells us that: Every unit increase in reboiler temperature reduces RVP by {G11} units and increases C5 by {G21} units. This makes sense, if we fire the reboiler harder, we boil stuff up the top so the bottom RVP decreases (less volatile, more heavier components), and some of the heavier components go up the column, so overhead C5 increases. Every unit increase in reflux reduces RVP by {G12} units and reduces C5 by {G22} units. This also makes sense, because increasing reflux improves separation and washes down the heavier materials from the top. Separation is improved up to a certain point, considering flooding limits etc.")
    st.subheader("Matrix Conditioning")
    st.markdown("The Singular Value Decomposition (SVD) operation allows us to break down the gain matrix $G$ into 3 matrices")
    st.latex(r"G = U \Sigma V^T")
    st.markdown("[Moore (1986)](https://ieeexplore.ieee.org/abstract/document/4789019) provides a really nice physical interpretation of these 3 matrices. The first column vector in the $U$ matrix tells the strongest direction of CV movements possible for this system, and the second column gives the second strongest direction, and so on. The $\Sigma$ singular values tell us the magnitude of these directions. The first row vector in the $V^T$ matrix tells us the MV movements required to achieve the strongest direction of CV movement, and the second row vector is the MV movements needed for the second strongest direction, and so on.")
    st.markdown(f"A geometric interpretation of the SVD is shown below. The left subplot shows a unit circle of MV movements. The right subplot shows the impact of the corresponding MV movements on the CVs. For example, point A is a unit increase in Reflux FC with no change to Reboiler TC, which causes RVP to decrease by {G12} and C5 % to decrease by {G22}.")
    st.markdown("The shape of the CV response ellipse tells us that there are strong and weak control directions. For this particular system, it is much more difficult to control the RVP compared to the C5. To get a 1 unit change in C5, we need about 10 times the magnitude of MV movement compared to making a 1 unit change in RVP.")
    st.info("Use the controls on the left to change the gain matrix and see the impact on the CVs and the shape of the CV response ellipse.")
    # st.pyplot(fig)
    try: # BUG: math error when user clicks too fast ValueError: $\Delta MV_1$
        st.pyplot(fig)
    except:
        try: # BUG: math error when user clicks too fast ValueError: $\Delta MV_1$
            st.pyplot(fig)
        except:
            pass
        st.write("Plot error, please try again.")

    # st.latex(rf"\begin{{bmatrix}} \
    #         {G11:.3f} & {G12:.3f} \\ \
    #         {G21:.3f} & {G22:.3f} \
    #     \end{{bmatrix}} = \
    #     \underbrace{{\
    #         \begin{{bmatrix}} \
    #             {u[0][0]:.3f} & {u[0][1]:.3f} \\ \
    #             {u[1][0]:.3f} & {u[1][1]:.3f} \
    #         \end{{bmatrix}}}}_{{U}} \
    #     \underbrace{{ \
    #         \begin{{bmatrix}} \
    #             {s[0]:.3f} & {0:d} \\ \
    #             {0:d} & {s[1]:.3f} \
    #         \end{{bmatrix}}}}_{{\Sigma}} \
    #     \underbrace{{ \
    #         \begin{{bmatrix}} \
    #             {Vt[0][0]:.3f} & {Vt[0][1]:.3f} \\ \
    #             {Vt[1][0]:.3f} & {Vt[1][1]:.3f} \
    #         \end{{bmatrix}}}}_{{V^T}}")

    st.latex(rf"\begin{{bmatrix}} \
            {G11:.3f} & {G12:.3f} \\ \
            {G21:.3f} & {G22:.3f} \
        \end{{bmatrix}} = \
        \left.\vphantom{{\begin{{matrix}} {u[0][0]:.3f} \\ {u[1][0]:.3f} \end{{matrix}}}}\right[\
        \overbrace{{ \begin{{matrix}} {u[0][0]:.3f} \\ {u[1][0]:.3f} \end{{matrix}}}}^{{u_1}}\
        \overbrace{{ \begin{{matrix}} {u[0][1]:.3f} \\ {u[1][1]:.3f} \end{{matrix}}}}^{{u_2}}\
        \left.\vphantom{{\begin{{matrix}} {u[0][1]:.3f} \\ {u[1][1]:.3f} \end{{matrix}}}}\right]\
        \
        \left.\vphantom{{\begin{{matrix}} {s[0]:.3f} \\ {0:d} \end{{matrix}}}}\right[\
        \overbrace{{ \begin{{matrix}} {s[0]:.3f} \\ {0:d} \end{{matrix}}}}^{{\sigma_1}}\
        \underbrace{{ \begin{{matrix}} {0:d} \\ {s[1]:.3f} \end{{matrix}}}}_{{\sigma_2}}\
        \left.\vphantom{{\begin{{matrix}} {0:d} \\ {s[1]:.3f} \end{{matrix}}}}\right]\
        \
        \begin{{bmatrix}} \
            {Vt[0][0]:.3f} & {Vt[0][1]:.3f} \\ \
            {Vt[1][0]:.3f} & {Vt[1][1]:.3f} \
        \end{{bmatrix}}\!\!\
        \begin{{matrix}} v_1 \\ v_2 \end{{matrix}}")

with row2[1]:
    st.image('ranade.png', caption="Figure 2: The 3 main DMC modules. From Ranade, S. M., & Torres, E. (2009). From dynamic mysterious control to dynamic manageable control. Hydrocarbon Processing, 88(3), 77-81.")
with row2[0]:
    st.header('Linear Program')
    st.markdown('The LP steady-state (SS) optimizer is responsible for generating SS targets for the move calculations. We will look at this component first.')
    st.markdown("We can impose upper and lower limits on the MVs. These are *hard constraints* that cannot be violated.")
    st.latex(r"\text{MV}_{1, \text{Lo}} \leq \text{MV}_{1} \leq \text{MV}_{1, \text{Hi}}\\\text{MV}_{2, \text{Lo}} \leq \text{MV}_{2} \leq \text{MV}_{2, \text{Hi}}\\")
    st.markdown("We can also impose upper and lower limits on the CVs. These are *soft constraints* that can be relaxed if the LP problem is infeasible.")
    st.latex(r"\text{CV}_{1, \text{Lo}} \leq \text{CV}_{1} \leq \text{CV}_{1, \text{Hi}}\\\text{CV}_{2, \text{Lo}} \leq \text{CV}_{2} \leq \text{CV}_{2, \text{Hi}}\\")
    st.info("MV limits are **hard constraints** which will not be violated. CV limits are **soft constraints** that can be violated if the LP problem is infeasible. A DMC tuning parameter called the **CV Rank** is used to determine the priority of CVs, with 1 being the most important and 999 being the least important.")
    st.markdown("Since the CVs are related to the MVs by the gain matrix, we can substitute the equations to get CV limits in terms of MV movements:")
    st.latex(r"G_{11} \Delta \text{MV}_{1} + G_{12} \Delta \text{MV}_{2} \leq \Delta \text{CV}_{1, \text{Hi}}\\G_{11} \Delta \text{MV}_{1} + G_{12} \Delta \text{MV}_{2} \geq \Delta \text{CV}_{1, \text{Lo}}\\G_{21} \Delta \text{MV}_{1} + G_{22} \Delta \text{MV}_{2} \leq \Delta \text{CV}_{2, \text{Hi}} \\G_{21} \Delta \text{MV}_{1} + G_{22} \Delta \text{MV}_{2} \geq \Delta \text{CV}_{2, \text{Lo}} \\")
    st.markdown("For this debutanizer problem, the current measurements captured at one particular steady-state, and the limits for the MVs and CVs are:")

    varvals = {'Reboiler TC': [MV1LoSS, MV1SSVal, MV1HiSS], 
         'Reflux FC': [MV2LoSS, MV2SSVal, MV2HiSS],
         'RVP': [CV1LoSS, CV1SSVal, CV1HiSS],
         'C5': [CV2LoSS, CV2SSVal, CV2HiSS],
         }
    st.dataframe(pd.DataFrame(varvals, index=['Lower Limit', 'Current Value', 'Upper Limit']).T.style.format("{:.1f}"))

with row3[1]:
    st.pyplot(plotLP(showVector=False, showOptimum=False))
with row3[0]:
    st.subheader("Feasible Region")
    st.markdown("We can plot the MV and CV limits as a function of MV movements. The limits are linear, so each limit forms a straight line. Since the limits are inequalities, the limit is actually a half-plane, where all points on one side satisfy the inequality. If we take the intersection of all the half-planes, we get a shaded area as shown in the figure on the right. The shaded area is known as the `feasible region` or `solution space` in the LP problem. It is defined based on the current process conditions and the distance of each variable from its constraints.")

    st.subheader("Objective Function and LP Costs")
    st.markdown("The feasible region tells us that the LP optimizer is allowed to move $\Delta MV_1$ and $\Delta MV_2$ within the shaded region to honour the CV limits. The question now is, out of all the possible points in the feasible region, which one should the optimizer pick and why?")
    st.markdown("The objective function in DMC is defined as a cost minimization function based on MV movements. The equations below are a simplified version of the actual objective function (see *Sorensen, R. C., & Cutler, C. R. (1998)* for details).")
    st.markdown("We want to assign an 'LP cost' to each MV, based on the economics and desired directionality of MV movement. For 2 MVs, we get:")
    st.latex(r"\min_{\Delta MV_1, \Delta MV_2} f(\Delta MV_1, \Delta MV_2) = c_1 \Delta MV_1 + c_2 \Delta MV_2")
    st.markdown("As a rule of thumb, a negative LP cost would incentivize the DMC LP to maximize that variable, and likewise, a positive cost would incentivize the DMC LP to minimize that variable. However, there are exceptions as we will see later on.")
    st.info("For now, let's assume that we have the following LP costs: $c_1 = -1$, $c_2 = -1$.")

def dir_text(soln):
    return "<span style='color:blue'>🠉</span> Up" if soln > 0 else "<span style='color:red'>🠋</span> Down"

o = {}
# st.text(prob.constraints.items())
for name, c in prob.constraints.items():
    o[name] = {'shadow price':c.pi, 'slack': abs(c.slack)}
    # st.write(name, c.slack)
# st.text(o)

constrained = {}
for var in ['MV1', 'MV2', 'CV1', 'CV2']:
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
df.loc['MV1', 'Value'] = MV1SSVal
df.loc['MV2', 'Value'] = MV2SSVal
df.loc['CV1', 'Value'] = CV1SSVal
df.loc['CV2', 'Value'] = CV2SSVal
df.loc['MV1', 'Target'] = soln[0] + MV1SSVal
df.loc['MV2', 'Target'] = soln[1] + MV2SSVal
df.loc['CV1', 'Target'] = G11*soln[0]+G12*soln[1] + CV1SSVal
df.loc['CV2', 'Target'] = G21*soln[0]+G22*soln[1] + CV2SSVal
df.loc['MV1', 'OpHi'] = MV1HiSS
df.loc['MV2', 'OpHi'] = MV2HiSS
df.loc['CV1', 'OpHi'] = CV1HiSS
df.loc['CV2', 'OpHi'] = CV2HiSS
df.loc['MV1', 'Direction'] = dir_text(soln[0])
df.loc['MV2', 'Direction'] = dir_text(soln[1])
df.loc['CV1', 'Direction'] = dir_text(G11*soln[0]+G12*soln[1])
df.loc['CV2', 'Direction'] = dir_text(G21*soln[0]+G22*soln[1])
df.loc['MV1', 'Delta'] = soln[0]
df.loc['MV2', 'Delta'] = soln[1]
df.loc['CV1', 'Delta'] = G11*soln[0]+G12*soln[1]
df.loc['CV2', 'Delta'] = G21*soln[0]+G22*soln[1]

df.rename(index={'MV1': 'Reboiler TC (°F)', 'MV2': 'Reflux FC (MBPD)', 'CV1': 'RVP (psi)', 'CV2': 'C5 (%)'}, inplace=True)
# df.style.format("{:.3f}")
df = df.style.apply(highlight_lo, axis=None, subset=['Constraint', 'OpLo'])\
             .apply(highlight_hi, axis=None, subset=['Constraint', 'OpHi'])\
             .applymap(color_constraint, subset=['Constraint'])\
             .format(dict.fromkeys(df.select_dtypes('float').columns, "{:.2f}"))
df = df.to_html()

st.header("Interactive LP Simulation")
cols=st.columns([0.5, 0.5])
with cols[0]:
    fig = plotLP()
    st.pyplot(fig)
with cols[1]:
    st.subheader('LP Solution')
    st.write(df, unsafe_allow_html=True)
    # st.markdown(f"- {dir_text(soln[0])} MV1 \n - {dir_text(soln[1])} MV2", unsafe_allow_html=True)
    # st.markdown(f"Objective Function: {V:.1f}")