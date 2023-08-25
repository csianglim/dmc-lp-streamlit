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
			G11 = st.number_input('$G_{11}$: Reboiler TC-RVP', value=-0.200, step=0.01, format="%.3f")
		with cols[1]:
			G12 = st.number_input('$G_{12}$: Reboiler TC-C5', value=-0.072, step=0.01, format="%.3f")
		cols=st.columns(2)
		with cols[0]:
			G21 = st.number_input('$G_{21}$: Reflux FC-RVP', value=0.125, step=0.01, format="%.3f")
		with cols[1]:
			G22 = st.number_input('$G_{22}$: Reflux FC-C5', value=-0.954, step=0.01, format="%.3f")
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

st.title('DMC LP Intro')

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
	st.write("TBD")

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