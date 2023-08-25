import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from pulp import LpVariable, LpProblem, LpMinimize, value, PULP_CBC_CMD
st.set_page_config(layout="wide")

CVStep = 0.05
with st.sidebar:
	st.header('Steady-State Gains')
	cols=st.columns(2)
	with cols[0]:
		G11 = st.number_input('$G_{11}$', value=0.1)
	with cols[1]:
		G12 = st.number_input('$G_{12}$', value=0.2)
	cols=st.columns(2)
	with cols[0]:
		G21 = st.number_input('$G_{21}$', value=0.1)
	with cols[1]:
		G22 = st.number_input('$G_{22}$', value=-0.1)
	st.subheader('Controlled Variables')
	with st.columns(1)[0]:
		CV1Lo, CV1Hi = st.slider(r'CV1 Limits', -8.0, 8.0, (-2.5, 2.5), step=CVStep)
	with st.columns(1)[0]:
		CV2Lo, CV2Hi = st.slider(r'CV2 Limits', -5.0, 5.0, (-2.5, 2.5), step=CVStep)

	st.subheader('Manipulated Variables')

	with st.columns(1)[0]:
		MV1Lo, MV1Hi = st.slider(r'MV1 Limits', -35.0, 35.0, (-20.0, 25.0))

	with st.columns(1)[0]:
		MV2Lo, MV2Hi = st.slider(r'MV2 Limits', -35.0, 35.0, (-20.0, 20.0))		
		
	st.subheader('LP Costs')
	cols=st.columns(2)
	with cols[0]:
		MV1Cost = st.number_input(r'MV1 Cost', value=1.0, step=0.1)
	with cols[1]:
		MV2Cost = st.number_input(r'MV2 Cost', value=0.2, step=0.1)	

st.title('DMC Steady-State Optimizer')
st.header('Linear Program')
st.markdown('The LP steady-state (SS) optimizer is responsible for generating SS targets for the move calculations. We will look at this component first, and the Prediction and Move Plan modules in subsequent notebooks.')
st.header('Gain Matrix')
st.markdown("The steady-state relationship between an MV and a CV is captured by its \
	        **steady-state process gain**. In multivariable controllers, we have a **gain matrix** \
	        that captures the relationships between multiple MVs and CVs.")
st.markdown("Consider a $2\\times2$ gain matrix, $G$ with 2 MVs and 2 CVs. There will be 4 elements, where each $G_{ij}$ describes the relationship between the $i$-th CV \
			and $j$-th MV.")


st.latex(rf"G = \begin{{bmatrix}} \
        G_{{11}} & G_{{12}} \\ \
        G_{{21}} & G_{{22}} \
    \end{{bmatrix}} = \begin{{bmatrix}} \
        {G11:.2f} & {G12:.2f} \\ \
        {G21:.2f} & {G22:.2f} \
    \end{{bmatrix}}")
		
st.markdown("The equation relating the CVs and MVs through the gain matrix is given by:")
st.latex(r"\Delta CV = G \cdot \Delta MV")

st.markdown("Multiplying the terms in the matrix, we get:")

st.latex(r"\Delta \text{CV}_{1} = G_{11} \Delta \text{MV}_{1} + G_{12} \Delta \text{MV}_{2}")
st.latex(r"\Delta \text{CV}_{2} = G_{21} \Delta \text{MV}_{1} + G_{22} \Delta \text{MV}_{2}")

limits = 40
d = np.linspace(-limits, limits, 100)
x,y = np.meshgrid(d,d)
stepsize = 0.1
v_new = 0

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

# Each of the variables is printed with it's resolved optimum value
#     for v in prob.variables():
#         print(v.name, "=", v.varValue)

soln = [v.varValue for v in prob.variables()]
V = prob.objective.value()

dvec = np.linspace(-limits + (0.1*limits), limits - (0.1*limits), 12)
xv, yv = np.meshgrid(dvec, dvec)
y_obj = (1/MV2Cost)*V - (MV1Cost/MV2Cost)*d
z_obj = (MV1Cost * xv) + (MV2Cost * yv)

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
ax.plot(soln[0], soln[1], 'ok');

# the obj func
ax.plot(d, y_obj, '-.k');

# plot vector field to show the direction of optimization (direction of decreasing cost)
ax.quiver(xv, yv,-MV1Cost,-MV2Cost, z_obj, cmap='gray', headwidth=4, width=0.003, scale=40, alpha=0.5)

ax.set_xlim((-limits, limits))
ax.set_ylim((-limits, limits))
ax.set_xlabel('MV1')
ax.set_ylabel('MV2')
# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')); # No decimal places
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=5);

def dir_text(soln):
	return "<span style='color:blue'>🠉</span> Increase" if soln > 0 else "<span style='color:red'>🠋</span> Decrease"

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

df = pd.DataFrame.from_dict(constrained, orient='index', columns=["Constraint"])
df['Target'] = 0
df.loc['MV1', 'Target'] = soln[0]
df.loc['MV2', 'Target'] = soln[1]
df.loc['CV1', 'Target'] = G11*soln[0]+G12*soln[1]
df.loc['CV2', 'Target'] = G21*soln[0]+G22*soln[1]
df.loc['MV1', 'Direction'] = dir_text(soln[0])
df.loc['MV2', 'Direction'] = dir_text(soln[1])
df.loc['CV1', 'Direction'] = '-'
df.loc['CV2', 'Direction'] = '-'
df = df.style.applymap(color_constraint, subset=['Constraint'])
df = df.to_html()

st.header("Results")
cols=st.columns([0.5, 0.5])
with cols[0]:
	st.pyplot(fig)
with cols[1]:
	st.subheader('LP Solution')
	st.write(df, unsafe_allow_html=True)
	# st.markdown(f"- {dir_text(soln[0])} MV1 \n - {dir_text(soln[1])} MV2", unsafe_allow_html=True)
	# st.markdown(f"Objective Function: {V:.1f}")



st.dataframe(pd.DataFrame(o).T)

