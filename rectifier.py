import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.ticker import StrMethodFormatter
from pulp import LpVariable, LpStatus, LpProblem, LpMinimize, PULP_CBC_CMD
from matplotlib.legend_handler import HandlerTuple
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.transforms import blended_transform_factory

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

st.set_page_config(layout="wide")

# Variable definitions
VARIABLES = {
    'MV1': {'Name': 'STM', 'UOM': 'klb/h', 'Cost': -40, 'SSVal': 10.5, 
            'Limits': (9.4, 12.5), 'EngLimits': (5.0, 15.0), 'TYPMOV': 0.01},
    'MV2': {'Name': 'REFLUX', 'UOM': 'BPD', 'Cost': 0.1, 'SSVal': 2500, 
            'Limits': (1500.0, 3000.0), 'EngLimits': (1000.0, 3800.0), 'TYPMOV': 0.1},
    'CV1': {'Name': 'RVP', 'UOM': 'psi', 'SSVal': 9.5,
            'Limits': (9.0, 11.0), 'EngLimits': (8.0, 13.0), 'TYPMOV': 0.01},
    'CV2': {'Name': 'PCT', 'UOM': '°F', 'SSVal': 165.0,
            'Limits': (131.0, 175.0), 'EngLimits': (115.0, 200.0), 'TYPMOV': 0.1},
    'CV3': {'Name': 'Valve', 'UOM': "%", 'SSVal': 60.0, 
            'Limits': (-5, 105.0), 'EngLimits': (-5.0, 105.0), 'TYPMOV': 0.1},
    'FF1': {'Name': 'DHT WILD NAP', 'UOM': 'BPD', 'SSVal': 1000},
    'FF2': {'Name': 'Crude OVHD', 'UOM': 'BPD', 'SSVal': 4000},
    'FF3': {'Name': 'SPLT OVHD', 'UOM': 'BPD', 'SSVal': 5500}
}

PLOT_LIMITS = {'x': 2.2, 'y': 1600}

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if "init" not in st.session_state:
        st.session_state["init"] = True
        
        # Initialize limits and costs
        for mv in ['MV1', 'MV2']:
            st.session_state[f"{mv}Limits"] = VARIABLES[mv]['Limits']
            st.session_state[f"{mv}Cost"] = VARIABLES[mv]['Cost']
            st.session_state[f'{mv}SSVal'] = VARIABLES[mv]['SSVal']
        
        for cv in ['CV1', 'CV2', 'CV3']:
            st.session_state[f"{cv}Limits"] = VARIABLES[cv]['Limits']
            st.session_state[f'{cv}SSVal'] = VARIABLES[cv]['SSVal']
        
        for ff in ['FF1', 'FF2', 'FF3']:
            st.session_state[f'{ff}SSVal'] = VARIABLES[ff]['SSVal']
        
        # Initialize visualization toggles
        for var in ['MV1', 'MV2', 'CV1', 'CV2', 'CV3']:
            st.session_state[f"shade_{var}"] = False
        
        st.session_state["shade_feasible"] = False
        st.session_state["show_optimum"] = False
        st.session_state["show_vectors"] = False
        st.session_state["show_isoprofit"] = False
        st.session_state["show_MV"] = True
        st.session_state["show_CV"] = True

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

def render_sidebar():
    """Render the sidebar with all controls"""
    with st.sidebar:
        st.title('LP-DMC Simulation')
        st.text("Model & Tuning Parameters")
        
        render_gain_matrix_inputs()
        render_lp_costs()
        render_feedforward_gains()
        render_mv_controls()
        render_cv_controls()
        render_ff_disturbances()

def render_gain_matrix_inputs():
    """Render gain matrix input controls"""
    with st.expander('Gain Matrix (MV-CV)'):
        gains = {
            'G11': (-1.2547, 0.1, "%.4f"),
            'G12': (0.0016, 0.001, "%.4f"),
            'G21': (9.6359, 0.5, "%.4f"),
            'G22': (-0.0262, 0.01, "%.4f"),
            'G31': (20.6982, 0.5, "%.4f"),
            'G32': (-0.0325, 0.01, "%.4f")
        }
        
        for key, (value, step, fmt) in gains.items():
            mv_idx = '1' if key[1] == '1' else '2'
            cv_idx = key[2]
            st.number_input(
                f"$G_{{{key[1:]}}}$: {VARIABLES[f'MV{mv_idx}']['Name']} vs. {VARIABLES[f'CV{cv_idx}']['Name']}", 
                key=key, value=value, step=step, format=fmt
            )

def render_lp_costs():
    """Render LP cost inputs"""
    with st.expander('LP Costs'):
        st.number_input(f"{VARIABLES['MV1']['Name']} Cost", step=0.5, key="MV1Cost")
        st.number_input(f"{VARIABLES['MV2']['Name']} Cost", step=0.02, key="MV2Cost")

def render_feedforward_gains():
    """Render feedforward gain inputs"""
    with st.expander('Feedforward Gains'):
        ff_gains = {
            'G13': (0.00120, 0.001, 1), 'G14': (-0.0047, 0.001, 2), 'G15': (0.0, 0.1, 3),
            'G23': (0.00120, 0.001, 1), 'G24': (-0.0047, 0.001, 2), 'G25': (0.0, 0.1, 3),
            'G33': (0.00120, 0.001, 1), 'G34': (-0.0047, 0.001, 2), 'G35': (0.0, 0.1, 3)
        }
        
        for key, (value, step, ff_num) in ff_gains.items():
            cv_idx = key[1]
            st.number_input(
                f"$G_{{{key[1:]}}}$: {VARIABLES[f'FF{ff_num}']['Name']} vs. {VARIABLES[f'CV{cv_idx}']['Name']}", 
                key=key, value=value, step=step, format="%.4f"
            )

def render_mv_controls():
    """Render MV limit and current value controls"""
    st.subheader('MV Limits')
    for mv in ['MV1', 'MV2']:
        var = VARIABLES[mv]
        st.slider(
            f"{var['Name']} Limits ({var['UOM']})", 
            var['EngLimits'][0], var['EngLimits'][1], 
            step=var['TYPMOV'], key=f"{mv}Limits"
        )
        st.number_input('Current Value', step=var['TYPMOV'], key=f"{mv}SSVal")
        st.divider()

def render_cv_controls():
    """Render CV limit and current value controls"""
    st.subheader('CV Limits')
    for cv in ['CV1', 'CV2', 'CV3']:
        var = VARIABLES[cv]
        st.slider(
            f"{var['Name']} ({var['UOM']})", 
            var['EngLimits'][0], var['EngLimits'][1], 
            step=var['TYPMOV'], key=f"{cv}Limits"
        )
        st.number_input('Current Value', step=var['TYPMOV'], key=f"{cv}SSVal")
        if cv != 'CV3':
            st.divider()

def render_ff_disturbances():
    """Render feedforward disturbance controls"""
    st.divider()
    st.subheader('FF Disturbances')
    
    ff_labels = [
        'DHT WILD NAP',
        'Crude OVHD',
        'Split OVHD'
    ]
    
    for i, (ff, label) in enumerate(zip(['FF1', 'FF2', 'FF3'], ff_labels), 1):
        var = VARIABLES[ff]
        st.number_input(label, step=100, key=f"{ff}SSVal")
        st.caption(f"{var['Name']} Baseline: {var['SSVal']}{var['UOM']}")

# ============================================================================
# LP SOLVER
# ============================================================================

def get_gain_values():
    """Extract gain values from session state"""
    gains = {}
    for i in range(1, 4):
        for j in range(1, 6):
            key = f'G{i}{j}'
            if key in st.session_state:
                gains[key] = st.session_state[key]
    return gains

def calculate_disturbances(gains):
    """Calculate disturbance terms from feedforward changes"""
    disturbances = {}
    for cv_idx in range(1, 4):
        d = 0
        for ff_idx in range(3, 6):
            ff_num = ff_idx - 2
            ff_change = st.session_state[f'FF{ff_num}SSVal'] - VARIABLES[f'FF{ff_num}']['SSVal']
            d += gains[f'G{cv_idx}{ff_idx}'] * ff_change
        disturbances[f'D{cv_idx}'] = d
    return disturbances

def calculate_limit_deltas(disturbances):
    """Calculate delta limits based on current values and disturbances"""
    deltas = {}
    
    for mv in ['MV1', 'MV2']:
        lo, hi = st.session_state[f"{mv}Limits"]
        pv = st.session_state[f'{mv}SSVal']
        deltas[f'{mv}Hi'] = hi - pv
        deltas[f'{mv}Lo'] = lo - pv
    
    for cv in ['CV1', 'CV2', 'CV3']:
        lo, hi = st.session_state[f"{cv}Limits"]
        pv = st.session_state[f'{cv}SSVal']
        d = disturbances[f'D{cv[2]}']
        deltas[f'{cv}Hi'] = hi - (pv + d)
        deltas[f'{cv}Lo'] = lo - (pv + d)
    
    return deltas

def solve_lp(gains, deltas):
    """Solve the linear programming problem"""
    prob = LpProblem("DMC", LpMinimize)
    
    MV1 = LpVariable("MV1", deltas['MV1Lo'])
    MV2 = LpVariable("MV2", deltas['MV2Lo'])
    
    # Objective function
    prob += st.session_state["MV1Cost"] * MV1 + st.session_state["MV2Cost"] * MV2
    
    # Add constraints
    for cv_idx in range(1, 4):
        cv = f'CV{cv_idx}'
        prob += gains[f'G{cv_idx}1']*MV1 + gains[f'G{cv_idx}2']*MV2 <= deltas[f'{cv}Hi'], f"{cv} High Limit"
        prob += gains[f'G{cv_idx}1']*MV1 + gains[f'G{cv_idx}2']*MV2 >= deltas[f'{cv}Lo'], f"{cv} Low Limit"
    
    for mv in ['MV1', 'MV2']:
        prob += eval(mv) >= deltas[f'{mv}Lo'], f"{mv} Low Limit"
        prob += eval(mv) <= deltas[f'{mv}Hi'], f"{mv} High Limit"
    
    status = prob.solve(PULP_CBC_CMD(msg=0))
    
    if LpStatus[status] == "Optimal":
        solution = [v.varValue for v in prob.variables()]
        objective_value = prob.objective.value()
    else:
        solution = [0, 0]
        objective_value = 0
    
    # Extract constraint information
    constraints_info = {}
    for name, c in prob.constraints.items():
        constraints_info[name] = {'shadow price': c.pi, 'slack': abs(c.slack)}
    
    return status, solution, objective_value, constraints_info

# ============================================================================
# DATA PROCESSING
# ============================================================================

def isConstrained(constraints_info, var):
    loslack = constraints_info[f'{var}_Low_Limit']['slack']
    hislack = constraints_info[f'{var}_High_Limit']['slack']
    
    if np.isclose(abs(loslack), 0, rtol=1e-20):
        return "Lo Limit"
    elif np.isclose(abs(hislack), 0, rtol=1e-20):
        return "Hi Limit"
    else:
        return "Normal"

def calculate_constraint_status(constraints_info):
    """Determine which variables are at their limits"""
    constrained = {}
    for var in ['MV1', 'MV2', 'CV1', 'CV2', 'CV3']:
        constrained[var] = isConstrained(constraints_info, var)
    
    return constrained

def dir_text(value, tol=1e-3):
    """Generate directional text with color coding"""
    if value > tol:
        return "<span style='color:blue'>⬆</span> Up"
    elif value < -tol:
        return "<span style='color:red'>⬇</span> Down"
    else:
        return "<span style='color:black'>-</span>"

def create_solution_dataframe(solution, gains, constrained):
    """Create the main solution dataframe"""
    df = pd.DataFrame.from_dict(constrained, orient='index', columns=["Status"])
    
    # Add limits
    for var_type in ['MV', 'CV']:
        for idx in [1, 2] if var_type == 'MV' else [1, 2, 3]:
            var = f'{var_type}{idx}'
            lo, hi = st.session_state[f"{var}Limits"]
            df.loc[var, 'LoLim'] = lo
            df.loc[var, 'HiLim'] = hi            
            df.loc[var, 'PV'] = st.session_state[f'{var}SSVal']

    # Calculate SS targets and deltas
    for i, mv in enumerate(['MV1', 'MV2'], 1):
        df.loc[mv, 'SSTarget'] = solution[i-1] + st.session_state[f'{mv}SSVal']
        df.loc[mv, 'Delta'] = solution[i-1]
        df.loc[mv, 'Move'] = dir_text(solution[i-1])
    
    for cv_idx in range(1, 4):
        cv = f'CV{cv_idx}'
        delta_cv = gains[f'G{cv_idx}1']*solution[0] + gains[f'G{cv_idx}2']*solution[1]
        df.loc[cv, 'SSTarget'] = delta_cv + st.session_state[f'{cv}SSVal']
        df.loc[cv, 'Delta'] = delta_cv
        df.loc[cv, 'Move'] = dir_text(delta_cv)
    
    df = df.loc[:, ['Status', 'Move', 'LoLim', 'PV', 'SSTarget', 'HiLim', 'Delta']]
    # Rename indices
    rename_dict = {
        f'{vt}{i}': f"{vt} - {VARIABLES[f'{vt}{i}']['Name']} ({VARIABLES[f'{vt}{i}']['UOM']})"
        for vt in ['MV', 'CV'] for i in ([1, 2] if vt == 'MV' else [1, 2, 3])
    }
    df.rename(index=rename_dict, inplace=True)
    
    return df

def create_feedforward_dataframe():
    """Create feedforward variables dataframe"""
    ff_data = {}
    for i in range(1, 4):
        ff = f'FF{i}'
        ff_change = st.session_state[f'{ff}SSVal'] - VARIABLES[ff]['SSVal']
        ff_data[ff] = {
            "PV": st.session_state[f'{ff}SSVal'],
            "Move": dir_text(ff_change),
            "Delta": ff_change
        }
    
    df_ff = pd.DataFrame.from_dict(ff_data, orient="index")
    
    rename_dict = {
        f'FF{i}': f"FF - {VARIABLES[f'FF{i}']['Name']} ({VARIABLES[f'FF{i}']['UOM']})"
        for i in range(1, 4)
    }
    df_ff.rename(index=rename_dict, inplace=True)
    
    return df_ff

def style_dataframes(df, df_ff):
    """Apply styling to dataframes"""
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
    
    def color_constraint(val):
        color = 'lightblue' if 'Limit' in val else ''
        return f"background-color: {color}"
    
    df_styled = df.style.apply(highlight_lo, axis=None, subset=['Status', 'LoLim'])\
                        .apply(highlight_hi, axis=None, subset=['Status', 'HiLim'])\
                        .map(color_constraint, subset=['Status'])\
                        .format(dict.fromkeys(df.select_dtypes('float').columns, "{:.2f}"))\
                        .to_html()
    
    df_ff_styled = df_ff.style.format(dict.fromkeys(df_ff.select_dtypes('float').columns, "{:.2f}"))\
                               .to_html()
    
    return df_styled, df_ff_styled

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def dollar_formatter(x, pos):
    """Format values as dollar amounts"""
    if x < 0:
        return f"\u2212${abs(x):,.0f}"
    else:
        return f"${x:,.0f}"

def calculate_constraint_lines(gains, deltas):
    """Calculate constraint line coordinates"""
    xspace = np.linspace(-PLOT_LIMITS['x'], PLOT_LIMITS['x'], 1000)
    
    lines = {}
    for cv_idx in range(1, 4):
        g1, g2 = gains[f'G{cv_idx}1'], gains[f'G{cv_idx}2']
        lines[f'y_c{2*cv_idx-1}'] = (deltas[f'CV{cv_idx}Hi'] - g1*xspace) / g2
        lines[f'y_c{2*cv_idx}'] = (deltas[f'CV{cv_idx}Lo'] - g1*xspace) / g2
    
    return xspace, lines

def create_constraint_masks(gains, deltas):
    """Create boolean masks for constraint regions"""
    xspace = np.linspace(-PLOT_LIMITS['x'], PLOT_LIMITS['x'], 1000)
    yspace = np.linspace(-PLOT_LIMITS['y'], PLOT_LIMITS['y'], 1000)
    x, y = np.meshgrid(xspace, yspace)
    
    masks = {}
    
    # CV constraints
    for cv_idx in range(1, 4):
        g1, g2 = gains[f'G{cv_idx}1'], gains[f'G{cv_idx}2']
        masks[f'c{2*cv_idx-1}'] = g1*x + g2*y <= deltas[f'CV{cv_idx}Hi']
        masks[f'c{2*cv_idx}'] = g1*x + g2*y >= deltas[f'CV{cv_idx}Lo']
    
    # MV constraints
    masks['m1'] = (x >= deltas['MV1Lo']) & (x <= deltas['MV1Hi'])
    masks['m2'] = (y >= deltas['MV2Lo']) & (y <= deltas['MV2Hi'])
    
    # Feasible region
    feasible = masks['c1'] & masks['c2'] & masks['c3'] & masks['c4'] & masks['c5'] & masks['c6'] & masks['m1'] & masks['m2']
    
    return x, y, masks, feasible

def plot_lp(gains, deltas, solution, objective_value, status, constraints_info):
    """Create the main LP visualization plot"""
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot axes
    ax.axvline(x=0, color='black', lw=0.2, linestyle='-')
    ax.axhline(y=0, color='black', lw=0.2, linestyle='-')
    ax.plot(0, 0, 'kx')

    x, y, masks, feasible = create_constraint_masks(gains, deltas)
    xspace, lines = calculate_constraint_lines(gains, deltas)
    
    # Plot MV limits
    m1l = ax.axvline(x=deltas['MV1Lo'], color='olive', lw=1, linestyle='--', visible=st.session_state["show_MV"])
    ax.axvline(x=deltas['MV1Hi'], color='olive', lw=1, linestyle='-', visible=st.session_state["show_MV"])
    ax.axhline(y=deltas['MV2Lo'], color='olive', lw=1, linestyle='--', visible=st.session_state["show_MV"])
    ax.axhline(y=deltas['MV2Hi'], color='olive', lw=1, linestyle='-', visible=st.session_state["show_MV"])

    # Shade regions if requested
    if st.session_state["shade_MV1"]:
        ax.imshow(masks['m1'], extent=(x.min(),x.max(),y.min(),y.max()), 
                 aspect='auto', origin="lower", cmap=mcolors.ListedColormap(['none', 'yellow']), alpha=0.10)
    if st.session_state["shade_MV2"]:
        ax.imshow(masks['m2'], extent=(x.min(),x.max(),y.min(),y.max()), 
                 aspect='auto', origin="lower", cmap=mcolors.ListedColormap(['none', 'yellow']), alpha=0.10)
    
    cv_colors = ['Reds', 'Blues', 'Greens']
    for cv_idx in range(1, 4):
        if st.session_state[f"shade_CV{cv_idx}"]:
            cv_mask = masks[f'c{2*cv_idx-1}'] & masks[f'c{2*cv_idx}']
            ax.imshow(cv_mask.astype(int), extent=(x.min(),x.max(),y.min(),y.max()), 
                     aspect='auto', origin="lower", cmap=cv_colors[cv_idx-1], alpha=0.15)
    
    # Plot constraint lines
    line_handles = []
    line_labels = []
    colors = ['r', 'b', 'g']
    for cv_idx in range(1, 4):
        hi, = ax.plot(xspace, lines[f'y_c{2*cv_idx-1}'], f'-{colors[cv_idx-1]}', visible=st.session_state["show_CV"])
        lo, = ax.plot(xspace, lines[f'y_c{2*cv_idx}'], f'--{colors[cv_idx-1]}', visible=st.session_state["show_CV"])
        # line_handles.append((hi, lo))
        line_handles.extend([hi, lo])

        constraint = isConstrained(constraints_info, f"CV{cv_idx}")
        constrainthi_label = f"*{VARIABLES[f'CV{cv_idx}']['Name']} HI" if constraint == "Hi Limit" else f"{VARIABLES[f'CV{cv_idx}']['Name']} HI"
        constraintlo_label = f"*{VARIABLES[f'CV{cv_idx}']['Name']} LO" if constraint == "Lo Limit" else f"{VARIABLES[f'CV{cv_idx}']['Name']} LO"
        
        line_labels.extend([
            constrainthi_label,
            constraintlo_label
        ])

    # Plot optimum and isoprofit line
    if LpStatus[status] == "Optimal":
        if st.session_state["show_optimum"]:
            ax.axvline(x=solution[0], color='k', lw=0.75, linestyle='--')
            ax.axhline(y=solution[1], color='k', lw=0.75, linestyle='--')
            ax.plot(solution[0], solution[1], 'Dk', ms=6, markeredgecolor='w', markeredgewidth=0.5)

            # Vertical bracket on right OUTSIDE plot limits for ΔMV2
            xlim = PLOT_LIMITS['x']
            x_bracket_right = xlim * 1.05  # Position outside the plot area
            bracket_width = xlim * 0.03
            ax.annotate("", (x_bracket_right, solution[1]), (x_bracket_right, 0),
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.0, shrinkA=0, shrinkB=0),
                        annotation_clip=False)
            ax.plot([x_bracket_right - bracket_width, x_bracket_right + bracket_width], [0, 0], 'k-', lw=1, clip_on=False)
            ax.text(x_bracket_right + bracket_width*1.5, solution[1]/2, 
                   f'$\Delta${VARIABLES['MV2']['Name']} = {solution[1]:.1f} {VARIABLES['MV2']['UOM']}', ha='left', va='center', fontsize=9, rotation=90, clip_on=False, color='k')

            # Horizontal bracket
            ylim = PLOT_LIMITS['y']
            y_bracket_top = ylim * 1.05      # slightly above the plot
            bracket_height = ylim * 0.03

            # horizontal arrow
            ax.annotate("", (solution[0], y_bracket_top), (0, y_bracket_top),
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.0, shrinkA=0, shrinkB=0),
                        annotation_clip=False)

            # # vertical caps at ends
            ax.plot([0, 0], [y_bracket_top - bracket_height, y_bracket_top + bracket_height], 'k-', lw=1, clip_on=False)

            # label centered above arrow
            ax.text(solution[0]/2, y_bracket_top + bracket_height*1.5,
                    f'$\Delta${VARIABLES['MV1']['Name']} = {solution[0]:.2f} {VARIABLES['MV1']['UOM']}', ha='center', va='bottom', fontsize=9, clip_on=False, color='k')

        
        if st.session_state["show_isoprofit"]:
            xspace_line = np.linspace(-PLOT_LIMITS['x'], PLOT_LIMITS['x'], 10)
            mv2_cost = st.session_state["MV2Cost"]
            mv1_cost = st.session_state["MV1Cost"]
            y_obj = (1/mv2_cost)*objective_value - (mv1_cost/mv2_cost)*xspace_line
            ax.plot(xspace_line, y_obj, ':k', lw=2)
            ax.quiver(solution[0], solution[1], -mv1_cost, -mv2_cost*((PLOT_LIMITS['y']/PLOT_LIMITS['x'])),
                     headwidth=6, width=0.005, alpha=0.95, scale_units='xy', scale=90)
        
        # Plot vector field
        if st.session_state["show_vectors"]:
            plot_vector_field(ax, solution, objective_value, fig)
    
    # Plot feasible region
    ax.imshow(feasible.astype(int), extent=(x.min(),x.max(),y.min(),y.max()), 
             aspect='auto', origin="lower", cmap="binary", alpha=0.10)
    if st.session_state["shade_feasible"]:
        ax.contourf(x, y, feasible.astype(int), levels=[0.5, 1], colors=['none'], hatches=["///"], alpha=0)
        
    # Set plot properties
    ax.set_xlim((-PLOT_LIMITS['x'], PLOT_LIMITS['x']))
    ax.set_ylim((-PLOT_LIMITS['y'], PLOT_LIMITS['y']))
    ax.set_xlabel(f"MV1 Move: $\Delta${VARIABLES['MV1']['Name']} ({VARIABLES['MV1']['UOM']})")
    ax.set_ylabel(f"MV2 Move: $\Delta${VARIABLES['MV2']['Name']} ({VARIABLES['MV2']['UOM']})")
    ax.set_aspect('auto')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    
    # Legend
    # Add MV limit handle
    mv_constraints = []
    for mvlabel in ['MV1', 'MV2']:
        mv_constraints.append(isConstrained(constraints_info, mvlabel))

    if any(c != 'Normal' for c in mv_constraints):
        line_labels.append('*MV Limits')
    else:
        line_labels.append('MV Limits')

    line_handles.append(m1l)
    leg = ax.legend(line_handles, line_labels, loc='lower center', bbox_to_anchor=(0.5, 1.08), ncol=4)

    for txt in leg.get_texts():
        s = txt.get_text()
        if '*' in txt.get_text():
            txt.set_bbox(dict(facecolor='#ff99ff', edgecolor='none', pad=2))

    return fig

def plot_vector_field(ax, solution, objective_value, fig):
    """Plot the vector field showing optimization direction"""
    dvecx = np.linspace(-PLOT_LIMITS['x'], PLOT_LIMITS['x'], 50)
    dvecy = np.linspace(-PLOT_LIMITS['y'], PLOT_LIMITS['y'], 50)
    xv, yv = np.meshgrid(dvecx, dvecy)
    
    mv1_cost = st.session_state["MV1Cost"]
    mv2_cost = st.session_state["MV2Cost"]
    
    mask_threshold = 2
    mask = (mv1_cost * xv) + (mv2_cost * yv) - mask_threshold <= objective_value
    x_masked = xv[~mask]
    y_masked = yv[~mask]
    z_obj = -((mv1_cost * x_masked) + (mv2_cost * y_masked))
    
    vmin = min(-300, np.min(z_obj))
    vmax = max(10, np.max(z_obj))
    
    if vmin < 0 and vmax > 0:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    
    q = ax.quiver(x_masked, y_masked, -mv1_cost, -mv2_cost*((PLOT_LIMITS['y']/PLOT_LIMITS['x'])), 
                  z_obj, norm=norm, cmap='RdYlGn',
                  headwidth=4, width=0.0025, alpha=0.45, scale_units='xy', scale=250)
    
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.13, pos.width, 0.03])
    cbar = fig.colorbar(q, cax=cbar_ax, orientation='horizontal', 
                       label=f"Profit for this move: {dollar_formatter(-objective_value, 2)}")
    
    if vmin < 0 and vmax > 0:
        cbar.set_ticks([vmin, vmin*3/4, vmin/2, vmin/4, 0, vmax/4, vmax/2, vmax*3/4, vmax])
    cbar_ax.xaxis.set_major_formatter(dollar_formatter)
    cbar_ax.tick_params(axis="x", labelsize=9)

# ============================================================================
# TAB CONTENT RENDERERS
# ============================================================================

def render_gain_matrix_tab():
    """Render the Gain Matrix explanation tab"""
    gains = get_gain_values()
    
    cols = st.columns([0.55, 0.45])
    with cols[0]:
        st.subheader("Gain Matrix")
        st.markdown(r"The $3\times2$ gain matrix, $G$ with 2 MVs and 3 CVs, has 6 elements, where each element $G_{ij}$ describes the *steady-state* relationship between $\text{CV}_{i}$ and $\text{MV}_{j}$.")
        
        varvals = {
            f"{VARIABLES['MV1']['Name']}": [gains['G11'], gains['G21'], gains['G31']],
            f"{VARIABLES['MV2']['Name']}": [gains['G12'], gains['G22'], gains['G32']],
            f"FF1": [gains['G13'], gains['G23'], gains['G33']],
            f"FF2": [gains['G14'], gains['G24'], gains['G34']],
            f"FF3": [gains['G15'], gains['G25'], gains['G35']],
        }
        
        def highlight_blue(val):
            return "color: darkblue"
        
        gainmatrix_df = pd.DataFrame(varvals, index=[
            VARIABLES['CV1']['Name'], 
            VARIABLES['CV2']['Name'], 
            VARIABLES['CV3']['Name']
        ])
        st.dataframe(gainmatrix_df.style.applymap(highlight_blue, subset=gainmatrix_df.columns[-3:]).format("{:.5f}"))
        st.info(r"An additional 3 Feedforward (FF) variables are disturbances that cannot be controlled by DMC for an overall gain matrix size of $3\times5$ including the FFs.")
        
        st.markdown("The equation relating the change in CVs, $\Delta CV$ to changes in independent variables MVs and FFs, through the gain matrix is given by:")
        st.latex(rf"\Delta CV = {{G_{{MV}} \cdot \Delta MV}} + \color{{darkblue}}{{G_{{FF}} \cdot \Delta FF}}")
        st.latex(rf"G_{{MV}} = \begin{{bmatrix}} {gains['G11']:.3f} & {gains['G12']:.3f}\\ {gains['G21']:.3f} & {gains['G22']:.3f}\\ {gains['G31']:.3f} & {gains['G32']:.3f}\end{{bmatrix}}")
        st.latex(rf"\color{{darkblue}}{{G_{{FF}} = \begin{{bmatrix}} {gains['G13']:.3f} & {gains['G14']:.3f} & {gains['G15']:.3f}\\ {gains['G23']:.3f} & {gains['G24']:.3f} & {gains['G25']:.3f}\\ {gains['G33']:.3f} & {gains['G34']:.3f} & {gains['G35']:.3f}\end{{bmatrix}}}}")
    
    cols = st.columns([0.75, 0.25])
    with cols[0]:
        st.markdown("Using the gain matrix, the CV relationships can be written in terms of its independent variables:")
        st.latex(rf"""
            \begin{{align}}
                \Delta \text{{{VARIABLES['CV1']['Name']}}} &= {gains['G11']:.3f} \cdot \Delta \text{{{VARIABLES['MV1']['Name']}}} + {gains['G12']:.3f} \cdot \Delta \text{{{VARIABLES['MV2']['Name']}}} + \color{{darkblue}}{{{gains['G13']:.3f} \cdot \Delta \text{{FF1}} + {gains['G14']:.3f} \cdot \Delta \text{{FF2}} + {gains['G15']:.3f} \cdot \Delta \text{{FF3}}}}\\ 
                \Delta \text{{{VARIABLES['CV2']['Name']}}} &= {gains['G21']:.3f} \cdot \Delta \text{{{VARIABLES['MV1']['Name']}}} + {gains['G22']:.3f} \cdot \Delta \text{{{VARIABLES['MV2']['Name']}}} + \color{{darkblue}}{{{gains['G23']:.3f} \cdot \Delta \text{{FF1}} + {gains['G24']:.3f} \cdot \Delta \text{{FF2}} + {gains['G25']:.3f} \cdot \Delta \text{{FF3}}}}\\
                \Delta \text{{{VARIABLES['CV3']['Name']}}} &= {gains['G31']:.3f} \cdot \Delta \text{{{VARIABLES['MV1']['Name']}}} + {gains['G32']:.3f} \cdot \Delta \text{{{VARIABLES['MV2']['Name']}}} + \color{{darkblue}}{{{gains['G33']:.3f} \cdot \Delta \text{{FF1}} + {gains['G34']:.3f} \cdot \Delta \text{{FF2}} + {gains['G35']:.3f} \cdot \Delta \text{{FF3}}}}
            \end{{align}}
            """)

def render_linear_program_tab():
    """Render the Linear Program explanation tab"""
    cols = st.columns([0.55, 0.45])
    with cols[0]:
        st.header('Linear Program')
        st.markdown('The LP steady-state (SS) optimizer is responsible for generating SS targets for the move calculations.')
        st.markdown("We can impose upper and lower limits on the MVs. These are *hard constraints* that cannot be violated.")
        st.latex(r"\text{MV}_{1, \text{LL}} \leq \text{MV}_{1} \leq \text{MV}_{1, \text{UL}}\\\text{MV}_{2, \text{LL}} \leq \text{MV}_{2} \leq \text{MV}_{2, \text{UL}}\\")
        st.info("MV limits are **hard constraints** which will not be violated.")
        st.markdown("We can also impose upper and lower limits on the CVs. These are *soft constraints* that can be relaxed if the LP problem is infeasible.")
        st.latex(r"\text{CV}_{1, \text{LL}} \leq \text{CV}_{1} \leq \text{CV}_{1, \text{UL}}\\\text{CV}_{2, \text{LL}} \leq \text{CV}_{2} \leq \text{CV}_{2, \text{UL}}\\\text{CV}_{3, \text{LL}} \leq \text{CV}_{3} \leq \text{CV}_{3, \text{UL}}")
        st.info("CV limits are **soft constraints** that can be violated if the LP problem is infeasible. A DMC tuning parameter called the **CV Rank** is used to determine the priority of CVs, with 1 being the most important and 999 being the least important.")
        st.warning("This ranked CV give-up routine is not programmed in this simple LP simulator. The app will just show that the problem is infeasible.")
        st.markdown("We solve the LP in terms of $\Delta$ Delta Moves, based on **current conditions** and relative distance from constraints. To transform the solution space in terms of relative moves $\Delta MV$ and $\Delta CV$, we just subtract the current PV from the inequality:")
        st.latex(r"\text{MV}_{1, \text{LL}} - \text{MV}_{1, \text{PV}} \leq \text{MV}_{1} - \text{MV}_{1, \text{PV}} \leq \text{MV}_{1, \text{UL}} - \text{MV}_{1, \text{PV}}\\\text{MV}_{2, \text{LL}} - \text{MV}_{2, \text{PV}} \leq \text{MV}_{2} - \text{MV}_{2, \text{PV}} \leq \text{MV}_{2, \text{UL}} - \text{MV}_{2, \text{PV}}\\")
        st.markdown("The steady-state value of CVs are affected by changes in the FFs, which cannot be controlled by DMC. We subtract a disturbance term to account for the changes in FF values and its predicted impact on the CVs based on the gain matrix.")
        st.info("For example, the impact of FF changes on CV1 is given by a disturbance term $D_1 = G_{13} \cdot \Delta FF_1 + G_{14} \cdot \Delta FF_2 + G_{15} \cdot \Delta FF_3$. The overall change on CV1 is given by adding the predicted steady-state changes from MV movements, $\Delta CV$ to the predicted steady-state changes from FF movements, $D$. For example, $\\text{CV}_1 = \\text{CV}_{1, \\text{PV}} + \Delta \\text{CV}_1 + D_1$")
        st.latex(r"\text{CV}_{1, \text{LL}} - \text{CV}_{1, \text{PV}} - D_1 \leq \text{CV}_{1} - \text{CV}_{1, \text{PV}} - D_1 \leq \text{CV}_{1, \text{UL}} - \text{CV}_{1, \text{PV}} - D_1 \\\text{CV}_{2, \text{LL}} - \text{CV}_{2, \text{PV}} - D_2 \leq \text{CV}_{2} - \text{CV}_{2, \text{PV}} - D_2 \leq \text{CV}_{2, \text{UL}} - \text{CV}_{2, \text{PV}} - D_2 \\\text{CV}_{3, \text{LL}} - \text{CV}_{3, \text{PV}} - D_3 \leq \text{CV}_{3} - \text{CV}_{3, \text{PV}} - D_3 \leq \text{CV}_{3, \text{UL}} - \text{CV}_{3, \text{PV}} - D_3")
        st.markdown("Which gives us the transformed inequalities in terms of how much movement the LP has available based on current PV and its relative distance to the limits:")
        st.latex(r"\Delta\text{MV}_{1, \text{LL}} \leq \Delta \text{MV}_{1} \leq \Delta \text{MV}_{1, \text{UL}}\\\Delta\text{MV}_{2, \text{LL}} \leq \Delta \text{MV}_{2} \leq \Delta \text{MV}_{2, \text{UL}}\\")
        st.latex(r"\Delta\text{CV}_{1, \text{LL}} \leq \Delta \text{CV}_{1} \leq \Delta \text{CV}_{1, \text{UL}}\\\Delta\text{CV}_{2, \text{LL}} \leq \Delta \text{CV}_{2} \leq \Delta \text{CV}_{2, \text{UL}}\\\Delta\text{CV}_{3, \text{LL}} \leq \Delta \text{CV}_{3} \leq \Delta \text{CV}_{3, \text{UL}}")
        st.markdown("Since the CVs are related to the MVs by the gain matrix, we can substitute the equations to get CV limits in terms of MV movements:")
        st.latex(r"G_{11} \Delta \text{MV}_{1} + G_{12} \Delta \text{MV}_{2} \leq \Delta \text{CV}_{1, \text{UL}}\\G_{11} \Delta \text{MV}_{1} + G_{12} \Delta \text{MV}_{2} \geq \Delta \text{CV}_{1, \text{LL}}\\G_{21} \Delta \text{MV}_{1} + G_{22} \Delta \text{MV}_{2} \leq \Delta \text{CV}_{2, \text{UL}} \\G_{21} \Delta \text{MV}_{1} + G_{22} \Delta \text{MV}_{2} \geq \Delta \text{CV}_{2, \text{LL}} \\")
    # with cols[1]:
    #     st.pyplot(fig1)

def render_lp_costs_tab(gains, deltas, solution, objective_value, status, fig):
    """Render the LP Costs explanation tab"""
    cols = st.columns([0.55, 0.45])
    with cols[0]:
        st.header("Feasible Region")
        st.markdown("We can plot the MV and CV limits as a function of MV movements. The limits are linear, so each limit forms a straight line. Since the limits are inequalities, the limit is actually a half-plane, where all points on one side satisfy the inequality. If we take the intersection of all the half-planes, we get a shaded area as shown in the figure on the right. The shaded area is known as the `feasible region` or `solution space` in the LP problem. It is defined based on the current process conditions and the distance of each variable from its constraints.")
        
        st.subheader("LP Objective Function")
        st.markdown("The feasible region tells us that the LP optimizer is allowed to move $\Delta MV_1$ and $\Delta MV_2$ within the shaded region to honour the CV limits. The question now is, out of all the possible points in the feasible region, which one should the optimizer pick and why?")
        st.markdown("From Linear Programming theory, we know that, solutions only exist at the 'corners' of the solution space. These solutions are defined by the objective function. The objective function in DMC is defined as a cost minimization function based on MV movements. The equations below are a simplified version of the actual objective function (see *Sorensen, R. C., & Cutler, C. R. (1998)* for details).")
        st.markdown("We want to assign an 'LP cost' to each MV, based on the economics and desired directionality of MV movement. The LP costs set the **preferred** direction of optimization. For 2 MVs, we get:")
        st.latex(r"\min_{\Delta MV_1, \Delta MV_2} f(\Delta MV_1, \Delta MV_2) = c_1 \Delta MV_1 + c_2 \Delta MV_2")
        
        st.subheader("LP Costs")
        st.markdown("As a rule of thumb, a negative LP cost would incentivize the DMC LP to _maximize_ that variable (**preferred, but not guaranteed!**), and likewise, a positive cost would incentivize the DMC LP to _minimize_ that variable. However, there are exceptions as we will see later on.")
        st.markdown("The DMC controller is designed with a set of LP costs that will drive the process to the desired constraint set under normal operation *('ideal constraints')*.")
        st.info("Is there only one unique set of LP costs that will drive the process to the desired targets? Is setting the costs as simple as just getting the signs right?")
    
    with cols[1]:
        st.pyplot(fig)
    
    cols = st.columns([0.55, 0.45])
    with cols[0]:
        st.header("Shadow Prices")
        st.markdown(r"In Linear Programming theory, the shadow price of a constraint is defined as the change in objective function for each engineering unit of moving a limit.")
        st.latex(r"\text{Shadow Price} = \Delta\text{Obj}/\Delta\text{Limit}")
        st.markdown("By definition, the shadow price of a **non-binding** constraint, which is a variable not at its limit, is equal to 0.")
        
        st.header("Case Study")
        st.warning("Reset the simulation to get the default limits and adjust the LP costs. How does that impact the LP solution? Did the constraints change? What are the new binding constraints (i.e. variables at their limits), and what is the shadow price of this new variable? How sensitive is the solution?")
        
        st.markdown(f"""
            - Value of Objective Function (Profit): ${-(objective_value):.2f}
            - Coodinates of Optimum Point: ({solution[0]:.3f}, {solution[1]:.3f})
        """, unsafe_allow_html=True)

def render_simulation_tab(gains, deltas, solution, objective_value, status, constraints_info, df_styled, df_ff_styled, fig):
    """Render the main simulation tab"""
    cols = st.columns([0.55, 0.45])
    
    with cols[1]:
        st.pyplot(fig)
        
        st.html("<hr><b>LP Visualization Details</b><hr>")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.checkbox("Shade Feasible", key='shade_feasible')
        with col2:
            st.checkbox("Profit Direction", key='show_vectors')
        with col3:
            st.checkbox("Iso-Profit Line", key='show_isoprofit')
        with col4:
            st.checkbox("Optimum Point", key='show_optimum')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.checkbox(f"Show MVs", key='show_MV')
        with col2:
            st.checkbox(f"MV1: {VARIABLES['MV1']['Name']}", key='shade_MV1')
        with col3:
            st.checkbox(f"MV2: {VARIABLES['MV2']['Name']}", key='shade_MV2')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.checkbox(f"Show CVs", key='show_CV')        
        with col2:
            st.checkbox(f"CV1: {VARIABLES['CV1']['Name']}", key='shade_CV1')
        with col3:
            st.checkbox(f"CV2: {VARIABLES['CV2']['Name']}", key='shade_CV2')
        with col4:
            st.checkbox(f"CV3: {VARIABLES['CV3']['Name']}", key='shade_CV3')
        
        if st.session_state["shade_MV1"] or st.session_state["shade_MV2"]:
            st.info("MV limits are *hard constraints* that cannot be violated.")
        
        if st.session_state["shade_CV1"] or st.session_state["shade_CV2"] or st.session_state["shade_CV3"]:
            st.info("CV limits are *soft constraints* that can be **given up** if the LP problem is infeasible.")
    
    with cols[0]:
        if LpStatus[status] != "Optimal":
            st.error(f"⚠ Status: {LpStatus[status]}. No feasible LP solution found.")
        else:
            st.badge("LP Solution Table")
            st.write(df_styled, unsafe_allow_html=True)
            # st.badge("Feedforward Variables")
            # st.write(df_ff_styled, unsafe_allow_html=True)
        
        # Display constraint equations if shaded
        for var_type, indices in [('MV', [1, 2]), ('CV', [1, 2, 3])]:
            for idx in indices:
                var = f'{var_type}{idx}'
                if st.session_state[f"shade_{var}"]:
                    lo, hi = st.session_state[f"{var}Limits"]
                    pv = st.session_state[f'{var}SSVal']
                    if var_type == 'CV':
                        d_term = f" + D_{idx}" if idx <= 3 else ""
                    else:
                        d_term = ""
                    st.latex(rf"\text{{{VARIABLES[var]['Name']} Constraint: }} {lo} \leq {{{pv:.2f}}} + \Delta\text{{{VARIABLES[var]['Name']}}} \leq {hi}")
        
        # Display explanations based on toggles
        if st.session_state["show_isoprofit"]:
            st.markdown("""
                #### Isoprofit Line
                The objective function is linear, so its slope will always be constant along any given direction in the xy plane. 
                Each value of the objective function will give a different straight line. These lines are all parallel to each other. 
                They are isoprofit lines, meaning the value of the objective function is the same anywhere on this line. 
                The LP will drive the system in the direction of maximum cost reduction, which is orthogonal to the isoprofit lines (see the vector field arrows).
            """)
        
        if st.session_state["show_vectors"]:
            st.markdown("""
                #### Direction of increasing profits
                The vector field (arrows) show the direction of increasing profits which are better solutions.
                The LP costs set the preferred direction of optimization. As a rule of thumb, a negative LP cost would incentivize 
                the DMC LP to maximize that variable *(preferred, but not guaranteed!)*, and likewise, a positive cost would 
                incentivize the DMC LP to minimize that variable.
            """)
        
        if st.session_state["show_optimum"]:
            st.markdown(f"""
                #### LP Optimum Point
                Out of all possible points in the feasible region, which one should the optimizer pick?
                DMC solves an **optimization problem** called a **Linear Program (LP)** to decide. The objective function $f$ is to minimize 
                the overall cost of MV movements $\Delta MV$ where the cost for each unit of movement of an MV is $c$.
                
                $\\min f= c_1 \\Delta MV_1 + c_2 \\Delta MV_2$
                
                Optimum ${VARIABLES['MV1']['Name']}^{{OPT}} = {st.session_state['MV1SSVal'] + solution[0]:.2f}$ {VARIABLES['MV1']['UOM']}
                
                Optimum ${VARIABLES['MV2']['Name']}^{{OPT}} = {st.session_state['MV2SSVal'] + solution[1]:.2f}$ {VARIABLES['MV2']['UOM']}
            """)
        
        if st.session_state["shade_feasible"]:
            st.info("""
                #### Feasible Region
                Each MV/CV high limit and low limit can be plotted as a shaded region between 2 straight lines. 
                If we take the intersection of all shaded regions formed by each constraint, we get the feasible region or solution space. 
                The feasible region tells us that DMC is allowed to make MV moves on $\Delta MV_1$ and $\Delta MV_2$ within that space 
                without violating any limits.
            """)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    initialize_session_state()
    render_sidebar()
    
    # Solve LP problem
    gains = get_gain_values()
    disturbances = calculate_disturbances(gains)
    deltas = calculate_limit_deltas(disturbances)
    status, solution, objective_value, constraints_info = solve_lp(gains, deltas)
    
    # Prepare data
    constrained = calculate_constraint_status(constraints_info)
    df = create_solution_dataframe(solution, gains, constrained)
    df_ff = create_feedforward_dataframe()
    df_styled, df_ff_styled = style_dataframes(df, df_ff)
    
    # Create plot
    fig = plot_lp(gains, deltas, solution, objective_value, status, constraints_info)
    # fig1 = plot_lp(gains, deltas, solution, objective_value, status, showMV=False, showCV=False, showFeasible=False)

    # Render tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Simulation", "🕵 Gain Matrix", "🕵 Linear Program", "🕵 LP Costs"])
    
    with tab1:
        render_simulation_tab(gains, deltas, solution, objective_value, status, constraints_info, 
                            df_styled, df_ff_styled, fig)
    
    with tab2:
        render_gain_matrix_tab()
    
    with tab3:
        render_linear_program_tab()
    
    with tab4:
        render_lp_costs_tab(gains, deltas, solution, objective_value, status, fig)
        st.text("\n")
        st.markdown("Shadow Prices")
        st.dataframe(pd.DataFrame(constraints_info).T)

if __name__ == "__main__":
    main()