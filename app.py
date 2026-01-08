import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="NumericalPro - Advanced Calculator",
    layout="wide",
    page_icon="🧮",
    initial_sidebar_state="expanded"
)

# --- Advanced Custom Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0; }
    
    .block-container { 
        padding: 2rem 3rem; 
        background: white; 
        border-radius: 20px; 
        margin: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        text-align: center;
    }
    
    .hero-title {
        font-size: 48px;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 18px;
        opacity: 0.95;
        margin-top: 10px;
    }
    
    .problem-card {
        background: white;
        padding: 30px 20px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 3px solid transparent;
        text-align: center;
        position: relative;
        overflow: hidden;
        min-height: 200px;
    }
    
    .problem-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
    }
    
    .selected-card {
        border: 3px solid #667eea;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    
    .selected-card .card-icon,
    .selected-card .card-title,
    .selected-card .card-desc {
        color: white !important;
    }
    
    .card-icon {
        font-size: 48px;
        margin-bottom: 15px;
        display: block;
    }
    
    .card-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #1f2937;
    }
    
    .card-desc {
        font-size: 13px;
        color: #6b7280;
        line-height: 1.5;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 16px 32px;
        font-weight: 700;
        font-size: 16px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f6f8fb 0%, #ffffff 100%);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        text-align: center;
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
    
    .metric-label {
        font-size: 14px;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #667eea;
        margin: 10px 0;
    }
    
    .metric-sub {
        font-size: 12px;
        color: #9ca3af;
    }
    
    .section-header {
        font-size: 28px;
        font-weight: 800;
        color: #1f2937;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
    }
    
    .formula-box {
        background: #f9fafb;
        padding: 15px 20px;
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        font-family: 'Courier New', monospace;
        font-size: 16px;
        color: #374151;
        margin: 15px 0;
        text-align: center;
        font-weight: 600;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 20px 0;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        padding: 12px;
        font-size: 16px;
        font-family: 'Courier New', monospace;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 40px 0;
    }
    
    div[data-testid="stExpander"] {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        margin: 10px 0;
    }
    
    .matrix-input {
        background: #f8fafc;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        margin: 10px 0;
    }
    
    .convergence-graph {
        margin: 20px 0;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .error-message {
        background: linear-gradient(135deg, #fecaca 0%, #fee2e2 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ef4444;
        color: #7f1d1d;
        margin: 10px 0;
    }
    
    .success-message {
        background: linear-gradient(135deg, #bbf7d0 0%, #dcfce7 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #22c55e;
        color: #14532d;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Enhanced Function Parser (Natural Input) ---
def parse_function(expr, angle_mode="Radians"):
    """Parse mathematical expressions with natural syntax"""
    expr = str(expr).strip()
    if not expr:
        raise ValueError("Function expression cannot be empty")
    
    # Replace common mathematical symbols
    expr = expr.replace('^', '**')
    expr = expr.replace('√', 'sqrt')
    expr = expr.replace('sin', 'np.sin')
    expr = expr.replace('cos', 'np.cos')
    expr = expr.replace('tan', 'np.tan')
    expr = expr.replace('exp', 'np.exp')
    expr = expr.replace('ln', 'np.log')
    expr = expr.replace('log', 'np.log10')
    expr = expr.replace('sqrt', 'np.sqrt')
    expr = expr.replace('abs', 'np.abs')
    expr = expr.replace('pi', 'np.pi')
    expr = expr.replace('π', 'np.pi')
    expr = expr.replace(' e ', ' np.e ')
    
    # Handle implicit multiplication
    expr = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'(\d+)\(', r'\1*(', expr)
    expr = re.sub(r'\)(\d)', r')*\1', expr)
    expr = re.sub(r'\)([a-zA-Z])', r')*\1', expr)
    
    # Handle e as exponential constant
    expr = re.sub(r'(?<!\w)e(?!\w)', 'np.e', expr)
    
    def func(x):
        try:
            val = np.deg2rad(x) if angle_mode == "Degrees" else x
            # Create safe evaluation environment
            safe_dict = {
                'x': val,
                'np': np,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'log10': np.log10,
                'sqrt': np.sqrt,
                'abs': np.abs,
                'pi': np.pi,
                'e': np.e
            }
            return eval(expr, {"__builtins__": {}}, safe_dict)
        except Exception as e:
            raise ValueError(f"Error evaluating function at x={x}: {str(e)}")
    
    return func

# --- ROOT FINDING METHODS ---
def bisection_method(f, a, b, tol, max_iter):
    steps = []
    fa, fb = f(a), f(b)
    
    if fa == 0:
        return a, steps, None
    if fb == 0:
        return b, steps, None
    
    if fa * fb >= 0:
        return None, steps, "Error: f(a) and f(b) must have opposite signs!"
    
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        error = abs(b - a) / 2
        
        steps.append({
            "Iteration": i+1, 
            "a": f"{a:.8f}", 
            "b": f"{b:.8f}", 
            "c": f"{c:.8f}", 
            "f(c)": f"{fc:.8f}", 
            "Error": f"{error:.2e}"
        })
        
        if abs(fc) < tol or error < tol:
            return c, steps, None
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    return c, steps, f"Maximum iterations ({max_iter}) reached"

def false_position_method(f, a, b, tol, max_iter):
    steps = []
    fa, fb = f(a), f(b)
    
    if fa == 0:
        return a, steps, None
    if fb == 0:
        return b, steps, None
    
    if fa * fb >= 0:
        return None, steps, "Error: f(a) and f(b) must have opposite signs!"
    
    for i in range(max_iter):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        
        steps.append({
            "Iteration": i+1, 
            "a": f"{a:.8f}", 
            "b": f"{b:.8f}", 
            "c": f"{c:.8f}", 
            "f(c)": f"{fc:.8f}"
        })
        
        if abs(fc) < tol:
            return c, steps, None
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    return c, steps, f"Maximum iterations ({max_iter}) reached"

def newton_raphson(f, x0, tol, max_iter):
    steps = []
    x = x0
    h = 1e-7
    
    for i in range(max_iter):
        fx = f(x)
        
        # Numerical derivative
        dfx = (f(x + h) - fx) / h
        
        # Check for division by zero
        if abs(dfx) < 1e-12:
            return x, steps, "Error: Derivative too small (near zero)"
        
        steps.append({
            "Iteration": i+1, 
            "x_n": f"{x:.8f}", 
            "f(x_n)": f"{fx:.8f}", 
            "f'(x_n)": f"{dfx:.8f}"
        })
        
        if abs(fx) < tol:
            return x, steps, None
        
        x_new = x - fx / dfx
        
        if abs(x_new - x) < tol:
            return x_new, steps, None
        
        # Check for divergence
        if abs(x_new) > 1e10:
            return x_new, steps, "Warning: Method may be diverging"
        
        x = x_new
    
    return x, steps, f"Maximum iterations ({max_iter}) reached"

def secant_method(f, x0, x1, tol, max_iter):
    steps = []
    
    for i in range(max_iter):
        fx0, fx1 = f(x0), f(x1)
        
        steps.append({
            "Iteration": i+1, 
            "x_n-1": f"{x0:.8f}", 
            "x_n": f"{x1:.8f}", 
            "f(x_n)": f"{fx1:.8f}"
        })
        
        if abs(fx1) < tol:
            return x1, steps, None
        
        # Check for division by zero
        if abs(fx1 - fx0) < 1e-12:
            return x1, steps, "Error: Division by zero in secant method"
        
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x_new
        
        if abs(x1 - x0) < tol:
            return x1, steps, None
    
    return x1, steps, f"Maximum iterations ({max_iter}) reached"

def fixed_point_iteration(g, x0, tol, max_iter):
    steps = []
    x = x0
    
    for i in range(max_iter):
        x_new = g(x)
        error = abs(x_new - x)
        
        steps.append({
            "Iteration": i+1, 
            "x_n": f"{x:.8f}", 
            "g(x_n)": f"{x_new:.8f}", 
            "Error": f"{error:.2e}"
        })
        
        if error < tol:
            return x_new, steps, None
        
        # Check for divergence
        if abs(x_new) > 1e10:
            return x_new, steps, "Warning: Method may be diverging"
        
        x = x_new
    
    return x, steps, f"Maximum iterations ({max_iter}) reached"

# --- INTERPOLATION METHODS ---
def lagrange_interpolation(x_points, y_points, x_eval):
    n = len(x_points)
    result = 0
    steps = []
    
    for i in range(n):
        L_i = 1
        for j in range(n):
            if i != j:
                L_i *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])
        
        term = y_points[i] * L_i
        result += term
        
        steps.append({
            "i": i, 
            "x_i": f"{x_points[i]:.4f}", 
            "y_i": f"{y_points[i]:.4f}", 
            "L_i(x)": f"{L_i:.6f}", 
            "Term": f"{term:.6f}"
        })
    
    return result, steps

def divided_difference_table(x_points, y_points):
    n = len(x_points)
    table = np.zeros((n, n))
    table[:, 0] = y_points
    
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x_points[i+j] - x_points[i])
    
    return table

def newton_divided_difference(x_points, y_points, x_eval):
    table = divided_difference_table(x_points, y_points)
    n = len(x_points)
    result = table[0][0]
    product = 1
    steps = []
    
    steps.append({
        "Term": 0, 
        "Coefficient": f"{table[0][0]:.6f}", 
        "Product": "1", 
        "Value": f"{table[0][0]:.6f}"
    })
    
    for i in range(1, n):
        product *= (x_eval - x_points[i-1])
        term_value = table[0][i] * product
        result += term_value
        
        steps.append({
            "Term": i, 
            "Coefficient": f"{table[0][i]:.6f}", 
            "Product": f"{product:.6f}", 
            "Value": f"{term_value:.6f}"
        })
    
    return result, table, steps

# --- LINEAR SYSTEMS ---
def jacobi_method(A, b, x0, tol, max_iter):
    n = len(b)
    x = x0.copy()
    steps = []
    
    # Check diagonal dominance
    for i in range(n):
        if abs(A[i][i]) <= sum(abs(A[i][j]) for j in range(n) if j != i):
            return x0, steps, "Warning: Matrix may not be diagonally dominant"
    
    for k in range(max_iter):
        x_new = np.zeros(n)
        
        for i in range(n):
            sum_val = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_val) / A[i][i]
        
        error = np.linalg.norm(x_new - x, np.inf)
        
        step_dict = {"Iteration": k+1, "Error": f"{error:.2e}"}
        for i in range(n):
            step_dict[f"x{i+1}"] = f"{x_new[i]:.6f}"
        
        steps.append(step_dict)
        
        if error < tol:
            return x_new, steps, None
        
        x = x_new.copy()
    
    return x, steps, f"Maximum iterations ({max_iter}) reached"

def gauss_seidel(A, b, x0, tol, max_iter):
    n = len(b)
    x = x0.copy()
    steps = []
    
    # Check diagonal dominance
    for i in range(n):
        if abs(A[i][i]) <= sum(abs(A[i][j]) for j in range(n) if j != i):
            return x0, steps, "Warning: Matrix may not be diagonally dominant"
    
    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            sum_val = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum_val) / A[i][i]
        
        error = np.linalg.norm(x - x_old, np.inf)
        
        step_dict = {"Iteration": k+1, "Error": f"{error:.2e}"}
        for i in range(n):
            step_dict[f"x{i+1}"] = f"{x[i]:.6f}"
        
        steps.append(step_dict)
        
        if error < tol:
            return x, steps, None
    
    return x, steps, f"Maximum iterations ({max_iter}) reached"

# --- DIFFERENTIATION ---
def numerical_derivative(f, x, h, method="central"):
    if method == "forward":
        return (f(x + h) - f(x)) / h
    elif method == "backward":
        return (f(x) - f(x - h)) / h
    else:  # central
        return (f(x + h) - f(x - h)) / (2 * h)

def calculate_derivative(f, x_values, h, method):
    results = []
    for x in x_values:
        try:
            derivative = numerical_derivative(f, x, h, method)
            results.append({
                "x": f"{x:.6f}",
                "f(x)": f"{f(x):.6f}",
                "f'(x)": f"{derivative:.6f}",
                "Method": method
            })
        except Exception as e:
            results.append({
                "x": f"{x:.6f}",
                "f(x)": "Error",
                "f'(x)": "Error",
                "Method": method
            })
    return results

# --- INTEGRATION ---
def trapezoidal_rule(f, a, b, n):
    if n <= 0:
        raise ValueError("Number of intervals must be positive")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = [f(xi) for xi in x]
    
    result = (h / 2) * (y[0] + 2 * sum(y[1:-1]) + y[-1])
    
    steps = []
    for i in range(len(x)):
        steps.append({
            "i": i, 
            "x_i": f"{x[i]:.6f}", 
            "f(x_i)": f"{y[i]:.6f}"
        })
    
    return result, steps

def simpsons_rule(f, a, b, n):
    if n <= 0:
        raise ValueError("Number of intervals must be positive")
    
    if n % 2 != 0:
        n += 1  # Make n even
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = [f(xi) for xi in x]
    
    # Simpson's rule formula
    result = (h / 3) * (y[0] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-1:2]) + y[-1])
    
    steps = []
    for i in range(len(x)):
        coeff = 1 if (i == 0 or i == n) else (4 if i % 2 == 1 else 2)
        steps.append({
            "i": i, 
            "x_i": f"{x[i]:.6f}", 
            "f(x_i)": f"{y[i]:.6f}", 
            "Coefficient": coeff
        })
    
    return result, steps

# --- SESSION STATE ---
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = "Root Finding"
if 'function_input' not in st.session_state:
    st.session_state.function_input = "x^3 - x - 2"

# --- HERO SECTION ---
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🧮 NumericalPro Calculator</div>
        <div class="hero-subtitle">Professional Numerical Methods Suite - Natural Input Syntax</div>
    </div>
""", unsafe_allow_html=True)

# --- PROBLEM TYPE SELECTION ---
st.markdown('<div class="section-header">📋 Select Problem Type</div>', unsafe_allow_html=True)

problems = [
    {"name": "Root Finding", "icon": "🎯", "desc": "Bisection, Newton, Secant, Compare All"},
    {"name": "Interpolation", "icon": "📈", "desc": "Lagrange, Newton Divided Difference"},
    {"name": "Linear Systems", "icon": "⚡", "desc": "Jacobi, Gauss-Seidel Methods"},
    {"name": "Differentiation", "icon": "📊", "desc": "Forward, Backward, Central"},
    {"name": "Integration", "icon": "∫", "desc": "Trapezoidal, Simpson's Rule"},
]

cols = st.columns(len(problems))
for idx, prob in enumerate(problems):
    with cols[idx]:
        is_selected = "selected-card" if st.session_state.problem_type == prob['name'] else ""
        st.markdown(f"""
            <div class="problem-card {is_selected}">
                <div class="card-icon">{prob['icon']}</div>
                <div class="card-title">{prob['name']}</div>
                <div class="card-desc">{prob['desc']}</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button(f"Select", key=f"btn_{idx}", use_container_width=True):
            st.session_state.problem_type = prob['name']
            st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

problem = st.session_state.problem_type

# ============= ROOT FINDING =============
if problem == "Root Finding":
    st.markdown('<div class="section-header">🎯 Root Finding Methods</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        method = st.selectbox("Select Method", 
            ["Bisection", "False Position", "Newton-Raphson", "Secant", "Fixed Point", "Compare All"])
        
        func_expr = st.text_input("Enter Function f(x)", value=st.session_state.function_input,
            help="Examples: x^3-x-2, sin(x)-x, e^x-3x, 2x^2+3x-5", key="func_input")
        st.session_state.function_input = func_expr
        
        st.markdown(f'<div class="formula-box">f(x) = {func_expr}</div>', unsafe_allow_html=True)
        angle_mode = st.radio("Angle Mode", ["Radians", "Degrees"], horizontal=True)
    
    with col2:
        st.markdown("**Parameters:**")
        a_val = st.number_input("Initial/Lower (a/x₀)", value=1.0, format="%.6f")
        b_val = st.number_input("Second/Upper (b/x₁)", value=2.0, format="%.6f",
            disabled=(method=="Newton-Raphson" or method=="Fixed Point"))
        tol = st.number_input("Tolerance", value=1e-6, format="%.1e")
        max_iter = st.number_input("Max Iterations", value=50, step=1, min_value=1)
    
    st.markdown('<div class="info-box"><strong>💡 Syntax:</strong> x^2 (power), sin(x), cos(x), e^x, ln(x), sqrt(x), pi, 2x (natural multiplication)</div>', unsafe_allow_html=True)
    
    if st.button("🚀 Calculate Root", use_container_width=True):
        try:
            f = parse_function(func_expr, angle_mode)
            
            # Test function at initial points
            test_a = f(a_val)
            if method not in ["Newton-Raphson", "Fixed Point"]:
                test_b = f(b_val)
            
            methods_to_run = ["Bisection", "False Position", "Newton-Raphson", "Secant", "Fixed Point"] if method == "Compare All" else [method]
            results = []
            
            for m in methods_to_run:
                try:
                    if m == "Bisection":
                        root, steps, error = bisection_method(f, a_val, b_val, tol, int(max_iter))
                    elif m == "False Position":
                        root, steps, error = false_position_method(f, a_val, b_val, tol, int(max_iter))
                    elif m == "Newton-Raphson":
                        root, steps, error = newton_raphson(f, a_val, tol, int(max_iter))
                    elif m == "Secant":
                        root, steps, error = secant_method(f, a_val, b_val, tol, int(max_iter))
                    elif m == "Fixed Point":
                        root, steps, error = fixed_point_iteration(f, a_val, tol, int(max_iter))
                    
                    if error:
                        st.markdown(f'<div class="error-message"><strong>{m}:</strong> {error}</div>', unsafe_allow_html=True)
                    elif steps:
                        results.append({"Method": m, "Root": root, "Iterations": len(steps), "Steps": steps})
                except Exception as e:
                    st.markdown(f'<div class="error-message"><strong>{m}:</strong> {str(e)}</div>', unsafe_allow_html=True)
            
            if results:
                st.markdown('<div class="success-message">✅ Calculation Completed Successfully!</div>', unsafe_allow_html=True)
                
                st.markdown("### 📊 Results Summary")
                cols = st.columns(len(results))
                for idx, res in enumerate(results):
                    with cols[idx]:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">{res['Method']}</div>
                                <div class="metric-value">{res['Root']:.6f}</div>
                                <div class="metric-sub">{res['Iterations']} iterations</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("### 📈 Convergence Visualization")
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Function Plot", "Error Convergence"))
                
                # Plot 1: Function
                x_range = np.linspace(min(a_val, b_val) - 2, max(a_val, b_val) + 2, 400)
                y_range = [f(x) for x in x_range]
                
                fig.add_trace(
                    go.Scatter(x=x_range, y=y_range, name="f(x)", line=dict(color='#1f2937', width=3)),
                    row=1, col=1
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
                
                # Plot roots
                for res in results:
                    fig.add_trace(
                        go.Scatter(x=[res['Root']], y=[f(res['Root'])], 
                                 mode='markers', name=f"{res['Method']} Root",
                                 marker=dict(size=12, symbol='diamond')),
                        row=1, col=1
                    )
                
                # Plot 2: Error convergence
                colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a']
                for idx, res in enumerate(results):
                    df = pd.DataFrame(res['Steps'])
                    iterations = list(range(1, len(df) + 1))
                    
                    if 'Error' in df.columns:
                        errors = [float(err) for err in df['Error']]
                        fig.add_trace(
                            go.Scatter(x=iterations, y=errors, name=f"{res['Method']} Error",
                                     line=dict(width=2, color=colors[idx % len(colors)])),
                            row=1, col=2
                        )
                
                fig.update_xaxes(title_text="x", row=1, col=1)
                fig.update_yaxes(title_text="f(x)", row=1, col=1)
                fig.update_xaxes(title_text="Iteration", row=1, col=2)
                fig.update_yaxes(title_text="Error", type="log", row=1, col=2)
                
                fig.update_layout(
                    height=500, 
                    template="plotly_white",
                    showlegend=True,
                    title_text="Root Finding Analysis"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 📋 Detailed Steps")
                for res in results:
                    with st.expander(f"{res['Method']} - Iteration Table ({res['Iterations']} iterations)"):
                        st.dataframe(pd.DataFrame(res['Steps']), use_container_width=True)
                        
                        # Add some metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Root", f"{res['Root']:.8f}")
                        with col2:
                            st.metric("Iterations", res['Iterations'])
                        with col3:
                            st.metric("f(root)", f"{f(res['Root']):.2e}")
        
        except Exception as e:
            st.markdown(f'<div class="error-message">❌ Error: {str(e)}</div>', unsafe_allow_html=True)

# ============= INTERPOLATION =============
elif problem == "Interpolation":
    st.markdown('<div class="section-header">📈 Interpolation Methods</div>', unsafe_allow_html=True)
    
    method = st.selectbox("Select Method", ["Lagrange Interpolation", "Newton Divided Difference"])
    
    col1, col2 = st.columns(2)
    with col1:
        x_input = st.text_input("X values (comma-separated)", value="0, 1, 2, 3")
    with col2:
        y_input = st.text_input("Y values (comma-separated)", value="1, 2, 0, 4")
    
    x_eval = st.number_input("Evaluate at x =", value=1.5)
    
    if st.button("🧮 Interpolate", use_container_width=True):
        try:
            x_points = np.array([float(x.strip()) for x in x_input.split(',')])
            y_points = np.array([float(y.strip()) for y in y_input.split(',')])
            
            if len(x_points) != len(y_points):
                st.error("❌ X and Y must have same number of points!")
            elif len(x_points) < 2:
                st.error("❌ At least 2 points are required!")
            else:
                if method == "Lagrange Interpolation":
                    result, steps = lagrange_interpolation(x_points, y_points, x_eval)
                    st.markdown(f'<div class="success-message">✅ Interpolated Value at x = {x_eval}: {result:.6f}</div>', unsafe_allow_html=True)
                    
                    # Visualization
                    fig = go.Figure()
                    
                    # Original points
                    fig.add_trace(go.Scatter(
                        x=x_points, y=y_points, mode='markers', name='Data Points',
                        marker=dict(size=12, color='red')
                    ))
                    
                    # Generate interpolated curve
                    x_curve = np.linspace(min(x_points), max(x_points), 100)
                    y_curve = []
                    for x_val in x_curve:
                        y_val, _ = lagrange_interpolation(x_points, y_points, x_val)
                        y_curve.append(y_val)
                    
                    fig.add_trace(go.Scatter(
                        x=x_curve, y=y_curve, mode='lines', name='Interpolated Curve',
                        line=dict(color='blue', width=3)
                    ))
                    
                    # Evaluation point
                    fig.add_trace(go.Scatter(
                        x=[x_eval], y=[result], mode='markers', name=f'Evaluated Point (x={x_eval})',
                        marker=dict(size=15, color='green', symbol='star')
                    ))
                    
                    fig.update_layout(
                        title="Lagrange Interpolation",
                        xaxis_title="x",
                        yaxis_title="y",
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📋 Calculation Steps"):
                        st.dataframe(pd.DataFrame(steps), use_container_width=True)
                    
                else:  # Newton Divided Difference
                    result, table, steps = newton_divided_difference(x_points, y_points, x_eval)
                    st.markdown(f'<div class="success-message">✅ Interpolated Value at x = {x_eval}: {result:.6f}</div>', unsafe_allow_html=True)
                    
                    # Visualization
                    fig = go.Figure()
                    
                    # Original points
                    fig.add_trace(go.Scatter(
                        x=x_points, y=y_points, mode='markers', name='Data Points',
                        marker=dict(size=12, color='red')
                    ))
                    
                    # Generate interpolated curve
                    x_curve = np.linspace(min(x_points), max(x_points), 100)
                    y_curve = []
                    for x_val in x_curve:
                        y_val, _, _ = newton_divided_difference(x_points, y_points, x_val)
                        y_curve.append(y_val)
                    
                    fig.add_trace(go.Scatter(
                        x=x_curve, y=y_curve, mode='lines', name='Interpolated Curve',
                        line=dict(color='purple', width=3)
                    ))
                    
                    # Evaluation point
                    fig.add_trace(go.Scatter(
                        x=[x_eval], y=[result], mode='markers', name=f'Evaluated Point (x={x_eval})',
                        marker=dict(size=15, color='green', symbol='star')
                    ))
                    
                    fig.update_layout(
                        title="Newton Divided Difference Interpolation",
                        xaxis_title="x",
                        yaxis_title="y",
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📋 Divided Difference Table"):
                        st.dataframe(pd.DataFrame(table), use_container_width=True)
                    
                    with st.expander("📋 Calculation Steps"):
                        st.dataframe(pd.DataFrame(steps), use_container_width=True)
        
        except Exception as e:
            st.markdown(f'<div class="error-message">❌ Error: {str(e)}</div>', unsafe_allow_html=True)

# ============= LINEAR SYSTEMS =============
elif problem == "Linear Systems":
    st.markdown('<div class="section-header">⚡ Linear Systems Solver</div>', unsafe_allow_html=True)
    
    method = st.selectbox("Select Method", ["Jacobi Method", "Gauss-Seidel Method"])
    
    st.markdown("### Matrix A (Coefficients)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n = st.selectbox("Matrix Size", [2, 3, 4], index=1)
    
    # Create matrix input
    st.markdown("#### Enter Matrix Elements (row by row):")
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # Matrix A input
    cols = st.columns(n)
    for i in range(n):
        st.write(f"**Row {i+1}:**")
        cols = st.columns(n + 1)  # +1 for b vector
        for j in range(n):
            with cols[j]:
                A[i][j] = st.number_input(f"A[{i+1},{j+1}]", value=float(4 if i == j else 1 if abs(i-j)==1 else 0), 
                                         key=f"A_{i}_{j}")
        with cols[n]:
            b[i] = st.number_input(f"b[{i+1}]", value=float(i+1), key=f"b_{i}")
    
    # Initial guess
    st.markdown("#### Initial Guess:")
    x0 = []
    cols = st.columns(n)
    for i in range(n):
        with cols[i]:
            x0.append(st.number_input(f"x0[{i+1}]", value=0.0, key=f"x0_{i}"))
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        tol = st.number_input("Tolerance", value=1e-6, format="%.1e")
    with col2:
        max_iter = st.number_input("Max Iterations", value=100, step=1, min_value=1)
    
    if st.button("⚡ Solve System", use_container_width=True):
        try:
            x0_array = np.array(x0)
            
            if method == "Jacobi Method":
                solution, steps, error = jacobi_method(A, b, x0_array, tol, max_iter)
            else:
                solution, steps, error = gauss_seidel(A, b, x0_array, tol, max_iter)
            
            if error:
                st.markdown(f'<div class="error-message">{error}</div>', unsafe_allow_html=True)
            
            if steps:
                st.markdown('<div class="success-message">✅ Solution Found!</div>', unsafe_allow_html=True)
                
                # Display solution
                st.markdown("### 📊 Solution Vector")
                cols = st.columns(n)
                for i in range(n):
                    with cols[i]:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">x{i+1}</div>
                                <div class="metric-value">{solution[i]:.6f}</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Verification
                st.markdown("### ✅ Verification (Ax = b)")
                verification = A @ solution
                verif_df = pd.DataFrame({
                    "b (target)": b,
                    "Ax (calculated)": verification,
                    "Difference": b - verification
                })
                st.dataframe(verif_df, use_container_width=True)
                
                # Convergence plot
                st.markdown("### 📈 Convergence Plot")
                fig = go.Figure()
                
                if steps:
                    df_steps = pd.DataFrame(steps)
                    iterations = df_steps["Iteration"].tolist()
                    
                    for i in range(n):
                        x_values = [float(df_steps[f"x{i+1}"][j]) for j in range(len(df_steps))]
                        fig.add_trace(go.Scatter(
                            x=iterations, y=x_values, 
                            name=f"x{i+1}", 
                            mode='lines+markers',
                            marker=dict(size=6)
                        ))
                    
                    fig.update_layout(
                        title="Solution Convergence",
                        xaxis_title="Iteration",
                        yaxis_title="Solution Value",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed steps
                with st.expander("📋 Iteration Details"):
                    st.dataframe(pd.DataFrame(steps), use_container_width=True)
        
        except Exception as e:
            st.markdown(f'<div class="error-message">❌ Error: {str(e)}</div>', unsafe_allow_html=True)

# ============= DIFFERENTIATION =============
elif problem == "Differentiation":
    st.markdown('<div class="section-header">📊 Numerical Differentiation</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        func_expr = st.text_input("Enter Function f(x)", value="sin(x)*exp(-x/5)", 
                                 help="Examples: x^2, sin(x), exp(x), ln(x)")
        st.markdown(f'<div class="formula-box">f(x) = {func_expr}</div>', unsafe_allow_html=True)
        angle_mode = st.radio("Angle Mode", ["Radians", "Degrees"], horizontal=True)
    
    with col2:
        method = st.selectbox("Method", ["Forward", "Backward", "Central", "Compare All"])
        h = st.number_input("Step size (h)", value=0.001, format="%.6f")
    
    st.markdown("### Evaluate at points:")
    eval_input = st.text_input("x values (comma-separated)", value="0, 1, 2, 3, 4")
    
    if st.button("📊 Calculate Derivatives", use_container_width=True):
        try:
            f = parse_function(func_expr, angle_mode)
            x_values = [float(x.strip()) for x in eval_input.split(',')]
            
            methods_to_run = ["Forward", "Backward", "Central"] if method == "Compare All" else [method]
            all_results = []
            
            for m in methods_to_run:
                results = calculate_derivative(f, x_values, h, m)
                all_results.extend(results)
            
            # Display results
            results_df = pd.DataFrame(all_results)
            st.markdown('<div class="success-message">✅ Derivatives Calculated Successfully!</div>', unsafe_allow_html=True)
            
            st.markdown("### 📋 Results Table")
            st.dataframe(results_df, use_container_width=True)
            
            # Visualization
            st.markdown("### 📈 Function and Derivatives Visualization")
            
            # Generate smooth curve for better visualization
            x_min, x_max = min(x_values), max(x_values)
            x_smooth = np.linspace(x_min - 1, x_max + 1, 200)
            y_smooth = [f(x) for x in x_smooth]
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Function f(x)", "First Derivative f'(x)", 
                              "Error Analysis", "Comparison of Methods"),
                specs=[[{"secondary_y": True}, {"secondary_y": True}],
                      [{"colspan": 2}, None]]
            )
            
            # Plot 1: Function
            fig.add_trace(
                go.Scatter(x=x_smooth, y=y_smooth, name="f(x)", 
                          line=dict(color='blue', width=3)),
                row=1, col=1
            )
            
            # Plot evaluation points
            eval_points = [f(x) for x in x_values]
            fig.add_trace(
                go.Scatter(x=x_values, y=eval_points, mode='markers', 
                          name="Evaluation Points", marker=dict(size=8, color='red')),
                row=1, col=1
            )
            
            # Plot 2: Derivatives
            colors = {'Forward': 'green', 'Backward': 'orange', 'Central': 'purple'}
            for m in methods_to_run:
                m_results = [r for r in all_results if r['Method'] == m]
                if m_results:
                    x_vals = [float(r['x']) for r in m_results]
                    deriv_vals = [float(r["f'(x)"]) for r in m_results]
                    
                    fig.add_trace(
                        go.Scatter(x=x_vals, y=deriv_vals, mode='lines+markers',
                                  name=f"{m} Difference", line=dict(width=2, color=colors.get(m, 'gray'))),
                        row=1, col=2
                    )
            
            # Plot 3: Error comparison
            if method == "Compare All" and len(methods_to_run) > 1:
                # For error analysis, we need the true derivative
                # Since we don't have it, we'll use central difference as reference
                ref_method = "Central"
                ref_results = {float(r['x']): float(r["f'(x)"]) for r in all_results if r['Method'] == ref_method}
                
                for m in methods_to_run:
                    if m != ref_method:
                        m_results = [r for r in all_results if r['Method'] == m]
                        x_vals = [float(r['x']) for r in m_results]
                        errors = []
                        
                        for r in m_results:
                            x_val = float(r['x'])
                            if x_val in ref_results:
                                errors.append(abs(float(r["f'(x)"]) - ref_results[x_val]))
                        
                        if errors:
                            fig.add_trace(
                                go.Scatter(x=x_vals, y=errors, mode='lines+markers',
                                          name=f"{m} Error (vs {ref_method})", 
                                          line=dict(width=2, dash='dot')),
                                row=2, col=1
                            )
            
            fig.update_xaxes(title_text="x", row=1, col=1)
            fig.update_yaxes(title_text="f(x)", row=1, col=1)
            fig.update_xaxes(title_text="x", row=1, col=2)
            fig.update_yaxes(title_text="f'(x)", row=1, col=2)
            fig.update_xaxes(title_text="x", row=2, col=1)
            fig.update_yaxes(title_text="Absolute Error", row=2, col=1)
            
            fig.update_layout(
                height=700,
                template="plotly_white",
                showlegend=True,
                title_text="Numerical Differentiation Analysis"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional metrics
            st.markdown("### 📊 Statistical Summary")
            if method == "Compare All":
                summary_data = []
                for m in methods_to_run:
                    m_results = [r for r in all_results if r['Method'] == m and r["f'(x)"] != "Error"]
                    if m_results:
                        derivs = [float(r["f'(x)"]) for r in m_results]
                        summary_data.append({
                            "Method": m,
                            "Mean": np.mean(derivs),
                            "Std Dev": np.std(derivs),
                            "Min": np.min(derivs),
                            "Max": np.max(derivs)
                        })
                
                if summary_data:
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        except Exception as e:
            st.markdown(f'<div class="error-message">❌ Error: {str(e)}</div>', unsafe_allow_html=True)

# ============= INTEGRATION =============
elif problem == "Integration":
    st.markdown('<div class="section-header">∫ Numerical Integration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        func_expr = st.text_input("Enter Function f(x)", value="sin(x)*exp(-x/5)", 
                                 help="Examples: x^2, sin(x), exp(x), ln(x)")
        st.markdown(f'<div class="formula-box">f(x) = {func_expr}</div>', unsafe_allow_html=True)
        angle_mode = st.radio("Angle Mode", ["Radians", "Degrees"], horizontal=True)
    
    with col2:
        method = st.selectbox("Method", ["Trapezoidal Rule", "Simpson's Rule", "Compare Both"])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        a = st.number_input("Lower limit (a)", value=0.0, format="%.6f")
    with col2:
        b = st.number_input("Upper limit (b)", value=2*np.pi, format="%.6f")
    with col3:
        n = st.number_input("Number of intervals", value=10, step=2, min_value=2)
    
    if st.button("∫ Calculate Integral", use_container_width=True):
        try:
            f = parse_function(func_expr, angle_mode)
            
            methods_to_run = ["Trapezoidal", "Simpson"] if method == "Compare Both" else [method.split("'")[0]]
            results = []
            
            for m in methods_to_run:
                try:
                    if m == "Trapezoidal":
                        integral, steps = trapezoidal_rule(f, a, b, int(n))
                    else:  # Simpson
                        integral, steps = simpsons_rule(f, a, b, int(n))
                    
                    results.append({
                        "Method": m,
                        "Integral": integral,
                        "Steps": steps,
                        "n": n
                    })
                except Exception as e:
                    st.markdown(f'<div class="error-message"><strong>{m}:</strong> {str(e)}</div>', unsafe_allow_html=True)
            
            if results:
                st.markdown('<div class="success-message">✅ Integration Completed Successfully!</div>', unsafe_allow_html=True)
                
                # Display results
                st.markdown("### 📊 Results")
                cols = st.columns(len(results))
                for idx, res in enumerate(results):
                    with cols[idx]:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">{res['Method']} Rule</div>
                                <div class="metric-value">{res['Integral']:.6f}</div>
                                <div class="metric-sub">{res['n']} intervals</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Visualization
                st.markdown("### 📈 Integration Visualization")
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Function Plot", "Integration Area", 
                                  "Error Analysis", "Convergence Study"),
                    specs=[[{}, {}], [{"colspan": 2}, None]]
                )
                
                # Plot 1: Function plot
                x_curve = np.linspace(a, b, 1000)
                y_curve = [f(x) for x in x_curve]
                
                fig.add_trace(
                    go.Scatter(x=x_curve, y=y_curve, name="f(x)", 
                              line=dict(color='blue', width=3)),
                    row=1, col=1
                )
                
                # Plot 2: Integration area
                for res in results:
                    x_points = np.linspace(a, b, int(res['n']) + 1)
                    y_points = [f(x) for x in x_points]
                    
                    # Add filled area
                    if res['Method'] == "Trapezoidal":
                        for i in range(len(x_points) - 1):
                            fig.add_trace(
                                go.Scatter(
                                    x=[x_points[i], x_points[i], x_points[i+1], x_points[i+1], x_points[i]],
                                    y=[0, y_points[i], y_points[i+1], 0, 0],
                                    fill="toself",
                                    fillcolor='rgba(255, 0, 0, 0.2)',
                                    line=dict(color='red', width=1),
                                    showlegend=(i==0),
                                    name=f"{res['Method']} Area",
                                    mode='lines'
                                ),
                                row=1, col=2
                            )
                    else:  # Simpson - show parabolic approximation
                        for i in range(0, len(x_points) - 1, 2):
                            if i+2 < len(x_points):
                                # Fit parabola through three points
                                x_parab = np.linspace(x_points[i], x_points[i+2], 50)
                                # Lagrange interpolation for parabola
                                def parabola(x):
                                    return (y_points[i] * (x - x_points[i+1]) * (x - x_points[i+2]) / 
                                           ((x_points[i] - x_points[i+1]) * (x_points[i] - x_points[i+2])) +
                                           y_points[i+1] * (x - x_points[i]) * (x - x_points[i+2]) / 
                                           ((x_points[i+1] - x_points[i]) * (x_points[i+1] - x_points[i+2])) +
                                           y_points[i+2] * (x - x_points[i]) * (x - x_points[i+1]) / 
                                           ((x_points[i+2] - x_points[i]) * (x_points[i+2] - x_points[i+1])))
                                
                                y_parab = [parabola(x) for x in x_parab]
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=list(x_parab) + [x_parab[-1], x_parab[0]],
                                        y=y_parab + [0, 0],
                                        fill="toself",
                                        fillcolor='rgba(0, 255, 0, 0.2)',
                                        line=dict(color='green', width=1),
                                        showlegend=(i==0),
                                        name=f"{res['Method']} Area",
                                        mode='lines'
                                    ),
                                    row=1, col=2
                                )
                
                # Plot evaluation points
                for res in results:
                    x_points = np.linspace(a, b, int(res['n']) + 1)
                    y_points = [f(x) for x in x_points]
                    
                    fig.add_trace(
                        go.Scatter(x=x_points, y=y_points, mode='markers',
                                  name=f"{res['Method']} Points", 
                                  marker=dict(size=6, symbol='circle')),
                        row=1, col=2
                    )
                
                # Plot 3: Convergence study
                if method == "Compare Both":
                    n_values = [4, 8, 16, 32, 64, 128]
                    trapezoidal_vals = []
                    simpson_vals = []
                    
                    for n_val in n_values:
                        try:
                            trap_val, _ = trapezoidal_rule(f, a, b, n_val)
                            trapezoidal_vals.append(trap_val)
                            
                            simp_val, _ = simpsons_rule(f, a, b, n_val)
                            simpson_vals.append(simp_val)
                        except:
                            continue
                    
                    if trapezoidal_vals and simpson_vals:
                        fig.add_trace(
                            go.Scatter(x=n_values[:len(trapezoidal_vals)], y=trapezoidal_vals,
                                      name="Trapezoidal Rule", mode='lines+markers',
                                      line=dict(color='red', width=2)),
                            row=2, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=n_values[:len(simpson_vals)], y=simpson_vals,
                                      name="Simpson's Rule", mode='lines+markers',
                                      line=dict(color='green', width=2)),
                            row=2, col=1
                        )
                
                fig.update_xaxes(title_text="x", row=1, col=1)
                fig.update_yaxes(title_text="f(x)", row=1, col=1)
                fig.update_xaxes(title_text="x", row=1, col=2)
                fig.update_yaxes(title_text="f(x)", row=1, col=2)
                fig.update_xaxes(title_text="Number of Intervals", row=2, col=1)
                fig.update_yaxes(title_text="Integral Value", row=2, col=1)
                
                fig.update_layout(
                    height=700,
                    template="plotly_white",
                    showlegend=True,
                    title_text="Numerical Integration Analysis"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed steps
                for res in results:
                    with st.expander(f"📋 {res['Method']} Rule - Detailed Steps"):
                        st.dataframe(pd.DataFrame(res['Steps']), use_container_width=True)
                        
                        # Show formula
                        if res['Method'] == "Trapezoidal":
                            st.latex(r"\int_a^b f(x)dx \approx \frac{h}{2}\left[f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\right]")
                        else:
                            st.latex(r"\int_a^b f(x)dx \approx \frac{h}{3}\left[f(x_0) + 4\sum_{i=1,3,5...}^{n-1} f(x_i) + 2\sum_{i=2,4,6...}^{n-2} f(x_i) + f(x_n)\right]")
        
        except Exception as e:
            st.markdown(f'<div class="error-message">❌ Error: {str(e)}</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 20px;">
        <p>🧮 <strong>NumericalPro Calculator</strong> | Advanced Numerical Methods Suite</p>
        <p style="font-size: 0.9em;">Built with Streamlit, Plotly, and NumPy | All calculations are performed numerically</p>
    </div>
""", unsafe_allow_html=True)