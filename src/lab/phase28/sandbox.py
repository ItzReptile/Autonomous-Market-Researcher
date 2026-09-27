"""
Phase 28 Code Generation Probe: AST Security & Subprocess Sandbox.
Enforces:
1. Static analysis rejecting lookahead bias (.shift(-n), future indexing) and dangerous imports/builtins.
2. Subprocess execution with strict 5.0-second hard timeout and module stripping.
3. Strict return value validation (pd.Series, matching length, bounds [-1.0, 1.0]).
"""
import ast
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_MODULES = {"numpy", "np", "pandas", "pd", "math", "scipy", "scipy.stats"}
FORBIDDEN_CALLS = {"open", "eval", "exec", "__import__", "globals", "locals", "getattr", "setattr", "delattr", "compile"}


@dataclass
class StaticCheckResult:
    passed: bool
    reason: str
    rejected_pattern: Optional[str] = None


def check_code_static(code_str: str) -> StaticCheckResult:
    """
    Performs static regex and AST analysis to detect lookahead bias and security violations.
    """
    # 1. Regex checks for lookahead patterns
    lookahead_patterns = [
        (r"\.shift\s*\(\s*-\s*\d+", "negative shift call (.shift(-n))"),
        (r"shift\s*\(\s*periods\s*=\s*-\s*\d+", "negative shift keyword argument (shift(periods=-n))"),
        (r"\.iloc\s*\[[^\]]*\+\s*[1-9]\d*", "forward iloc indexing (iloc[... + n])"),
        (r"\.loc\s*\[[^\]]*\+\s*[1-9]\d*", "forward loc indexing (loc[... + n])"),
    ]
    for pat, desc in lookahead_patterns:
        if re.search(pat, code_str, re.IGNORECASE):
            return StaticCheckResult(passed=False, reason=f"Lookahead bias detected: {desc}", rejected_pattern=desc)

    # 2. Parse AST
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return StaticCheckResult(passed=False, reason=f"SyntaxError in generated code: {e}")

    # 3. Walk AST for security and forward shift arguments
    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod not in ALLOWED_MODULES:
                    return StaticCheckResult(passed=False, reason=f"Unauthorized import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod not in ALLOWED_MODULES:
                return StaticCheckResult(passed=False, reason=f"Unauthorized from-import: {node.module}")

        # Check forbidden function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                return StaticCheckResult(passed=False, reason=f"Forbidden built-in call: {node.func.id}()")

            # Check shift calls in AST: obj.shift(-...)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "shift":
                for arg in node.args:
                    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                        return StaticCheckResult(passed=False, reason="Lookahead bias: negative argument in .shift()")
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and arg.value < 0:
                        return StaticCheckResult(passed=False, reason="Lookahead bias: negative argument in .shift()")
                for kw in node.keywords:
                    if kw.arg in ("periods", "n"):
                        if isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.op, ast.USub):
                            return StaticCheckResult(passed=False, reason="Lookahead bias: negative periods in .shift()")
    return StaticCheckResult(passed=True, reason="Static checks passed")


class LookbackValidator(ast.NodeVisitor):

    def __init__(self, min_lookback: int = 12):
        self.min_lookback = min_lookback
        self.violations = []
        self.constants = {}

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.constants[target.id] = int(node.value.value)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if name == "rolling":
                val = None
                if node.args:
                    val = self._extract_int(node.args[0])
                for kw in node.keywords:
                    if kw.arg == "window":
                        val = self._extract_int(kw.value)
                if val is not None and val < self.min_lookback:
                    self.violations.append(f"rolling window {val} < {self.min_lookback}")
            elif name in ("diff", "pct_change"):
                val = 1  # default if no positional/keyword arg
                if node.args:
                    val = self._extract_int(node.args[0])
                for kw in node.keywords:
                    if kw.arg == "periods":
                        val = self._extract_int(kw.value)
                if val is not None and val < self.min_lookback:
                    self.violations.append(f"{name} period {val} < {self.min_lookback}")
            elif name == "ewm":
                for kw in node.keywords:
                    if kw.arg in ("span", "halflife", "com"):
                        val = self._extract_int(kw.value)
                        if val is not None and val < self.min_lookback:
                            self.violations.append(f"ewm {kw.arg} {val} < {self.min_lookback}")
        self.generic_visit(node)

    def _extract_int(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return int(node.value)
        if isinstance(node, ast.Name) and node.id in self.constants:
            return self.constants[node.id]
        return None


def check_code_lookbacks(code_str: str, min_lookback: int = 12) -> Tuple[bool, List[str]]:
    """
    Statically checks whether all rolling windows, diffs, and pct_change periods are >= min_lookback.
    Returns (passed, list_of_violations).
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, [f"SyntaxError: {e}"]

    validator = LookbackValidator(min_lookback=min_lookback)
    validator.visit(tree)
    passed = len(validator.violations) == 0
    return passed, validator.violations


class CompositionValidator(ast.NodeVisitor):
    """
    AST visitor that enforces canonical additive composition:
    1. Forbids division by signed/zero-crossing denominators (e.g. dividing by returns, diffs, or signed metrics).
       Permitted denominators: rolling std/var, rolling mean of volume/price, high-low range, constants, etc.
    2. Forbids multiplication between two signed data series/variables (e.g. mom * vol_z, ret * accel).
       Permitted multiplications: scalar weights * series (e.g. 0.4 * mom_12), or constant scaling.
    """
    def __init__(self):
        self.violations: List[str] = []
        self.var_types: Dict[str, str] = {}

    def _is_scalar_constant(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            return True
        return False

    def _infer_type(self, node: ast.AST) -> str:
        """
        Returns 'signed', 'positive', 'scalar', or 'unknown'.
        """
        if self._is_scalar_constant(node):
            return "scalar"

        if isinstance(node, ast.Name):
            name = node.id
            if name in self.var_types:
                return self.var_types[name]
            name_lower = name.lower()
            if any(k in name_lower for k in ["std", "var", "vol_mean", "volume_mean", "parkinson", "atr", "range", "high_low", "scale", "abs"]):
                return "positive"
            if any(k in name_lower for k in ["mom", "ret", "accel", "diff", "pct", "z_", "zscore", "score", "trend", "curvature"]):
                return "signed"
            if any(k in name_lower for k in ["vol", "close", "high", "low", "open", "price", "volume"]):
                return "positive"
            return "unknown"

        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                col = node.slice.value.lower()
                if col in ("open", "high", "low", "close", "volume"):
                    return "positive"
            elif isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                # e.g. weights[0]
                if isinstance(node.value, ast.Name) and self.var_types.get(node.value.id) == "scalar":
                    return "scalar"
            return "unknown"

        if isinstance(node, ast.Call):
            # Check if call is a math/numpy function on scalar arguments (e.g. np.log(2.0), math.sqrt(5))
            all_scalar_args = node.args and all(self._infer_type(a) == "scalar" for a in node.args)
            if all_scalar_args:
                return "scalar"

            if isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr in ("pct_change", "diff"):
                    return "signed"
                if attr in ("std", "var"):
                    return "positive"
                if attr in ("rolling", "ewm", "shift"):
                    return self._infer_type(node.func.value)
                if attr in ("mean", "sum", "median", "min", "max"):
                    return self._infer_type(node.func.value)
                if attr in ("fillna", "dropna", "clip", "replace"):
                    return self._infer_type(node.func.value)
                if attr in ("sqrt", "exp", "abs", "pow"):
                    return "positive"
                if attr == "tanh":
                    return "signed"
            elif isinstance(node.func, ast.Name):
                if node.func.id in ("abs", "min", "max", "sqrt", "pow"):
                    return "positive"
            return "unknown"

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                t = self._infer_type(node.operand)
                return "scalar" if t == "scalar" else ("signed" if t in ("signed", "positive") else t)
            return self._infer_type(node.operand)

        if isinstance(node, ast.BinOp):
            left_t = self._infer_type(node.left)
            right_t = self._infer_type(node.right)

            # Both scalars produce scalar
            if left_t == "scalar" and right_t == "scalar":
                return "scalar"

            if isinstance(node.op, ast.Pow):
                if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)) and node.right.value % 2 == 0:
                    return "positive"

            if isinstance(node.op, ast.Sub):
                l_str = ast.unparse(node.left).lower()
                r_str = ast.unparse(node.right).lower()
                if ("high" in l_str and "low" in r_str) or ("max" in l_str and "min" in r_str):
                    return "positive"
                return "signed"

            if isinstance(node.op, ast.Add):
                if left_t == "signed" or right_t == "signed":
                    return "signed"
                if left_t in ("positive", "scalar") and right_t in ("positive", "scalar"):
                    return "positive"
                return "signed"

            if isinstance(node.op, ast.Mult):
                if left_t == "scalar":
                    return right_t
                if right_t == "scalar":
                    return left_t
                if left_t == "positive" and right_t == "positive":
                    return "positive"
                return "signed"

            if isinstance(node.op, ast.Div):
                if left_t == "signed" and right_t in ("positive", "scalar"):
                    return "signed"
                if left_t in ("positive", "scalar") and right_t in ("positive", "scalar"):
                    return "positive"
                return "signed"

        return "unknown"

    def visit_Assign(self, node: ast.Assign):
        inferred = self._infer_type(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                name_lower = name.lower()
                if inferred == "scalar":
                    self.var_types[name] = "scalar"
                elif isinstance(node.value, (ast.List, ast.Tuple)) and all(self._infer_type(elt) == "scalar" for elt in node.value.elts):
                    self.var_types[name] = "scalar"
                elif any(k in name_lower for k in ["std", "variance", "vol_mean", "volume_mean", "parkinson", "atr", "range"]):
                    self.var_types[name] = "positive"
                elif inferred != "unknown":
                    self.var_types[name] = inferred
                elif any(k in name_lower for k in ["mom", "ret", "accel", "diff", "pct", "z_", "zscore", "ratio"]):
                    self.var_types[name] = "signed"
                elif any(k in name_lower for k in ["vol", "close", "high", "low", "open", "price", "volume"]):
                    self.var_types[name] = "positive"
            elif isinstance(target, (ast.Tuple, ast.List)):
                # Unpacking e.g. w1, w2 = 0.5, 0.3
                if isinstance(node.value, (ast.Tuple, ast.List)) and len(node.value.elts) == len(target.elts):
                    for t_sub, v_sub in zip(target.elts, node.value.elts):
                        if isinstance(t_sub, ast.Name):
                            self.var_types[t_sub.id] = self._infer_type(v_sub)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        left_t = self._infer_type(node.left)
        right_t = self._infer_type(node.right)

        # Check 1: Denominator in Division cannot be signed/zero-crossing
        if isinstance(node.op, ast.Div):
            if right_t == "signed":
                self.violations.append(
                    f"Forbidden division by signed/zero-crossing quantity: {ast.unparse(node.right)[:40]}"
                )

        # Check 2: Multiplication cannot multiply two non-scalar signed quantities
        elif isinstance(node.op, ast.Mult):
            is_l_scalar = (left_t == "scalar")
            is_r_scalar = (right_t == "scalar")
            if not is_l_scalar and not is_r_scalar:
                if left_t == "signed" and right_t == "signed":
                    self.violations.append(
                        f"Forbidden multiplication of two signed quantities: ({ast.unparse(node.left)[:30]}) * ({ast.unparse(node.right)[:30]})"
                    )
                elif (left_t == "signed" and right_t not in ("scalar", "positive")) or (right_t == "signed" and left_t not in ("scalar", "positive")):
                    self.violations.append(
                        f"Forbidden nonlinear multiplication of signed series: ({ast.unparse(node.left)[:30]}) * ({ast.unparse(node.right)[:30]})"
                    )

        self.generic_visit(node)


def check_code_composition(code_str: str) -> Tuple[bool, List[str]]:
    """
    Statically checks whether code adheres to canonical additive composition:
    - Forbids dividing by signed/zero-crossing denominators.
    - Forbids multiplying signed series by other signed series.
    Returns (passed, list_of_violations).
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, [f"SyntaxError: {e}"]
    validator = CompositionValidator()
    validator.visit(tree)
    passed = len(validator.violations) == 0
    return passed, validator.violations


@dataclass

class ExecutionResult:
    status: str                         # SUCCESS, LOOKAHEAD_REJECTED, TIMEOUT, SYNTAX_ERROR, RUNTIME_ERROR, INVALID_OUTPUT
    weights: Optional[np.ndarray] = None
    error_message: str = ""
    duration_sec: float = 0.0
    stdout: str = ""
    stderr: str = ""


SANDBOX_HARNESS_TEMPLATE = """
import sys
import pickle
import numpy as np
import pandas as pd

# Strip dangerous modules
for mod in ['socket', 'http', 'urllib', 'requests', 'subprocess', 'shutil', 'os']:
    sys.modules[mod] = None

# Injected user code
{user_code}

def main():
    data_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(data_path, 'rb') as f:
        df = pickle.load(f)

    fn = globals().get('compute_raw_factor') or globals().get('compute_signal')
    if fn is None:
        raise NameError("Function 'compute_raw_factor' (or 'compute_signal') was not defined in generated code")

    res = fn(df)

    if not isinstance(res, pd.Series):
        raise TypeError(f"Signal function must return pd.Series, got {{type(res).__name__}}")

    if len(res) != len(df):
        raise ValueError(f"Output series length {{len(res)}} does not match df length {{len(df)}}")

    # Convert to float numpy array, fill nans with 0.0
    arr = res.to_numpy(dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Check bounds
    if np.any(arr > 1.05) or np.any(arr < -1.05):
        raise ValueError(f"Output factor exceeds [-1.0, 1.0] bounds (min={{np.min(arr)}}, max={{np.max(arr)}})")


    arr = np.clip(arr, -1.0, 1.0)

    with open(out_path, 'wb') as f:
        pickle.dump(arr, f)

if __name__ == '__main__':
    main()
"""



class SignalSandbox:
    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout_seconds = timeout_seconds

    def execute_code(
        self,
        code_str: str,
        df: pd.DataFrame,
        allow_lookahead_for_control: bool = False,
    ) -> ExecutionResult:
        """
        allow_lookahead_for_control
            Bypasses the lookahead rejection. Exists ONLY for positive-control
            diagnostics, where a deliberately clairvoyant strategy is used to
            test whether the gate stack can pass anything at all. Default False,
            and every use prints a warning, because code that reaches this path
            is by construction not a tradeable strategy and must never be scored
            as a candidate. The security checks (imports, forbidden builtins)
            still apply -- only the lookahead veto is lifted.
        """
        # 1. Static security and lookahead check
        chk = check_code_static(code_str)
        if not chk.passed and allow_lookahead_for_control and "Lookahead" in chk.reason:
            print(f"[SANDBOX] *** LOOKAHEAD BYPASS ENGAGED (control diagnostic) *** {chk.reason}")
            chk = StaticCheckResult(passed=True, reason="lookahead bypassed for control")
        if not chk.passed:
            return ExecutionResult(
                status="LOOKAHEAD_REJECTED" if "Lookahead" in chk.reason else "SYNTAX_ERROR",
                error_message=chk.reason,
            )

        # 2. Write temp files for execution
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_file = tmp_path / "df_in.pkl"
            out_file = tmp_path / "weights_out.pkl"
            script_file = tmp_path / "exec_script.py"

            with open(data_file, "wb") as f:
                pickle.dump(df, f)

            script_content = SANDBOX_HARNESS_TEMPLATE.format(user_code=code_str)
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(script_content)

            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, str(script_file), str(data_file), str(out_file)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    cwd=str(tmp_path),
                )
                dur = time.perf_counter() - t0
                if proc.returncode != 0:
                    return ExecutionResult(
                        status="RUNTIME_ERROR",
                        error_message=proc.stderr.strip() or proc.stdout.strip() or f"Process exited with returncode {proc.returncode}",
                        duration_sec=dur,
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                    )

                if not out_file.exists():
                    return ExecutionResult(
                        status="INVALID_OUTPUT",
                        error_message="Output file was not created",
                        duration_sec=dur,
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                    )

                with open(out_file, "rb") as f:
                    weights = pickle.load(f)

                return ExecutionResult(
                    status="SUCCESS",
                    weights=weights,
                    duration_sec=dur,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                )

            except subprocess.TimeoutExpired:
                dur = time.perf_counter() - t0
                return ExecutionResult(
                    status="TIMEOUT",
                    error_message=f"Execution exceeded hard timeout of {self.timeout_seconds}s",
                    duration_sec=dur,
                )
            except Exception as e:
                dur = time.perf_counter() - t0
                return ExecutionResult(
                    status="RUNTIME_ERROR",
                    error_message=str(e),
                    duration_sec=dur,
                )


def apply_hysteresis_filter(
    raw_factor: np.ndarray,
    theta_enter: float = 0.5,
    theta_exit: float = 0.2,
    position_size: float = 1.0,
) -> np.ndarray:
    """
    Standard stateful hysteresis filter applied mechanically to raw continuous factors:
        pos[0] = 0
        for t in 1..n:
            if abs(raw_factor[t]) > theta_enter:
                pos[t] = sign(raw_factor[t]) * position_size
            elif abs(raw_factor[t]) < theta_exit:
                pos[t] = 0
            else:
                pos[t] = pos[t-1]  # hold
    """
    n = len(raw_factor)
    pos = np.zeros(n, dtype=np.float64)
    pos[0] = 0.0
    for t in range(1, n):
        val = raw_factor[t]
        abs_val = abs(val)
        if abs_val > theta_enter:
            pos[t] = float(np.sign(val) * position_size)
        elif abs_val < theta_exit:
            pos[t] = 0.0
        else:
            pos[t] = pos[t - 1]
    return pos


def apply_adaptive_hysteresis_filter(
    raw_factor: np.ndarray,
    enter_pct: float = 95.0,
    exit_pct: float = 50.0,
    window: int = 720,
    min_obs: int = 336,
    position_size: float = 1.0,
) -> np.ndarray:
    """
    Hysteresis on each factor's OWN scale, via causal rolling percentiles.

    The fixed 0.50/0.20 thresholds were calibrated for the Trial 106 family,
    whose volume-ratio term sits near 1.0 and lifts the tanh output. Measured
    on the 3-year window, 0.50 sits at roughly the 97th percentile of that
    factor's |S| distribution. A generated factor whose natural scale is
    +/-0.15 never reaches 0.50 at all: in Phase 32, 15 of 26 evaluated trials
    placed ZERO trades, not because the idea was poor but because the entry
    threshold was unreachable. Those trials tested nothing.

    Percentiles judge each formula on its own distribution, so a small-scale
    factor and a large-scale one face the same *statistical* bar.

    STRICTLY CAUSAL: thresholds at bar t come from |S| over [t-window, t),
    excluding t itself. No future information, and no full-sample scaling.

    Before min_obs observations accumulate, the position is held rather than
    guessed -- the same fail-closed instinct used elsewhere in the pipeline.
    """
    n = len(raw_factor)
    pos = np.zeros(n, dtype=np.float64)
    if n == 0:
        return pos
    a = np.abs(np.asarray(raw_factor, dtype=np.float64))

    for t in range(1, n):
        lo = max(0, t - window)
        hist = a[lo:t]                      # strictly before t
        if hist.size < min_obs:
            pos[t] = pos[t - 1]
            continue
        th_enter = float(np.percentile(hist, enter_pct))
        th_exit = float(np.percentile(hist, exit_pct))
        val = a[t]
        # Inclusive comparisons, entry checked first.
        #
        # A saturated or discrete factor breaks strict inequalities. Phase 32
        # trial 42 emits |S| = 1.0 on 99% of bars across only 4 distinct
        # values, so every rolling percentile collapses to 1.0 and `val >
        # th_enter` can never fire -- the position would never open, which is
        # the opposite of that factor's behaviour under fixed thresholds.
        # With >=, a saturated factor enters on its plateau and exits on its
        # zeros, matching what the fixed threshold did for it.
        if val >= th_enter:
            pos[t] = float(np.sign(raw_factor[t]) * position_size)
        elif val <= th_exit:
            pos[t] = 0.0
        else:
            pos[t] = pos[t - 1]
    return pos


def compute_reference_performance(
    weights: np.ndarray,
    close_prices: np.ndarray,
    cost_rate: float = 0.0010, # 10 bps round-trip friction
) -> Dict[str, float]:
    """
    Computes reference backtest performance for a single asset.
    Physics:
    - Weight decided at bar t applies to price return from t to t+1.
    - Turnover delta |w_{t} - w_{t-1}| pays cost_rate.
    """
    n_bars = len(close_prices)
    if len(weights) != n_bars:
        raise ValueError("Weights and prices must have identical length")

    # Bar-to-bar returns: r_{t+1} = (P_{t+1} - P_t) / P_t
    ret_arr = np.zeros(n_bars)
    ret_arr[1:] = (close_prices[1:] - close_prices[:-1]) / np.maximum(1e-8, close_prices[:-1])

    # Position at bar t earns return at bar t+1
    gross_pnl = np.zeros(n_bars)
    turnover = np.zeros(n_bars)
    costs = np.zeros(n_bars)

    cur_w = 0.0
    for t in range(n_bars - 1):
        target_w = weights[t]
        dw = abs(target_w - cur_w)
        turnover[t] = dw
        costs[t] = dw * cost_rate
        # Return across t -> t+1
        gross_pnl[t + 1] = target_w * ret_arr[t + 1]
        cur_w = target_w

    net_pnl = gross_pnl - costs

    # Ignore first 24 warmup bars
    valid_net = net_pnl[24:]
    valid_gross = gross_pnl[24:]

    mu_net = np.mean(valid_net)
    sd_net = np.std(valid_net) + 1e-8
    sharpe_net = float(np.sqrt(8760) * (mu_net / sd_net))

    mu_gross = np.mean(valid_gross)
    sd_gross = np.std(valid_gross) + 1e-8
    sharpe_gross = float(np.sqrt(8760) * (mu_gross / sd_gross))

    # Max Drawdown
    cum_ret = np.cumsum(valid_net)
    running_max = np.maximum.accumulate(cum_ret)
    dd = cum_ret - running_max
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    annual_turnover = float(np.sum(turnover[24:]) * (8760.0 / len(valid_net))) if len(valid_net) > 0 else 0.0

    return {
        "annualized_sharpe_net": round(sharpe_net, 4),
        "annualized_sharpe_gross": round(sharpe_gross, 4),
        "max_drawdown": round(max_dd, 4),
        "annual_turnover": round(annual_turnover, 2),
    }
