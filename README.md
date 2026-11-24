# Graph Signal Recovery (group project)

Group project for Mathematics Foundation: 7 methods to recover a graph signal, one per member.
The assignment itself is described in `Student_Graph_Signal_Assignment_MSSV_MODULO.ipynb` and `README_for_Students.md`.

## Repository layout

```
public_cases.json                     the 100 public graphs (no ground truth), shared by everyone
Student_Graph_Signal_Assignment_...   assignment notebook and README_for_Students.md (keep at the root,
                                      the assignment notebook reads public_cases.json from its own folder)
src/method_<#>/                       code and notebooks of method <#>
figures/method_<#>/                   figures of method <#>
results/method_<#>/                   xhat_<#>.npy of method <#>
```

Each member works only in their own `method_<#>` folders, so pushes do not conflict.
Notebooks find `public_cases.json` by searching upward from their folder, so they run from any subfolder:

```python
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "public_cases.json").exists())
FIG_DIR = ROOT / "figures" / "method_<#>"
RES_DIR = ROOT / "results" / "method_<#>"
```

### Method 4 (LAD / L1 data fit + Laplacian regularization)

Model: `min_x Σ_observed |x_i − y_i| + λ xᵀLx`. The L1 fit bounds the pull of each outlier and keeps
`x_hat_i = y_i` exactly at many nodes (sparse residual); for `λ ≤ λ0 = 1 / (2‖Ly‖∞)` the solution is `y` itself.

| file | what it is |
|---|---|
| `src/method_4/method_4.ipynb` | everything for method 4 (Vietnamese): data, solvers, correctness checks (CVXPY vs IRLS, KKT, `λ0`), IRLS convergence, 20 synthetic seeds, λ by `cv_score`, tests on all 100 public graphs, CSV export, result row, write-up (Mục VI) |
| `src/method_4/method4_lad.py` | the solvers as a module (`solve_lad`, `solve_lad_irls`, `solve_tikhonov`, `lambda0`, `kkt_violation`), same code as the notebook |
| `results/method_4/xhat_4.npy` | final `x_hat` on the real data |
| `results/method_4/method_4.csv` | `x_hat` in the 4-column submission format of Mục VIII |
| `figures/method_4/fig4_convergence.png`, `fig4_lambda_curve.png`, `fig4_graph_y_vs_xhat.png` | figures for the report |

Result row (placeholder MSSV `20260001`, re-run once the real list is fixed):

| method | parameters | synthetic RMSE, λ chosen by CV on each seed (seeds 0-19) | `cv_score` (real data, fold seed 0) |
|---|---|---|---|
| 4. LAD + Laplacian | `lam = 0.2234` (= 3.16·λ0) | 0.2596 ± 0.0576 | 0.3002 |

Needs `cvxpy` (solver Clarabel, installed with cvxpy). The notebook runs in about 2 minutes.

```python
sys.path.append(str(ROOT / "src" / "method_4"))
from method4_lad import solve_lad, lambda0
x_hat = solve_lad(G, y, lam=0.2234)
```

### Method 6 (two stages: flag outliers, then refill)

| file | what it is |
|---|---|
| `src/method_6/method_6.ipynb` | full version: vocabulary, worked examples, explanations, appendices |
| `src/method_6/method_6_short.ipynb` | short version: same code and results, only assumptions, formulas and results |
| `src/method_6/method6_outliers.py` | the method as a module (`flag_outliers`, `refill_outliers`, `solve_two_stage`) |
| `results/method_6/xhat_6.npy` | final `x_hat` on the real data |
| `figures/method_6/fig_y_vs_xhat_method6.png`, `fig_cv_k_method6.png` | figures for the report |

Not tracked by git (see `.gitignore`): `all_cases_figures/`, `logs/`, `.vscode/`, `*.code-workspace`.

## Setup

```bash
conda create -n graph_signal_recovery python=3.11
conda activate graph_signal_recovery
pip install numpy scipy networkx matplotlib cvxpy jupyter ipykernel nbstripout
python -m ipykernel install --user --name graph_signal_recovery --display-name "Python (graph_signal_recovery)"
```

In VS Code, pick the kernel "Python (graph_signal_recovery)" for the notebooks.

## Using method 6 in another method

```python
import sys
sys.path.append(str(ROOT / "src" / "method_6"))      # ROOT as in "Repository layout"
from method6_outliers import flag_outliers, refill_outliers, solve_two_stage

flagged = flag_outliers(G, y)              # bool array, True = suspected outlier
x_hat = refill_outliers(G, y, flagged)     # flagged nodes replaced by their neighbours' weighted average
x_hat = solve_two_stage(G, y)              # both stages in one call (same result)

# Use only the flags in your own solver:
x_hat = solve_yours(G, y, mask=~flag_outliers(G, y))
```

Defaults (`k = 2.5`, `lam = 0`, prune pass on) are the values used in the group result; no tuning is needed.
When you compute `cv_score`, call `flag_outliers` inside the solver so the flagging is redone in every fold.

## Version control (git)

### Notebook outputs are stripped automatically

This repository uses [nbstripout](https://github.com/kynan/nbstripout) as a git filter.
When you `git add` a notebook, git stores it **without outputs** (no plots, printed results or execution counts).
Your file on disk keeps its outputs; only the committed copy is clean. This keeps diffs readable and avoids
merge conflicts on outputs. To see results, run the notebook after pulling.

The rule is in `.gitattributes` (committed). The filter command itself lives in each person's local git
config, so **every teammate must run this once after cloning**:

```bash
conda activate graph_signal_recovery
pip install nbstripout
nbstripout --install --attributes .gitattributes
nbstripout --status
```

Without this step, `git add` on a notebook fails with an error about the `nbstripout` filter
(the filter is marked as required).

### Everyday commands

```bash
git status                        # what changed
git diff                          # line-by-line changes (notebooks shown without outputs)
git add src/method_6/method_6.ipynb   # stage a file
git commit -m "feat(method6): add prune pass to stage 1"
git log --oneline                 # history
```

Working with a shared remote (for example GitHub):

```bash
git remote add origin <repository-url>    # once
git push -u origin main                   # first push
git pull                                  # get teammates' changes before you start
git push                                  # send your commits
```

Suggested habits:
- Pull before you start working, commit small steps, push when a step works.
- Work on a branch for bigger changes: `git switch -c feature/method6-tikhonov`, then open a pull request.
- Commit messages follow Conventional Commits: `<type>(<scope>): <description>`,
  e.g. `fix(method6): handle nodes with no observed neighbour`, `docs(readme): add setup steps`.
- Edit only your own `src/method_<#>/`, `figures/method_<#>/` and `results/method_<#>/` to avoid conflicts; code others reuse goes in a `.py` file.
