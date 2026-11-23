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

### Method 3 (Huber + Laplacian, solved by IRLS)

| file | what it is |
|---|---|
| `src/method_3/method_3.ipynb` | full version: vocabulary, worked examples, IRLS derivation, appendices |
| `src/method_3/method_3_short.ipynb` | short version: same code and results, only assumptions, formulas and results |
| `src/method_3/method3_huber.py` | the method as a module (`solve_huber`, `solve_tikhonov`, `noise_scale`) |
| `results/method_3/xhat_3.npy` | final `x_hat` on the real data |
| `figures/method_3/fig_huber_loss_method3.png`, `fig_y_vs_xhat_method3.png`, `fig_convergence_method3.png`, `fig_cv_delta_method3.png` | figures for the report |

Model: `min_x sum_i Huber_delta(x_i - y_i) + lam * x^T L x`, solved by IRLS, i.e. the linear system
`(W + 2*lam*L) x = W y` with `w_i = min(1, delta/|x_i - y_i|)` in every iteration.
Group result: `delta = 0.5 * s0 = 0.337` (with `s0 = 1.4826 * MAD(y_i - neighbour mean)`) and `lam = 0.1`;
40 IRLS iterations, `cv_score = 0.3309` (5 splits) / `0.3745` (20 splits).

Using method 3 in another method:

```python
import sys
sys.path.append(str(ROOT / "src" / "method_3"))      # ROOT as in "Repository layout"
from method3_huber import solve_huber, solve_tikhonov, noise_scale

x_hat = solve_huber(G, y, delta=0.337, lam=0.1)      # or solve_huber(G, y) for the group defaults
x_hat = solve_huber(G, y, mask=mask, delta=0.337, lam=0.1)   # inside cv_score
x_l2 = solve_tikhonov(G, y, lam=0.1)                 # method 1 reference

x_hat, info = solve_huber(G, y, delta=0.337, lam=0.1, return_info=True)
info["weights"]          # w_i < 0.5 = node down-weighted by Huber (suspected outlier)
info["n_iter"], info["kkt"]
```

Defaults are `delta = 0.5 * noise_scale(G, y)` and `lam = 0.1` (the group result); `noise_scale` adapts `delta`
to the data scale, but `lam` should be re-tuned with `cv_score` on a different case (see Appendix A of the notebook).

### Method 6 (two stages: flag outliers, then refill)

| file | what it is |
|---|---|
| `src/method_6/method_6.ipynb` | full version: vocabulary, worked examples, explanations, appendices |
| `src/method_6/method_6_short.ipynb` | short version: same code and results, only assumptions, formulas and results |
| `src/method_6/method6_outliers.py` | the method as a module (`flag_outliers`, `refill_outliers`, `solve_two_stage`) |
| `results/method_6/xhat_6.npy` | final `x_hat` on the real data |
| `figures/method_6/fig_y_vs_xhat_method6.png`, `fig_cv_k_method6.png` | figures for the report |

Not tracked by git (see `.gitignore`): `.venv/`, `all_cases_figures/`, `logs/`, `.vscode/`, `*.code-workspace`.

## Setup

```bash
conda create -n graph_signal_recovery python=3.11
conda activate graph_signal_recovery
pip install numpy scipy networkx matplotlib jupyter ipykernel nbstripout
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
