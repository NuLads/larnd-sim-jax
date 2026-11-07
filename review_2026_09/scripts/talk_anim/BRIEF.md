# Talk animations: shared brief

Pierre Granger is giving a Keynote talk on calibrating a liquid-argon TPC with a differentiable simulation (larnd-sim-jax), on
data-like samples. The audience is physicists who know LArTPCs but not this method. Animations play full-bleed on a slide while
the presenter talks, so they must explain the mechanism visually with very little text.

## Non-negotiables
- **Real data only.** Every plotted number comes from a saved fit, a cache, or the production simulation code run on real
  inputs. Never invent values, never interpolate between saved fit states and present it as fit evolution. A purely schematic
  element (for example, an arrow) is allowed if it carries no data. If an element is illustrative, say so on screen.
- Read the production code before drawing the physics. Do not trust your memory of what LArPix or larnd-sim does.
- All of this is **simulation**, not detector data. Each animation carries a footer saying what the data is and which run or file
  it came from (`footer(fig, text, right=...)`). In this project the word "data" means the noisy stochastic simulation at known
  truth (the fit's target).
- Do not modify anything outside `review_2026_09/scripts/talk_anim/`. No git commands. No SLURM submissions and no GPU use: GPUs
  are scarce. Only run through the allocation wrapper below. Do not edit `common.py`; if you need a change, add a helper to
  your own script.

## Tools
- Scripts live in `/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/talk_anim/`. Name them `a<NN>_<slug>.py`.
- Run everything with `T=/sdf/scratch/neutrino/pgranger/talk_animations; $T/run.sh <ncpus> <abs path to script> [args]`. It runs on
  a shared 64-CPU milano node with JAX on CPU, from the repo root, with `src/`, `talk_anim/`, `eventdisplay/`, `paperfigs/` on the path.
  Use at most 12 CPUs per command. The login node is overloaded, so never run Python there. Long runs: use the Bash tool's
  `run_in_background`, or poll with bounded waits.
- Shell: the Bash tool is zsh, so unmatched globs abort; use `bash -c` around glob-heavy snippets.
- `common.py` holds the rendering toolkit; read its docstring first. `draw(s, i, t)` must be a pure function of the frame index,
  because frames render out of order in parallel workers.
- `--preview 3,12.5` writes 1920x1080 PNGs into `$T/preview/<name>/`. `--draft` writes a low-res MP4. `--final` writes the 4K MP4,
  the poster PNG and beats JSON into `$T/output/`.
- Look at frames with the Read tool. Check beats and mid-transition frames, not just the end state.

## Shared data (`$T/data/`)
- `traces.npz`: per-iteration traces of the production runs. Arrays `Ab, eField, tran_diff, long_diff, lifetime` with shape
  (runs, 20001); `losses_iter`, `pos_residual_iter`, `pos_residual_batch`, `dedx_mae_truth_iter`, `mean_dedx_iter`, `<p>_target`,
  `<p>_init`, `names`, `data_seed`, `argv`. Runs: `EXP<s>C75H` = 36 realistic-end samples (endpoints smeared by 1 mm),
  `KN10N<s>C75H` = 12 true-end samples. Use the `names` array to select.
- `hero.pkl` (pickle): the example event. Sample 8, batch 35, event 2, main track (2,4), 63 cm, the same one as the paper's event
  display (`review_2026_09/scripts/eventdisplay/event_display.py` shows how to read the same quantities). Keys: `data` (target hits:
  pixel_id, pixel_x, pixel_y, ticks, adcs, q in ke), `fields`, `tfields`, `col`, `true_rows` (true segments), `states` (`start`,
  `iter10000`, `iter14000`, `iter20000`, each with `calib`, `rows_tracks`, `pred`, `dev_um`, `s_cm`, `dedx_main`), `main_ctx`, `calib_true`, `calib_init`.
  `pred` has per-pixel `pid, x, y, nhit, q, lam0, t0, t0sd`, plus `hit_prob` (Npix, Nslot, Ntick, P(hit n at tick t)) and `q_tick`.
- Rebuilding from a checkpoint: `review_2026_09/scripts/eventdisplay/rebuild.py` (`FitRebuild`). Example of evaluating the production model
  on CPU: `extract_hero.py`. Truth parameters and response: `paper_common.params()`, in `/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS`.
- `review_2026_09/scripts/paperfigs/` holds the paper's figure scripts, many of them directly reusable.
- The paper draft (`/sdf/group/neutrino/pgranger/NuLads_draft/main.tex`) describes the method. Read-only.

## Visual system
Every animation should look like it belongs to one talk.
- 3840x2160 output, designed in 1920x1080 design pixels. Margins 60 px. Dark theme by default. Colours only from `C.*` tokens, so that
  `--theme light` also works; never hard-code hex.
- **Colour has one meaning everywhere:** data = `C.data` (orange), model = `C.model` (blue), truth = `C.truth` (ink, dashed),
  start/guess = `C.start` (grey). The five calibration parameters use `C.Ab, C.eField, C.tran_diff, C.long_diff, C.lifetime`.
  Pixel charge: `cmap_to(C.data)` for data, `cmap_to(C.model)` for model. Residuals: `cmap_diverging()`, centred on zero. No rainbow maps.
- Names: `PARAM_LABEL` and `PARAM_NAME` from `common.py`.
- Type: nothing below 22 design px. Title via `title()`, one line, stating the idea, not the method. At most one short caption
  at a time. Text uses the ink tokens, never a series colour.
- Marks: 2-4 px lines, recessive grid, few ticks, no chart junk, no dual y-axes. Label lines directly when there are at most 4; otherwise
  a legend. No text on top of data; check for collisions in every beat.
- One idea per animation. Motion must explain something: use `ease()`, no gratuitous camera moves, nothing flickers.
- Pacing: 20-50 s. Hold long enough for a presenter to say a sentence at each beat. Static end hold of at least 2.5 s.
- Provide `beats=[(t, label), ...]`: the natural moments for the presenter to pause. Every beat should look complete as a still frame.

## Process
1. Read the code and data. 2. Sketch the beats. 3. `--preview` stills and look at them. 4. Fix. 5. `--draft`, then look at frames around
transitions. 6. Critique it as a hostile reviewer: is it clear with the sound off? Is any label wrong? Is any number unsupported? Iterate at
least twice. 7. `--final`. Then check the MP4 plays and its duration with `$T/tools/ffmpeg -i <file>`.

## Report back
Output path, duration, beats, exact data sources, anything you could not do, and any finding that looks surprising or scientifically
worrying. Say plainly what is schematic. Do not claim checks you did not run.
