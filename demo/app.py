"""Gradio interface for the kidney-segmentation demo (DEMO_PLAN.md). All logic is in demo_core.

Run:  python app.py        then open http://127.0.0.1:7860
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

import demo_core as D

SAMPLES = D.sample_catalogue()
OUT_DIR = Path(tempfile.mkdtemp(prefix="ie643_demo_"))
_CASES: dict = {}                    # loaded cases, so re-running other models is instant

ARMS = {                             # choice label -> catalogue key prefix
    "Arm 1b: SAM ViT-B off the shelf (+ box)": "arm1b",
    "Arm 2: SAM ViT-B + LoRA, the teacher (+ box)": "arm2",
    "Arm 3: U-Net, no prompt": "arm3_s",
    "Arm 3 ablation: U-Net + box": "arm3box_s",
    "Arm 4: U-Net distilled from arm 2, no prompt": "arm4_s",
}
DEFAULT_SEED = "644"                 # best validation score of the three arm-3 seeds (D64)


def available_arms() -> list:
    cat = D.model_catalogue()
    return [lab for lab, pre in ARMS.items() if any(k == pre or k.startswith(pre) for k in cat)]


def seeds() -> list:
    s = sorted({k.split("_s")[-1] for k in D.model_catalogue() if k.startswith(("arm3_s", "arm4_s"))})
    return s or [DEFAULT_SEED]


def resolve(arm_labels: list, seed: str) -> tuple[list, list]:
    """Chosen arms -> model specs; the seed applies to arms 3 and 4 (the box ablation has one seed)."""
    cat, specs, notes = D.model_catalogue(), [], []
    for lab in arm_labels:
        pre = ARMS[lab]
        if pre in cat:
            specs.append(cat[pre])
        elif pre + seed in cat:
            specs.append(cat[pre + seed])
        else:
            alt = sorted(k for k in cat if k.startswith(pre))
            if alt:                  # e.g. the box ablation, trained with one seed only
                specs.append(cat[alt[0]])
                if len(alt) > 1:
                    notes.append(f"{lab.split(':')[0]}: no seed {seed}, used {alt[0].split('_s')[-1]}")
    return specs, notes


def get_case(source, sample, scan, label, dataset):
    if source == "Ready-made test case":
        if sample not in SAMPLES:
            raise D.DemoError("pick a test case from the list")
        key = ("sample", sample)
        if key not in _CASES:
            _CASES[key] = D.load_sample(SAMPLES[sample])
        return _CASES[key]
    if not scan or not label:
        raise D.DemoError("upload both the scan and its ground-truth label file")
    return D.load_nifti_case(scan, label, {"AMOS": "amos", "WORD": "word"}[dataset])


def case_info(case: D.Case) -> str:
    Z, h, w = case.img.shape
    return (f"**{case.label}**: {Z} slices of {h}×{w} px, pixel {case.sp_row:.3g}×{case.sp_col:.3g} mm, "
            f"slice {case.sp_slice:.3g} mm. **{case.n_instances()} kidney instances** are scored "
            f"(one per kidney per slice with at least 50 px of kidney, as in our evaluation).")


def run(source, sample, scan, label, dataset, arm_labels, seed, noise, zoom, progress=gr.Progress()):
    try:
        if not arm_labels:
            raise D.DemoError("tick at least one model")
        progress(0, desc="loading the scan")
        case = get_case(source, sample, scan, label, dataset)
        specs, notes = resolve(arm_labels, seed)
        level = int(noise.rstrip("%")) / 100
        preds, rows, files = [], [], []
        for k, spec in enumerate(specs):
            progress(k / len(specs), desc=f"running {spec.title.split(':')[0]}")
            p = D.predict(spec, case, level, progress=lambda f, k=k, spec=spec: progress(
                (k + f) / len(specs), desc=f"running {spec.title.split(':')[0]}"))
            s = D.score(p, case)
            preds.append(p)
            rows.append([
                spec.title, "box" if spec.needs_box else "none",
                f"{int(round(level * 100))}%" if spec.needs_box else "",
                round(s["dice"], 4), round(s["nsd_3"], 4),
                f"{s['vol_pred_right']:.1f} / {s['vol_gt_right']:.1f}",
                f"{s['vol_pred_left']:.1f} / {s['vol_gt_left']:.1f}",
                round(p.seconds, 2), round(1000 * p.seconds / max(1, p.n_forward), 1),
                D.verify_weights(spec)])
            out = OUT_DIR / f"{case.name}_{spec.key}{'' if not spec.needs_box else f'_noise{int(round(level * 100))}'}.nii.gz"
            files.append(str(D.save_nifti(p.labels(), case, out)))
        z = D.best_slice(case)
        info = case_info(case) + ("\n\n" + "; ".join(notes) if notes else "")
        return (info, rows, gr.update(maximum=case.img.shape[0] - 1, value=z, interactive=True),
                D.render_grid(case, z, preds, zoom=zoom), files, (case, preds))
    except D.DemoError as e:
        raise gr.Error(str(e))


def show_slice(z, zoom, state):
    if not state:
        return None
    case, preds = state
    return D.render_grid(case, int(z), preds, zoom=zoom)


HEADERS = ["Model", "Prompt", "Box noise", "Dice", "NSD @ 3 mm", "Right kidney mL (pred / GT)",
           "Left kidney mL (pred / GT)", "Time (s)", "ms per slice", "Weights"]

with gr.Blocks(title="Kidney segmentation: SAM vs distilled students") as demo:
    gr.Markdown("## Kidney segmentation on CT and MRI: SAM (with a box) vs small U-Nets (no prompt)\n"
                "Pick a scan from our test sets, choose the models, and run. SAM arms get one box per "
                "kidney from the ground truth, with the same seeded box noise as in our evaluation; "
                "the U-Nets get only the image. Dice and NSD are the mean over the scan's kidney instances.")
    with gr.Row():
        with gr.Column(scale=1):
            source = gr.Radio(["Ready-made test case", "Upload a test scan"], value="Ready-made test case",
                              label="Input")
            sample = gr.Dropdown(list(SAMPLES), value=next(iter(SAMPLES), None), label="Test case")
            with gr.Group(visible=False) as up:
                scan = gr.File(label="Scan (.nii / .nii.gz)", file_types=[".nii", ".gz"])
                label = gr.File(label="Ground-truth label (.nii / .nii.gz)", file_types=[".nii", ".gz"])
                dataset = gr.Radio(["AMOS", "WORD"], value="AMOS", label="Dataset (fixes the kidney label numbers)")
            arms = gr.CheckboxGroup(available_arms(), value=available_arms(), label="Models")
            seed = gr.Dropdown(seeds(), value=DEFAULT_SEED if DEFAULT_SEED in seeds() else seeds()[0],
                               label="Training seed for arms 3 and 4")
            noise = gr.Radio(["0%", "5%", "10%", "20%"], value="10%",
                             label="Box noise for the boxed models (10% = our headline setting)")
            go = gr.Button("Run", variant="primary")
        with gr.Column(scale=3):
            info = gr.Markdown()
            table = gr.Dataframe(headers=HEADERS, label="Results", wrap=True, interactive=False)
            with gr.Row():
                z = gr.Slider(0, 1, value=0, step=1, label="Slice (opens on the slice with the most kidney)",
                              interactive=False, scale=4)
                zoom = gr.Checkbox(value=True, label="Zoom to the kidneys", scale=1)
            view = gr.Image(label="Fill = prediction (orange right kidney, blue left kidney); yellow outline = "
                                  "ground truth; rectangles = boxes given to the model. Image left = patient's right.",
                            type="pil")
            files = gr.File(label="Predicted masks (NIfTI, 0 background / 1 right / 2 left kidney)",
                            file_count="multiple")
    state = gr.State(None)
    source.change(lambda s: (gr.update(visible=s == "Ready-made test case"), gr.update(visible=s != "Ready-made test case")),
                  source, [sample, up])
    go.click(run, [source, sample, scan, label, dataset, arms, seed, noise, zoom], [info, table, z, view, files, state])
    z.change(show_slice, [z, zoom, state], view)
    zoom.change(show_slice, [z, zoom, state], view)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True)
