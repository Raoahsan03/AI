"""
gui.py — Tkinter-based Graphical User Interface for the Smart Agriculture DSS.

Layout:
  ┌─ Header ───────────────────────────────────────────────────┐
  │ Input Panel (left) │ Results Cards (right-top)             │
  │                    │ Visualization Notebook (right-bottom) │
  └────────────────────────────────────────────────────────────┘
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Resolve project root so the package can be imported from any cwd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models import load_models
from src.utils import FEATURE_RANGES, FEATURE_COLS, get_cluster_info, compute_yield_bounds

MODELS_DIR  = os.path.join(_ROOT, 'models')
RESULTS_DIR = os.path.join(_ROOT, 'results')

# ── Colour palette ───────────────────────────────────────────
C = {
    'bg':       '#0d1117',
    'surface':  '#161b22',
    'card':     '#1c2128',
    'accent':   '#1f6feb',
    'primary':  '#e94560',
    'success':  '#3fb950',
    'warning':  '#d29922',
    'text':     '#e6edf3',
    'muted':    '#8b949e',
    'border':   '#30363d',
}

FONT_H1   = ('Segoe UI', 15, 'bold')
FONT_H2   = ('Segoe UI', 11, 'bold')
FONT_H3   = ('Segoe UI', 10, 'bold')
FONT_BODY = ('Segoe UI', 9)
FONT_MONO = ('Consolas', 10)


class SmartAgriApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('Smart Agriculture Decision Support System')
        self.root.geometry('1440x860')
        self.root.configure(bg=C['bg'])
        self.root.minsize(1200, 720)

        self.models: dict = {}
        self.feature_vars: dict[str, tk.DoubleVar] = {}

        self._load_models()
        self._build_ui()
        self.root.after(300, self._load_static_plots)

    # ── Model loading ────────────────────────────────────────

    def _load_models(self):
        self.models = load_models(MODELS_DIR)
        if not self.models:
            messagebox.showwarning(
                'Models Missing',
                'Serialized models not found.\n'
                'Please run  python train_models.py  first, then reopen the application.',
            )

    # ── UI construction ──────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        body = tk.Frame(self.root, bg=C['bg'])
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = tk.Frame(body, bg=C['surface'], width=295, bd=1, relief=tk.FLAT)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.pack_propagate(False)
        self._build_input_panel(left)

        right = tk.Frame(body, bg=C['bg'])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        results_frame = tk.Frame(right, bg=C['surface'], height=185, bd=1, relief=tk.FLAT)
        results_frame.pack(fill=tk.X, pady=(0, 6))
        results_frame.pack_propagate(False)
        self._build_results_panel(results_frame)

        viz_frame = tk.Frame(right, bg=C['surface'], bd=1, relief=tk.FLAT)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        self._build_viz_panel(viz_frame)

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C['accent'], height=52)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(
            hdr,
            text='  Smart Agriculture Decision Support System',
            font=FONT_H1, fg='white', bg=C['accent'],
        ).pack(side=tk.LEFT, padx=10)
        tk.Label(
            hdr,
            text='Decision Tree  ·  KMeans Clustering  ·  Linear Regression  ',
            font=FONT_BODY, fg='#cdd9e5', bg=C['accent'],
        ).pack(side=tk.RIGHT, padx=10)

    # ── Input panel ─────────────────────────────────────────

    def _build_input_panel(self, parent):
        tk.Label(parent, text='Soil & Climate Parameters',
                 font=FONT_H2, fg=C['primary'], bg=C['surface']).pack(pady=(12, 4), padx=12, anchor='w')
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, padx=12, pady=4)

        scroll_canvas = tk.Canvas(parent, bg=C['surface'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(scroll_canvas, bg=C['surface'])
        window_id = scroll_canvas.create_window((0, 0), window=inner, anchor='nw')

        def _on_resize(event):
            scroll_canvas.itemconfig(window_id, width=event.width)
        scroll_canvas.bind('<Configure>', _on_resize)
        inner.bind('<Configure>', lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox('all')))

        self._val_labels: dict[str, tk.Label] = {}

        for feat in FEATURE_COLS:
            info = FEATURE_RANGES[feat]
            var  = tk.DoubleVar(value=info['default'])
            self.feature_vars[feat] = var

            frame = tk.Frame(inner, bg=C['surface'])
            frame.pack(fill=tk.X, padx=12, pady=4)

            lbl_text = f"{info['label']}  ({info['unit']})" if info['unit'] else info['label']
            tk.Label(frame, text=lbl_text, font=FONT_BODY, fg=C['muted'], bg=C['surface'],
                     anchor='w').pack(fill=tk.X)

            row = tk.Frame(frame, bg=C['surface'])
            row.pack(fill=tk.X)

            slider = tk.Scale(
                row, variable=var,
                from_=info['min'], to=info['max'],
                resolution=0.1, orient=tk.HORIZONTAL,
                bg=C['surface'], fg=C['text'],
                troughcolor=C['border'], activebackground=C['accent'],
                highlightthickness=0, showvalue=False, length=185,
                command=lambda v, f=feat: self._sync_label(f),
            )
            slider.pack(side=tk.LEFT)

            val_lbl = tk.Label(row, text=f"{info['default']:.1f}",
                               font=FONT_MONO, fg=C['success'], bg=C['surface'], width=7)
            val_lbl.pack(side=tk.LEFT, padx=4)
            self._val_labels[feat] = val_lbl

        ttk.Separator(inner, orient='horizontal').pack(fill=tk.X, padx=12, pady=8)

        btn_frame = tk.Frame(inner, bg=C['surface'])
        btn_frame.pack(fill=tk.X, padx=12, pady=4)

        tk.Button(
            btn_frame, text='ANALYZE FIELD',
            font=FONT_H3, fg='white', bg=C['primary'],
            activebackground='#b0304a', relief=tk.FLAT,
            padx=12, pady=8, cursor='hand2',
            command=self._run_analysis,
        ).pack(fill=tk.X, pady=(0, 4))

        tk.Button(
            btn_frame, text='Reset to Defaults',
            font=FONT_BODY, fg=C['muted'], bg=C['card'],
            activebackground=C['border'], relief=tk.FLAT,
            padx=8, pady=5, cursor='hand2',
            command=self._reset_defaults,
        ).pack(fill=tk.X)

        self._status_var = tk.StringVar(value='Ready — enter parameters and click Analyze.')
        tk.Label(inner, textvariable=self._status_var, font=('Segoe UI', 8),
                 fg=C['muted'], bg=C['surface'], wraplength=255,
                 justify=tk.LEFT).pack(padx=12, pady=6, anchor='w')

    def _sync_label(self, feat):
        lbl = self._val_labels.get(feat)
        if lbl:
            lbl.config(text=f"{self.feature_vars[feat].get():.1f}")

    def _reset_defaults(self):
        for feat, var in self.feature_vars.items():
            var.set(FEATURE_RANGES[feat]['default'])
            self._sync_label(feat)

    # ── Results panel ────────────────────────────────────────

    def _build_results_panel(self, parent):
        tk.Label(parent, text='Analysis Results', font=FONT_H2,
                 fg=C['primary'], bg=C['surface']).pack(anchor='w', padx=14, pady=(8, 4))

        cards_row = tk.Frame(parent, bg=C['surface'])
        cards_row.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        def _card(parent, header, value_var, sub_var, value_color):
            f = tk.Frame(parent, bg=C['card'], padx=12, pady=8)
            f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            tk.Label(f, text=header, font=FONT_BODY, fg=C['muted'], bg=C['card'],
                     anchor='w').pack(fill=tk.X)
            tk.Label(f, textvariable=value_var, font=('Segoe UI', 17, 'bold'),
                     fg=value_color, bg=C['card'], anchor='w').pack(fill=tk.X)
            tk.Label(f, textvariable=sub_var, font=('Segoe UI', 8),
                     fg=C['muted'], bg=C['card'], anchor='w', wraplength=220,
                     justify=tk.LEFT).pack(fill=tk.X)

        self._crop_var      = tk.StringVar(value='—')
        self._crop_sub      = tk.StringVar(value='Run analysis to see recommendation')
        self._zone_var      = tk.StringVar(value='—')
        self._zone_sub      = tk.StringVar(value='')
        self._yield_var     = tk.StringVar(value='—')
        self._yield_sub     = tk.StringVar(value='')
        self._guidance_var  = tk.StringVar(value='—')
        self._guidance_sub  = tk.StringVar(value='')

        _card(cards_row, 'Recommended Crop',       self._crop_var,     self._crop_sub,     C['success'])
        _card(cards_row, 'Soil Zone',               self._zone_var,     self._zone_sub,     C['warning'])
        _card(cards_row, 'Predicted Yield',         self._yield_var,    self._yield_sub,    C['primary'])
        _card(cards_row, 'Agronomic Action',        self._guidance_var, self._guidance_sub, C['text'])

    # ── Visualization panel ──────────────────────────────────

    def _build_viz_panel(self, parent):
        tk.Label(parent, text='Model Visualizations', font=FONT_H2,
                 fg=C['primary'], bg=C['surface']).pack(anchor='w', padx=14, pady=(8, 2))

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TNotebook', background=C['surface'], borderwidth=0)
        style.configure('Dark.TNotebook.Tab', background=C['card'], foreground=C['muted'],
                        padding=[12, 5], font=FONT_BODY)
        style.map('Dark.TNotebook.Tab',
                  background=[('selected', C['accent'])],
                  foreground=[('selected', 'white')])

        nb = ttk.Notebook(parent, style='Dark.TNotebook')
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 8))

        def _make_tab(title):
            tab = tk.Frame(nb, bg=C['surface'])
            nb.add(tab, text=f'  {title}  ')
            fig = Figure(facecolor=C['surface'])
            cvs = FigureCanvasTkAgg(fig, master=tab)
            cvs.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            return fig, cvs

        self._fig1, self._cvs1 = _make_tab('Feature Importance')
        self._fig2, self._cvs2 = _make_tab('Soil Cluster Distribution')
        self._fig3, self._cvs3 = _make_tab('Residual Analysis')

    def _load_static_plots(self):
        specs = [
            (self._fig1, self._cvs1, 'feature_importance.png'),
            (self._fig2, self._cvs2, 'cluster_scatter.png'),
            (self._fig3, self._cvs3, 'residual_plot.png'),
        ]
        for fig, cvs, fname in specs:
            fig.clear()
            ax = fig.add_subplot(111)
            path = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(path):
                img = mpimg.imread(path)
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.set_facecolor(C['surface'])
                ax.text(0.5, 0.5,
                        f'Plot not found: {fname}\nRun train_models.py first.',
                        ha='center', va='center', color=C['muted'], fontsize=11,
                        transform=ax.transAxes)
                ax.axis('off')
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            cvs.draw()

    # ── Inference ────────────────────────────────────────────

    def _run_analysis(self):
        if not self.models:
            messagebox.showerror('Models not loaded',
                                 'Please run  python train_models.py  and restart the app.')
            return

        self._status_var.set('Analyzing…')
        self.root.update_idletasks()

        try:
            raw = np.array([[self.feature_vars[f].get() for f in FEATURE_COLS]])

            scaler = self.models.get('scaler')
            if scaler is None:
                raise RuntimeError('Scaler artifact missing.')
            X = scaler.transform(raw)

            # Decision Tree
            dt = self.models.get('decision_tree')
            le = self.models.get('label_encoder')
            if dt and le:
                crop_idx   = dt.predict(X)[0]
                crop_name  = le.inverse_transform([crop_idx])[0].capitalize()
                confidence = dt.predict_proba(X)[0].max() * 100
            else:
                crop_name, confidence = 'N/A', 0.0

            # KMeans
            km = self.models.get('kmeans')
            if km:
                zone_id   = int(km.predict(X)[0])
                zone_info = get_cluster_info(zone_id)
            else:
                zone_id   = 0
                zone_info = {}

            # Linear Regression
            # One-hot encode the crop prediction and append to soil/climate features
            # so the model knows which crop's yield range to predict.
            reg     = self.models.get('linear_regression')
            ohe     = self.models.get('crop_ohe')
            std_res = self.models.get('reg_std_residual')
            if ohe is not None:
                crop_ohe = ohe.transform(np.array([[crop_idx]]))
                X_reg    = np.hstack([X, crop_ohe])
            else:
                X_reg = X

            if reg:
                y_mid = float(reg.predict(X_reg)[0])
                if std_res:
                    # 95% prediction interval using residual standard deviation
                    margin = 1.96 * float(std_res)
                    y_lo   = max(0.0, y_mid - margin)
                    y_hi   = y_mid + margin
                else:
                    y_lo, _, y_hi = compute_yield_bounds(y_mid)
            else:
                y_mid = y_lo = y_hi = 0.0

            # Update result cards
            self._crop_var.set(crop_name)
            self._crop_sub.set(f'Confidence: {confidence:.1f}%')

            self._zone_var.set(zone_info.get('name', f'Zone {zone_id}'))
            self._zone_sub.set(zone_info.get('description', ''))

            self._yield_var.set(f"{y_mid:,.0f} kg/ha")
            self._yield_sub.set(f"95% interval: {y_lo:,.0f} - {y_hi:,.0f} kg/ha")

            self._guidance_var.set('')
            self._guidance_sub.set(zone_info.get('action', '—'))

            self._status_var.set(
                f'Done — Crop: {crop_name}  |  Zone: {zone_info.get("name", "")}  |  '
                f'Yield: {y_mid:,.0f} kg/ha'
            )

        except Exception as exc:
            messagebox.showerror('Analysis Error', str(exc))
            self._status_var.set(f'Error: {exc}')


# ── Entry point ──────────────────────────────────────────────

def main():
    root = tk.Tk()
    SmartAgriApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
