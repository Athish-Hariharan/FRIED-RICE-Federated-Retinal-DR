"""
NetworkCanvas — Federated topology visualiser with IID-ness tracking.

Dark background guaranteed (manual patch drawing, no nx.draw()).
Click any node for a detailed popup showing:
  - For clients: Accuracy, Drift, QWK Kappa, IID score (JSD),
                 DR grade label distribution mini-bar chart
  - For server:  Aggregation algorithm, client count, round info
"""

import math
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas

_BG    = "#0b0f14"
_BG2   = "#111720"
_BG3   = "#161d28"
_CYAN  = "#00d4ff"
_GREEN = "#00e676"
_ORG   = "#ffaa00"
_RED   = "#ff4444"
_TEXT  = "#e2e8f0"
_MUTED = "#8899a6"
_DIM   = "#2a3f52"
_PURP  = "#c77dff"

_GRADE_COLORS = ["#00e676", "#ffee58", "#ffa726", "#ff7043", "#f44336"]
_GRADE_NAMES  = ["No DR", "Mild", "Mod", "Sev", "PDR"]


def _acc_color(acc: float) -> str:
    if acc > 80: return _GREEN
    if acc > 70: return _ORG
    return _RED


def _iid_color(score: float) -> str:
    """score=0 → perfectly IID (green), score=1 → maximally non-IID (red)"""
    if score < 0.2:  return _GREEN
    if score < 0.45: return _ORG
    return _RED


class NetworkCanvas(Canvas):

    def __init__(self, clients: int = 3):
        matplotlib.rcParams.update({
            "figure.facecolor":  _BG,
            "axes.facecolor":    _BG,
            "text.color":        _TEXT,
            "savefig.facecolor": _BG,
        })

        self._fig = Figure(facecolor=_BG, tight_layout=True)
        self._fig.patch.set_facecolor(_BG)
        self.ax = self._fig.add_subplot(111)
        self.ax.set_facecolor(_BG)
        self.ax.axis("off")

        super().__init__(self._fig)

        self.clients = clients

        self.G = nx.Graph()
        self.G.add_node("Server")
        for i in range(clients):
            self.G.add_node(f"C{i+1}")
            self.G.add_edge("Server", f"C{i+1}")

        self.pos = nx.spring_layout(self.G, k=1.4, iterations=100, seed=42)

        # Per-client state (updated each round)
        self.client_sizes   = [1000] * clients
        self.client_acc     = [65.0] * clients
        self.client_sources = [f"Dataset-{i+1}" for i in range(clients)]
        self._drifts        = [0.5]  * clients
        self._iid_scores    = [0.0]  * clients   # JSD non-IID score [0,1]
        self._kappas        = [0.0]  * clients   # quadratic-weighted kappa
        self._label_dists   = [[200]*5 for _ in range(clients)]  # grade counts

        # Round tracking for server popup
        self._current_round = 0
        self._total_rounds  = 20

        self._node_hitboxes  = {}
        self._popup_artists  = []
        self._popup_node     = None

        self.mpl_connect("button_press_event", self._on_click)
        self._redraw()

    # ── Public ────────────────────────────────────────────────────────────────

    def update_network(self, drifts, client_sizes=None, client_acc=None,
                       sources=None, iid_scores=None, kappas=None,
                       label_dists=None, current_round=0, total_rounds=20):
        if client_sizes  is not None: self.client_sizes   = client_sizes
        if client_acc    is not None: self.client_acc     = client_acc
        if sources       is not None: self.client_sources = sources
        if iid_scores    is not None: self._iid_scores    = iid_scores
        if kappas        is not None: self._kappas        = kappas
        if label_dists   is not None: self._label_dists   = label_dists
        self._drifts        = list(drifts)
        self._current_round = current_round
        self._total_rounds  = total_rounds
        self._clear_popup()
        self._redraw()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _redraw(self):
        self.ax.cla()
        self.ax.set_facecolor(_BG)
        self._fig.patch.set_facecolor(_BG)
        self.ax.axis("off")
        self._node_hitboxes.clear()

        all_x = [v[0] for v in self.pos.values()]
        all_y = [v[1] for v in self.pos.values()]
        pad   = 0.42
        xmin, xmax = min(all_x) - pad, max(all_x) + pad
        ymin, ymax = min(all_y) - pad, max(all_y) + pad
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        xspan = xmax - xmin
        yspan = ymax - ymin
        node_r   = xspan * 0.065
        server_r = xspan * 0.085

        client_nodes = [n for n in self.G.nodes() if n != "Server"]

        # ── Edges ─────────────────────────────────────────────────────
        for i, cnode in enumerate(client_nodes):
            drift = self._drifts[i] if i < len(self._drifts) else 0.5
            iid   = self._iid_scores[i] if i < len(self._iid_scores) else 0.0
            lw    = max(0.8, min(drift * 2.0, 5.0))
            alpha = min(0.25 + drift * 0.2, 0.75)
            sx, sy = self.pos["Server"]
            cx, cy = self.pos[cnode]
            self.ax.plot([sx, cx], [sy, cy], color=_CYAN, lw=lw, alpha=alpha,
                         solid_capstyle="round", zorder=1)
            # Drift + IID badge at midpoint
            mx, my = (sx+cx)/2, (sy+cy)/2
            iid_col = _iid_color(iid)
            self.ax.text(mx, my, f"Δ{drift:.2f}  IID:{1-iid:.2f}",
                         color=_MUTED, fontsize=6.5, ha="center", va="center", zorder=3,
                         bbox=dict(boxstyle="round,pad=0.2", fc=_BG, ec=_DIM,
                                   linewidth=0.5, alpha=0.88))

        # ── Nodes ─────────────────────────────────────────────────────
        for node in self.G.nodes():
            nx_, ny_ = self.pos[node]
            is_server = (node == "Server")
            r     = server_r if is_server else node_r
            color = _CYAN if is_server else _acc_color(
                self.client_acc[int(node[1:])-1]
                if not is_server and int(node[1:])-1 < len(self.client_acc) else 65.0
            )
            # IID ring for client nodes
            if not is_server:
                idx   = int(node[1:]) - 1
                iid   = self._iid_scores[idx] if idx < len(self._iid_scores) else 0.0
                icol  = _iid_color(iid)
                ring  = plt.Circle((nx_, ny_), r*1.45, color=icol, alpha=0.15, zorder=2)
                self.ax.add_patch(ring)

            glow = plt.Circle((nx_, ny_), r*1.28, color=color, alpha=0.12, zorder=2)
            body = plt.Circle((nx_, ny_), r, fc=_BG2, ec=color, lw=2.2, zorder=4)
            self.ax.add_patch(glow)
            self.ax.add_patch(body)

            label = "SERVER\n(Global)" if is_server else node
            self.ax.text(nx_, ny_, label, color=color,
                         fontsize=7 if is_server else 9, fontweight="bold",
                         ha="center", va="center", zorder=5, multialignment="center")

            self._node_hitboxes[node] = (nx_, ny_, r)

        self.ax.set_title(
            "Federated Client Network  ·  click any node for details",
            color=_MUTED, fontsize=8, pad=4
        )
        self.draw_idle()

    # ── Click / Popup ─────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            self._clear_popup(); self.draw_idle(); return

        hit = None
        for node, (hx, hy, hr) in self._node_hitboxes.items():
            if math.hypot(event.xdata-hx, event.ydata-hy) <= hr*1.35:
                hit = node; break

        if hit is None:
            self._clear_popup(); self.draw_idle(); return

        if self._popup_artists and self._popup_node == hit:
            self._clear_popup(); self.draw_idle()
            self._popup_node = None; return

        self._popup_node = hit
        self._draw_popup(hit)

    def _clear_popup(self):
        for a in self._popup_artists:
            try: a.remove()
            except Exception: pass
        self._popup_artists.clear()

    def _draw_popup(self, node: str):
        self._clear_popup()
        hx, hy, hr = self._node_hitboxes[node]
        is_server  = (node == "Server")
        xlim = self.ax.get_xlim(); ylim = self.ax.get_ylim()
        xspan = xlim[1]-xlim[0];  yspan = ylim[1]-ylim[0]

        if is_server:
            title  = "⬡  Global Server"
            color  = _CYAN
            rows   = [
                ("Algorithm",  "FedAvg"),
                ("Clients",    str(self.clients)),
                ("Round",      f"{self._current_round} / {self._total_rounds}"),
                ("Status",     "Active"),
            ]
            extra_height = 0
        else:
            idx   = int(node[1:]) - 1
            acc   = self.client_acc[idx]   if idx < len(self.client_acc)   else 65.0
            size  = self.client_sizes[idx] if idx < len(self.client_sizes) else 1000
            src   = self.client_sources[idx] if idx < len(self.client_sources) else "—"
            drift = self._drifts[idx]      if idx < len(self._drifts)      else 0.5
            iid   = self._iid_scores[idx]  if idx < len(self._iid_scores)  else 0.0
            kappa = self._kappas[idx]      if idx < len(self._kappas)      else 0.0
            color = _acc_color(acc)
            iid_col   = _iid_color(iid)
            iid_label = "High IID" if iid < 0.2 else ("Moderate" if iid < 0.45 else "Non-IID")
            health    = "Good ✓" if acc > 80 else ("Fair ⚠" if acc > 70 else "Poor ✗")

            title = f"◎  Client {node}"
            rows  = [
                ("Dataset",   src),
                ("Samples",   f"{size:,}"),
                ("Accuracy",  f"{acc:.2f}%"),
                ("QWK Kappa", f"{kappa:.3f}"),
                ("Drift",     f"{drift:.4f}"),
                ("IID Score", f"{1-iid:.3f}  ({iid_label})"),
                ("Health",    health),
            ]
            extra_height = yspan * 0.22   # space for label distribution bar

        pw   = xspan * 0.32
        rh   = yspan * 0.072
        ph   = rh * len(rows) + yspan * 0.14 + extra_height

        gap = xspan * 0.06
        px  = hx + hr + gap
        if px + pw > xlim[1] - xspan*0.01:
            px = hx - hr - gap - pw
        py  = max(ylim[0]+ph/2+yspan*0.01, min(hy, ylim[1]-ph/2-yspan*0.01))

        # Box
        box = FancyBboxPatch((px, py-ph/2), pw, ph, boxstyle="round,pad=0.01",
                             fc=_BG3, ec=color, lw=1.8, zorder=10, alpha=0.97)
        self.ax.add_patch(box); self._popup_artists.append(box)

        # Connector
        ln, = self.ax.plot([hx+hr*0.9, px], [hy, py], color=color,
                           lw=0.9, ls="--", alpha=0.55, zorder=9)
        self._popup_artists.append(ln)

        # Title bar
        th = yspan*0.055
        tb = FancyBboxPatch((px, py+ph/2-th), pw, th, boxstyle="round,pad=0.01",
                            fc=color, ec=color, lw=0, zorder=11, alpha=0.18)
        self.ax.add_patch(tb); self._popup_artists.append(tb)
        t = self.ax.text(px+pw/2, py+ph/2-th/2, title, color=color,
                         fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=12)
        self._popup_artists.append(t)

        # Divider
        dy = py+ph/2-th
        dl, = self.ax.plot([px+pw*0.04, px+pw*0.96], [dy, dy],
                           color=color, lw=0.5, alpha=0.4, zorder=11)
        self._popup_artists.append(dl)

        # Rows
        rs = dy - rh*0.6
        for i, (key, val) in enumerate(rows):
            ry = rs - i*rh
            # IID Score row gets colored value
            val_color = _TEXT
            if key == "IID Score":
                val_color = _iid_color(self._iid_scores[idx] if not is_server else 0)
            if key == "QWK Kappa":
                val_color = _GREEN if kappa > 0.7 else _ORG if kappa > 0.5 else _RED

            if i % 2 == 0:
                rb = FancyBboxPatch((px+pw*0.02, ry-rh*0.42), pw*0.96, rh*0.84,
                                   boxstyle="round,pad=0.005", fc=_BG2, ec="none",
                                   lw=0, zorder=10, alpha=0.5)
                self.ax.add_patch(rb); self._popup_artists.append(rb)

            kt = self.ax.text(px+pw*0.08, ry, key, color=_MUTED,
                              fontsize=7.5, ha="left", va="center", zorder=12)
            vt = self.ax.text(px+pw*0.93, ry, val, color=val_color,
                              fontsize=7.5, fontweight="bold", ha="right", va="center", zorder=12)
            self._popup_artists += [kt, vt]

        # ── DR label distribution mini-bar (client nodes only) ────────
        if not is_server and idx < len(self._label_dists):
            dist  = self._label_dists[idx]
            total = sum(dist) or 1
            bar_y = py - ph/2 + yspan*0.045
            bar_x = px + pw*0.05
            bar_w = pw*0.90
            bar_h = yspan*0.028

            # Background
            bg = FancyBboxPatch((bar_x, bar_y), bar_w, bar_h*2.8,
                                boxstyle="round,pad=0.01", fc=_BG2, ec=_DIM,
                                lw=0.5, zorder=11, alpha=0.7)
            self.ax.add_patch(bg); self._popup_artists.append(bg)

            # Title
            lt = self.ax.text(bar_x + bar_w/2, bar_y + bar_h*2.8 + yspan*0.008,
                              "DR Grade Distribution", color=_MUTED,
                              fontsize=6.5, ha="center", va="bottom", zorder=12)
            self._popup_artists.append(lt)

            # Stacked horizontal bar
            x_cursor = bar_x + pw*0.01
            available = bar_w - pw*0.02
            for g, (count, gcol) in enumerate(zip(dist, _GRADE_COLORS)):
                seg_w = (count / total) * available
                if seg_w < 0.001: continue
                seg = FancyBboxPatch((x_cursor, bar_y + bar_h*0.5), seg_w, bar_h,
                                    boxstyle="square,pad=0", fc=gcol, ec="none",
                                    lw=0, zorder=12, alpha=0.85)
                self.ax.add_patch(seg); self._popup_artists.append(seg)
                if seg_w > available*0.08:
                    pct_t = self.ax.text(x_cursor + seg_w/2, bar_y + bar_h,
                                         f"{count/total*100:.0f}%",
                                         color="black", fontsize=5.5,
                                         ha="center", va="center", zorder=13)
                    self._popup_artists.append(pct_t)
                x_cursor += seg_w

            # Legend
            leg_y = bar_y + 0.005
            for g, (gcol, gname) in enumerate(zip(_GRADE_COLORS, _GRADE_NAMES)):
                lx = bar_x + g*(bar_w/5)
                dot = FancyBboxPatch((lx, leg_y), bar_w/5*0.18, bar_h*0.55,
                                    boxstyle="square,pad=0", fc=gcol, ec="none",
                                    lw=0, zorder=12, alpha=0.85)
                self.ax.add_patch(dot); self._popup_artists.append(dot)
                lt2 = self.ax.text(lx + bar_w/5*0.22, leg_y + bar_h*0.28, gname,
                                   color=_MUTED, fontsize=5, ha="left", va="center", zorder=12)
                self._popup_artists.append(lt2)

        self.draw_idle()
