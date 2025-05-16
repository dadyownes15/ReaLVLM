import os
import math
from typing import Sequence, Union, Callable
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt


class Results:
    def __init__(
        self,
        score_path: str,
        normal_videos: Sequence[str],
        *,
        window_sizes: Sequence[int] = (2, 4, 8, 16),
        show_normal: bool = True,
        show_other: bool = True,
        show_detailed: bool = False,
        plot: bool = False,
        top_n_anomaly: int = 0,
    ) -> None:
        self.score_path = score_path
        self.normal_videos = set(normal_videos)
        self.window_sizes = tuple(window_sizes)
        self.show_normal = show_normal
        self.show_other = show_other
        self.show_detailed = show_detailed
        self.plot = plot
        self.top_n_anomaly = top_n_anomaly

        self.stats = {"normal": defaultdict(list), "other": defaultdict(list)}
        self.rolling_means = {w: [] for w in self.window_sizes}
        self.video_scores: dict[str, torch.Tensor] = {}

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------

    def _collect(self):
        videos = []
        for name in os.listdir(self.score_path):
            path = os.path.join(self.score_path, name, "scores.pt")
            if os.path.exists(path):
                videos.append((name, torch.load(path)))
        videos.sort(key=lambda x: x[0])
        return videos

    def _update(self, bucket: str, name: str, scores: torch.Tensor):
        store = self.stats[bucket]
        # keep raw scores for later analysis
        self.video_scores[name] = scores

        if scores.numel() == 0:
            for key in ["max", "min", "mean", *[f"r{w}" for w in self.window_sizes]]:
                store[key].append(float("nan"))
            for w in self.window_sizes:
                self.rolling_means[w].append((name, float("nan")))
            return

        store["max"].append(torch.max(scores).item())
        store["min"].append(torch.min(scores).item())
        store["mean"].append(torch.mean(scores.float()).item())
        for w in self.window_sizes:
            if scores.numel() >= w:
                m = scores.float().unfold(0, w, 1).mean(1).mean().item()
            else:
                m = float("nan")
            store[f"r{w}"].append(m)
            self.rolling_means[w].append((name, m))
        if self.show_detailed:
            print(
                name,
                store["max"][-1],
                store["min"][-1],
                store["mean"][-1],
                [store[f"r{w}"][-1] for w in self.window_sizes],
            )

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def load(self):
        for n, s in self._collect():
            bucket = "normal" if n in self.normal_videos else "other"
            if (bucket == "normal" and self.show_normal) or (
                bucket == "other" and self.show_other
            ):
                self._update(bucket, n, s)
        if self.top_n_anomaly:
            self._top_anomalies()
        if self.plot:
            self._plot()

    # -----------------------------------------------------------
    # evaluation helpers
    # -----------------------------------------------------------

    def results_summary(self, detection_rule: Callable[[torch.Tensor], bool]) -> dict:
        if not self.video_scores:
            # ensure data is loaded
            self.load()

        tp = fp = tn = fn = 0
        for name, scores in self.video_scores.items():
            pred_anom = bool(detection_rule(scores))
            true_anom = name not in self.normal_videos
            if pred_anom and true_anom:
                tp += 1
            elif pred_anom and not true_anom:
                fp += 1
            elif not pred_anom and true_anom:
                fn += 1
            else:
                tn += 1

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else float("nan")
        )

        print(f"Total videos:      {total}")
        print(f"True positives:    {tp}")
        print(f"False positives:   {fp}")
        print(f"True negatives:    {tn}")
        print(f"False negatives:   {fn}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 score:  {f1:.4f}")
        return dict(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    def results_statistics(self) -> None:
        if not self.stats["normal"] and not self.stats["other"]:
            self.load()

        metrics = set()
        for bucket in ("normal", "other"):
            metrics.update(self.stats[bucket].keys())

        for metric in sorted(metrics):
            normal_vals = [v for v in self.stats["normal"].get(metric, []) if not np.isnan(v)]
            other_vals = [v for v in self.stats["other"].get(metric, []) if not np.isnan(v)]

            def _describe(arr):
                if not arr:
                    return "no data"
                return (f"mean={np.mean(arr):.4f}, std={np.std(arr):.4f}, "
                        f"min={np.min(arr):.4f}, max={np.max(arr):.4f}, n={len(arr)}")

            print(f"{metric} -> normal: {_describe(normal_vals)} | other: {_describe(other_vals)}")

        print("\nFinished summarising statistics.")

    # ------------------------------------------------------------------
    # deep‑dive
    # ------------------------------------------------------------------

    def deepdive(
        self,
        video_ids: Union[str, Sequence[str]],
        *,
        stride: int = 1,
        fps: int = 30,
        frames_per_clip: int = 16,
    ) -> None:
        if isinstance(video_ids, str):
            video_ids = [video_ids]

        fig, axs = plt.subplots(
            len(video_ids), 1, figsize=(10, 3 * len(video_ids)), squeeze=False
        )
        axs = axs.flatten()
        for ax, vid in zip(axs, video_ids):
            path = os.path.join(self.score_path, vid, "scores.pt")
            if not os.path.exists(path):
                ax.set_title(f"{vid} – no data")
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                continue
            scores = torch.load(path).float().cpu().numpy()
            t = ((np.arange(scores.size) * stride) + frames_per_clip / 2) / fps
            ax.plot(t, scores)
            ax.set_title(vid)
            ax.set_xlabel("seconds")
            ax.set_ylabel("anomaly score")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # private visual helpers
    # ------------------------------------------------------------------

    def _top_anomalies(self):
        for w in self.window_sizes:
            vals = [(n, v) for n, v in self.rolling_means[w] if not np.isnan(v)]
            if not vals:
                continue
            vals.sort(key=lambda x: x[1], reverse=True)
            print(f"Top {min(self.top_n_anomaly, len(vals))} by r{w}:")
            for i, (n, v) in enumerate(vals[: self.top_n_anomaly], 1):
                print(i, n, f"{v:.4f}")

    def _plot(self):
        metrics = ["mean", "max", "min", *[f"r{w}" for w in self.window_sizes]]
        cols = 4
        rows = math.ceil(len(metrics) / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axs = axs.flatten()
        for i, m in enumerate(metrics):
            n_vals = [x for x in self.stats["normal"].get(m, []) if not np.isnan(x)]
            o_vals = [x for x in self.stats["other"].get(m, []) if not np.isnan(x)]
            if not n_vals and not o_vals:
                axs[i].set_title(m)
                axs[i].text(0.5, 0.5, "no data", ha="center", va="center")
                continue
            axs[i].hist(n_vals, alpha=0.5, bins=10, label="normal")
            axs[i].hist(o_vals, alpha=0.5, bins=10, label="other")
            axs[i].set_title(m)
            axs[i].legend()
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])
        plt.tight_layout()
        plt.show()
