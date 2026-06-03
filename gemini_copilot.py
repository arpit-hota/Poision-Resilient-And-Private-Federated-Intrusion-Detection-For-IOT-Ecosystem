"""
gemini_copilot.py — Security Co-Pilot powered by Gemini AI

DESIGN PRINCIPLE: Completely isolated from training.
- Runs in a background daemon thread — training never waits for it
- If API key missing, SDK not installed, or API call fails → silent skip
- No import of this file affects training in any way
- Thread is daemon=True so it dies automatically if main process exits

Setup:
    pip install google-generativeai
    set GEMINI_API_KEY=your_key_here   (Windows CMD)

    Get a free key at: https://aistudio.google.com/app/apikey

Usage (called from simulation.py evaluate() every round):
    from gemini_copilot import GeminiCoPilot
    copilot = GeminiCoPilot()
    copilot.analyse_round_async(round_num, accuracy, loss, metrics_dict)
    # returns immediately — analysis runs in background
"""

import os
import json
import datetime
import threading


class GeminiCoPilot:
    """
    Security Co-Pilot that analyses per-round FL-IDS evaluation results
    using Gemini AI and prints a brief analysis to the terminal.

    Every call to analyse_round_async() launches a daemon background thread.
    The calling code (training loop) returns IMMEDIATELY and is never blocked.

    Thread safety: each round gets its own independent thread and API call.
    No shared mutable state between threads.
    """

    def __init__(
        self,
        api_key:    str = None,
        model_name: str = "gemini-2.0-flash",
        report_dir: str = "gemini_reports",
    ):
        self.model_name = model_name
        self.report_dir = report_dir
        self.enabled    = False
        self._model     = None

        os.makedirs(report_dir, exist_ok=True)

        # Resolve API key
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()

        if not self.api_key:
            print(
                "\n  [Gemini Co-Pilot] ⚠️  No API key — Co-Pilot disabled.\n"
                "  Training runs normally. To enable:\n"
                "    Windows CMD: set GEMINI_API_KEY=your_key_here\n"
                "  Get a free key: https://aistudio.google.com/app/apikey\n"
            )
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model  = genai.GenerativeModel(model_name)
            self.enabled = True
            print(f"  [Gemini Co-Pilot] ✅ Ready — model: {model_name}")
            print(f"  [Gemini Co-Pilot] Analysis will appear in background each round.\n")
        except ImportError:
            print(
                "\n  [Gemini Co-Pilot] ⚠️  google-generativeai not installed.\n"
                "  Run: pip install google-generativeai\n"
                "  Training runs normally.\n"
            )
        except Exception as e:
            print(f"\n  [Gemini Co-Pilot] ⚠️  Init failed: {e} — disabled.\n")

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyse_round_async(
        self,
        round_num:   int,
        accuracy:    float,
        loss:        float,
        class_names: list  = None,
        all_preds:   list  = None,
        all_targets: list  = None,
        dp_epsilon:  float = 5.0,
    ):
        """
        Launches a background thread to analyse this round's results.
        Returns IMMEDIATELY — caller (training loop) is never blocked.

        Args:
            round_num:   current FL round number
            accuracy:    global model accuracy this round
            loss:        global model loss this round
            class_names: list of class name strings
            all_preds:   list of predicted class indices (from evaluate)
            all_targets: list of true class indices (from evaluate)
            dp_epsilon:  DP epsilon value in use
        """
        if not self.enabled:
            return  # silently skip — training not affected

        # Build per-class metrics snapshot for this round
        metrics_dict = self._build_metrics(
            all_preds, all_targets, class_names
        )

        # Snapshot all args — thread gets its own copy, no shared state
        thread = threading.Thread(
            target=self._run_analysis,
            args=(round_num, accuracy, loss, metrics_dict, dp_epsilon),
            daemon=True,   # dies automatically when main process exits
            name=f"gemini-round-{round_num}",
        )
        thread.start()
        # Return immediately — training continues

    # ── Background thread work ─────────────────────────────────────────────────

    def _run_analysis(self, round_num, accuracy, loss, metrics_dict, dp_epsilon):
        """Runs in background thread. All exceptions caught — never propagates."""
        try:
            prompt   = self._build_prompt(round_num, accuracy, loss,
                                          metrics_dict, dp_epsilon)
            response = self._model.generate_content(prompt)
            text     = response.text

            # Print to terminal (thread-safe in CPython for print)
            self._print_analysis(round_num, accuracy, text)

            # Save to disk
            self._save(round_num, accuracy, loss, metrics_dict, text)

        except Exception as e:
            # Silently swallow ALL errors — training must not be affected
            try:
                print(f"\n  [Gemini Co-Pilot] Round {round_num}: skipped ({e})\n")
            except Exception:
                pass

    # ── Prompt ────────────────────────────────────────────────────────────────

    def _build_prompt(self, round_num, accuracy, loss, metrics_dict, dp_epsilon):
        SKIP  = {"accuracy", "macro avg", "weighted avg"}
        lines = []
        for cls, s in metrics_dict.items():
            if cls in SKIP:
                continue
            p   = s.get("precision", 0)
            r   = s.get("recall",    0)
            f1  = s.get("f1-score",  0)
            sup = s.get("support",   0)
            flag = "✅" if f1 >= 0.90 else ("⚠️" if f1 >= 0.70 else "🔴")
            lines.append(
                f"  {flag} {cls}: P={p:.3f} R={r:.3f} F1={f1:.3f} n={sup}"
            )
        metrics_text = "\n".join(lines) or "  No per-class data."

        return f"""You are a network security AI reviewing a Federated IDS training round.

Round {round_num}/50 | Accuracy: {accuracy*100:.2f}% | Loss: {loss:.4f}
DP ε={dp_epsilon} | Defense: Multi-Krum

Per-class results:
{metrics_text}

Give a SHORT round analysis (max 6 lines total):
1. One sentence on overall model health this round.
2. Flag any class with F1 < 0.80 and why it matters.
3. One specific recommended action if any class needs attention.
4. One line on whether the FL system appears stable.

Be direct and concise. No preamble."""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_metrics(self, all_preds, all_targets, class_names):
        """Builds per-class metrics dict from predictions. Returns {} on failure."""
        try:
            if all_preds is None or all_targets is None or len(all_preds) == 0:
                return {}
            from sklearn.metrics import classification_report
            import numpy as np
            return classification_report(
                np.array(all_targets),
                np.array(all_preds),
                target_names=class_names,
                zero_division=0,
                output_dict=True,
            )
        except Exception:
            return {}

    def _print_analysis(self, round_num, accuracy, text):
        """Prints a clearly labelled Gemini analysis block to the terminal."""
        print(f"\n{'─'*60}")
        print(f"  🤖 GEMINI CO-PILOT | Round {round_num} | Acc: {accuracy*100:.2f}%")
        print(f"{'─'*60}")
        # Print up to 500 chars so terminal doesn't flood
        preview = text.strip()[:500]
        if len(text.strip()) > 500:
            preview += "\n  [... full analysis saved to gemini_reports/]"
        print(preview)
        print(f"{'─'*60}\n")

    def _save(self, round_num, accuracy, loss, metrics_dict, text):
        """Saves full analysis to disk. Silent on failure."""
        try:
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(
                self.report_dir, f"round_{round_num:02d}_{ts}.txt"
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"Round {round_num} | Accuracy: {accuracy*100:.2f}% | Loss: {loss:.4f}\n")
                f.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
                f.write("─" * 60 + "\n\n")
                f.write(text)
                f.write("\n\n" + "─" * 60 + "\n")
                if metrics_dict:
                    f.write("\nFull metrics:\n")
                    f.write(json.dumps(metrics_dict, indent=2))
        except Exception:
            pass  # saving failure never affects anything
