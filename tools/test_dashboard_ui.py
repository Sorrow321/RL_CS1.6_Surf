"""Selenium suite for the dashboard record buttons.

Every check here exists because something actually broke in the field:

  * ``poll()`` re-renders the run view every 4 s. The old code kept the
    in-flight recording state ON the clicked button, so the first refresh
    destroyed it: the job ran to completion invisibly and a pristine,
    enabled button sat on screen. Reported, correctly, as "the button does
    nothing". Tests 13-14, 22.
  * A config-mirroring guard false-positived and failed EVERY recording
    with rc=1. Tests 15, 23.
  * ``--ep-ticks`` was silently overridden to the ckpt's 12000, so a
    "quick" 2x2000 preview really ran 24,000 single-env ticks (~100 s) -
    and got slower the better the agent became. Tests 17, 18-21.
  * Dashboards were once restarted on a port the tunnel did not forward.
    Tests 01-03.

Usage::

    python tools/test_dashboard_ui.py --url http://localhost:8082/viewer/runs.html
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

NL = chr(10)


def A(x):
    """The terminal here is cp1251; the UI uses record/check glyphs."""
    return str(x).encode("ascii", "replace").decode("ascii")


class Suite:
    def __init__(self, url, timeout=180):
        self.url, self.timeout = url, timeout
        self.results = []
        o = Options()
        o.add_argument("--headless=new")
        o.add_argument("--no-sandbox")
        o.add_argument("--disable-dev-shm-usage")
        o.add_argument("--window-size=1600,1200")
        o.set_capability("goog:loggingPrefs", {"browser": "ALL"})
        self.d = webdriver.Chrome(options=o)
        self.d.set_page_load_timeout(60)

    # ---------- plumbing ----------
    def check(self, name, cond, detail=""):
        ok = bool(cond)
        self.results.append((name, ok, detail))
        print(("  PASS  " if ok else "  FAIL  ") + name +
              (("  | " + A(detail)) if detail else ""))
        return ok

    def api(self, path):
        base = self.url.rsplit("/viewer/", 1)[0]
        with urllib.request.urlopen(base + path, timeout=20) as r:
            return json.loads(r.read().decode())

    def open(self):
        self.d.get(self.url)
        time.sleep(4)

    def cards(self):
        return self.d.find_elements(By.CSS_SELECTOR, ".runcard")

    def select_run(self, name):
        for c in self.cards():
            if name in c.text:
                self.d.execute_script("arguments[0].click();", c)
                time.sleep(3)
                return True
        return False

    def buttons(self):
        return self.d.find_elements(By.CSS_SELECTOR, "button.rec")

    def button(self, label):
        for b in self.buttons():
            if label in (b.get_attribute("data-label") or "") or label in b.text:
                return b
        return None

    def n_arts(self):
        return len(self.d.find_elements(By.CSS_SELECTOR, "#artifacts .art"))

    def severe(self):
        return [e for e in self.d.get_log("browser")
                if e["level"] == "SEVERE" and "favicon" not in e["message"]]

    def live_run(self):
        for c in self.cards():
            if "live" in c.text:
                return c.text.split("live")[0].strip()
        cs = self.cards()
        return cs[0].text.split(NL)[0].strip() if cs else None

    # ---------- the suite ----------
    def run(self):
        print("=== " + self.url + " ===")
        self.open()

        self.check("01 page loads", "runs" in (self.d.title or "").lower()
                   or bool(self.cards()))
        self.check("02 sidebar lists runs", len(self.cards()) > 0,
                   str(len(self.cards())) + " cards")
        try:
            runs = self.api("/api/runs")
            self.check("03 /api/runs responds", "runs" in runs,
                       str(len(runs.get("runs", []))) + " runs")
        except Exception as e:
            self.check("03 /api/runs responds", False, e)
        e0 = self.severe()
        self.check("04 no severe console errors on load", not e0,
                   e0[0]["message"][:120] if e0 else "")

        run = self.live_run()
        if not self.check("05 a run is present", run is not None, A(run)):
            return self.finish()
        self.check("06 run is selectable", self.select_run(run))

        btns = self.buttons()
        self.check("07 record buttons render", len(btns) >= 2, str(len(btns)))
        self.check("08 every button carries data-key",
                   all(b.get_attribute("data-key") for b in btns))
        self.check("09 every button carries data-label",
                   all(b.get_attribute("data-label") for b in btns))
        keys = [b.get_attribute("data-key") for b in btns]
        self.check("10 data-keys unique", len(keys) == len(set(keys)), A(keys))
        self.check("11 buttons start enabled", all(b.is_enabled() for b in btns))
        self.check("12 a frontier or greedy button exists",
                   self.button("frontier") is not None
                   or self.button("greedy") is not None)

        b = self.button("frontier") or self.button("greedy")
        label = b.get_attribute("data-label")
        arts0 = self.n_arts()
        self.d.execute_script("arguments[0].click();", b)
        time.sleep(2)

        b2 = self.button(label)
        self.check("13 button enters recording state",
                   b2 is not None and not b2.is_enabled(),
                   A(b2.text if b2 else "button vanished"))

        texts, pcts = set(), []
        err_txt, done_art = "", False
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            time.sleep(2)
            cur = self.button(label)
            if cur is None:
                break
            t = A(cur.text.strip())
            texts.add(t)
            if "%" in t:
                try:
                    pcts.append(int(t.split("%")[0].strip().split()[-1]))
                except Exception:
                    pass
            low = t.lower()
            if "recording" not in low and "starting" not in low and cur.is_enabled():
                if self.n_arts() <= arts0:
                    err_txt = t
                    break
            # NB: the traj file is created when recording STARTS, so a new
            # artifact card appearing does NOT mean the job finished. Wait
            # for the button to leave the recording state instead.
            if self.n_arts() > arts0:
                done_art = True
            if done_art and cur.is_enabled():
                break
        elapsed = time.time() - t0

        self.check("14 state survives the 4s re-render", len(texts) >= 2,
                   str(len(texts)) + " distinct states")
        self.check("15 no error state", not err_txt, err_txt)
        self.check("16 a new artifact appeared", done_art,
                   str(arts0) + " -> " + str(self.n_arts()))
        self.check("17 completed in under 90s", done_art and elapsed < 90,
                   "%.0fs" % elapsed)
        self.check("18 a progress percentage was shown", len(pcts) > 0,
                   str(len(pcts)) + " samples " + str(pcts[:6]))
        self.check("19 percentage is non-decreasing", pcts == sorted(pcts),
                   str(pcts[:8]))
        self.check("20 percentage within 0..100",
                   all(0 <= p <= 100 for p in pcts), str(pcts[:8]))
        self.check("21 percentage advanced past zero", any(p > 0 for p in pcts),
                   str(pcts[:8]))
        # the artifact file lands slightly BEFORE the JS's 1s poll sees
        # status=done, so give the UI a moment to settle rather than racing it
        reenabled = False
        for _ in range(15):
            cur = self.button(label)
            if cur is not None and cur.is_enabled():
                reenabled = True
                break
            time.sleep(1)
        self.check("22 button re-enabled after completion",
                   bool(done_art and reenabled))
        e1 = self.severe()
        self.check("23 no severe console errors while recording", not e1,
                   e1[0]["message"][:120] if e1 else "")

        arts = self.d.find_elements(By.CSS_SELECTOR, "#artifacts .art")
        self.check("24 artifact list non-empty", len(arts) > 0, str(len(arts)))
        txt = A(arts[0].text) if arts else ""
        self.check("25 artifact shows a step count", "steps" in txt, txt[:60])
        self.check("26 artifact has non-zero size",
                   "KB" in txt and "0 KB" not in txt, txt[:60])
        self.check("27 artifact offers a watch button",
                   bool(arts) and len(arts[0].find_elements(
                       By.CSS_SELECTOR, "button.watch")) == 1)

        b3 = self.button(label)
        if b3:
            self.d.execute_script("arguments[0].click();", b3)
            time.sleep(1)
            self.d.execute_script("arguments[0].click();", b3)
            time.sleep(3)
        self.check("28 double click does not wedge the button",
                   self.button(label) is not None)
        try:
            self.check("29 server healthy after double click",
                       "runs" in self.api("/api/runs"))
        except Exception as e:
            self.check("29 server healthy after double click", False, e)
        self.check("30 page still responsive at end", len(self.cards()) > 0)
        return self.finish()

    def finish(self):
        try:
            self.d.quit()
        except Exception:
            pass
        p = sum(1 for _, ok, _ in self.results if ok)
        n = len(self.results)
        print("--- " + str(p) + "/" + str(n) + " passed")
        return p, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8082/viewer/runs.html")
    ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args()
    s = Suite(a.url, a.timeout)
    try:
        p, n = s.run()
    except Exception as e:
        try:
            s.d.quit()
        except Exception:
            pass
        print("SUITE ERROR: " + A(e))
        sys.exit(2)
    print("RESULT: " + ("ALL PASS" if p == n else str(n - p) + " FAILED"))
    sys.exit(0 if p == n else 1)


if __name__ == "__main__":
    main()
