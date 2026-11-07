"""Browser test of the web UI with Playwright + local Chrome.

usage: python scripts/ui_test.py BASE_URL [--video data.mp4] [--record SECONDS] [--shots DIR]
--video   upload flow with this file
--record  live-record flow using Chrome's fake camera for N seconds
"""

import argparse
import os
import time

from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--video")
    ap.add_argument("--record", type=float, default=0)
    ap.add_argument("--shots", default="/tmp/asi_ui_shots")
    a = ap.parse_args()
    os.makedirs(a.shots, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=[
            "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
        ctx = browser.new_context(viewport={"width": 1400, "height": 1000}, permissions=["camera"])
        page = ctx.new_page()
        errors = []
        page.on("console", lambda m: m.type in ("error", "warning") and errors.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.goto(a.base, wait_until="networkidle")
        page.screenshot(path=f"{a.shots}/0_home.png")

        if a.record:
            page.click("[data-tab=record]")
            page.click("#camStart")
            page.wait_for_selector("#recStart:not([disabled])")
            page.click("#recStart")
            print("recording as:", page.inner_text("#recHint"))
            time.sleep(a.record)
            page.click("#recStop")
            page.wait_for_selector("#selected:not([hidden])")
            print("recorded:", page.inner_text("#srcName"), page.inner_text("#srcInfo"))
        else:
            page.set_input_files("#fileInput", a.video)

        t0 = time.time()
        page.click("#analyze")
        last = ""
        while time.time() - t0 < 900:
            if page.is_visible("#results"):
                break
            if page.is_visible("#errorBox"):
                print("ERROR:", page.inner_text("#errorBox"))
                break
            s = page.inner_text("#statusLine")
            if s != last:
                print(f"[{time.time() - t0:5.0f}s] {s[:160]}")
                last = s
            time.sleep(1)
        page.screenshot(path=f"{a.shots}/1_after.png", full_page=True)

        if page.is_visible("#results"):
            print("title:", page.inner_text("#title"))
            print("steps:", page.locator(".step").count(), "procedure rows:", page.locator("#procedure tr").count())
            print("legend:", page.inner_text("#legend").replace("\n", " ")[:300])
            print("stats:", page.inner_text("#trackStats"))
            page.locator(".step").nth(min(1, page.locator(".step").count() - 1)).click()
            time.sleep(1.5)
            painted = page.evaluate("""() => { const c = document.querySelector('#overlay');
              const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0;
              for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return [c.width, c.height, n]; }""")
            print("overlay canvas (w, h, painted px):", painted)
            page.locator(".player").screenshot(path=f"{a.shots}/2_player.png")
            with page.expect_download() as dl:
                page.click("#exportMd")
            print("export:", dl.value.suggested_filename)
        print("console:", errors or "clean")
        browser.close()


if __name__ == "__main__":
    main()
