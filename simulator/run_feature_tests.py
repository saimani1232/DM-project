"""
run_feature_tests.py
Runs feature_tests.js against ABS_Print_Lab.html in headless Microsoft Edge (or Chrome) through the
DevTools protocol: every control is driven like a user would (slider events, button clicks, canvas
clicks) and each result is checked against the model. Needs: pip install websocket-client

Run from the project root:  python simulator/run_feature_tests.py [--runs N] [--mobile]
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

HERE = Path(__file__).resolve().parent
PAGE = (HERE / 'ABS_Print_Lab.html').as_uri()
TESTS = (HERE / 'feature_tests.js').read_text(encoding='utf-8')
CANDIDATES = [
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
]


def main():
    runs = int(sys.argv[sys.argv.index('--runs') + 1]) if '--runs' in sys.argv else 1
    browser = next((p for p in CANDIDATES if os.path.exists(p)), None)
    if not browser:
        raise SystemExit('Microsoft Edge or Google Chrome not found.')
    profile = tempfile.mkdtemp(prefix='printlab_')
    port = 9344
    proc = subprocess.Popen([browser, '--headless=new', '--disable-gpu', f'--remote-debugging-port={port}',
                             f'--user-data-dir={profile}', '--window-size=1440,900', 'about:blank'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    total_fail = 0
    try:
        page = None
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json'))
                page = next(t for t in tabs if t.get('type') == 'page')
                break
            except Exception:
                time.sleep(0.2)
        ws = websocket.create_connection(page['webSocketDebuggerUrl'], timeout=300, suppress_origin=True)
        counter = [0]
        errors = []

        def send(method, params=None):
            counter[0] += 1
            ws.send(json.dumps({'id': counter[0], 'method': method, 'params': params or {}}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get('method') == 'Runtime.exceptionThrown':
                    errors.append(msg['params']['exceptionDetails'].get('text', '') + ' ' +
                                  str(msg['params']['exceptionDetails'].get('exception', {}).get('description', '')))
                if msg.get('method') == 'Runtime.consoleAPICalled' and msg['params']['type'] == 'error':
                    errors.append(' '.join(str(a.get('value', a.get('description', ''))) for a in msg['params']['args']))
                if msg.get('id') == counter[0]:
                    return msg.get('result', {})

        send('Runtime.enable')
        send('Page.enable')
        mobile = '--mobile' in sys.argv
        send('Emulation.setDeviceMetricsOverride', {'width': 375 if mobile else 1440, 'height': 812 if mobile else 900,
                                                    'deviceScaleFactor': 2 if mobile else 1, 'mobile': mobile})
        print('viewport:', 'phone 375 px' if mobile else 'desktop 1440 px')
        for run in range(1, runs + 1):
            send('Page.navigate', {'url': PAGE})
            time.sleep(2.5)
            res = send('Runtime.evaluate', {'expression': TESTS, 'awaitPromise': True, 'returnByValue': True})
            if 'exceptionDetails' in res:
                print('Test script crashed:', res['exceptionDetails'])
                total_fail += 1
                continue
            out = res['result']['value']
            print(f'--- run {run}: {out["pass"]} passed, {out["fail"]} failed')
            for line in out['log']:
                if line.startswith('FAIL') or 'meets the goal' in line:
                    print('   ', line)
            total_fail += out['fail']
        if errors:
            print('Page errors:', *errors, sep='\n  ')
            total_fail += len(errors)
        else:
            print('No JavaScript errors on the page.')
        ws.close()
    finally:
        proc.terminate()
    print('ALL FEATURE TESTS PASSED' if total_fail == 0 else f'{total_fail} FAILURE(S)')
    sys.exit(1 if total_fail else 0)


if __name__ == '__main__':
    main()
