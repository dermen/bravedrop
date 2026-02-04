import os
import time
import requests
import psutil
import signal
import subprocess
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pynvml import *
import brave

from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument("--python", required=True, type=str, help="path to bython binary for running brave")
ap.add_argument("--host", type=str, default="0.0.0.0", help="host for running this healcheck dash")
ap.add_argument("--port", default=8888, type=int, help="port for running the healthcheck dash")
ap.add_argument("--marcoPort", default=8000, type=int, help="port for relaunching marco score server")
ap.add_argument("--xtalhuntPort", default=8001, type=int, help="port for relaunching xtalhunting server")
args = ap.parse_args()

app = FastAPI()

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.join(os.path.dirname(brave.__file__), "../servers")

LAUNCH_COMMANDS = {
    8000: f"nohup {args.python} marco_server.py --port {args.marcoPort} > marco.log 2>&1 &",
    8001: f"nohup {args.python} xtalhunting_server.py --port {args.xtalhuntPort} > xtal.log 2>&1 &"
}

MONITORED_SERVICES = {
    8000: "MARCO Scorer",
    8001: "Crystal Hunter"
}

def get_service_info(port):
    info = {"status": "OFFLINE", "v_server": "N/A", "v_torch": "N/A", "pid": None}
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            try:
                p = psutil.Process(conn.pid)
                info.update({"status": "ONLINE", "pid": conn.pid})
                try:
                    resp = requests.get(f"http://127.0.0.1:{port}/version", timeout=0.6)
                    if resp.status_code == 200:
                        data = resp.json()
                        info["v_server"] = data.get("server-version", "N/A")
                        info["v_torch"] = data.get("torch-version", "N/A")
                except: pass
            except: pass
    return info

@app.post("/terminate/{pid}")
async def terminate_process(pid: int):
    try:
        os.kill(pid, signal.SIGTERM)
        return {"status": "terminated"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/launch/{port}")
async def launch_service(port: int):
    try:
        command = LAUNCH_COMMANDS.get(port)
        if command:
            # We use 'cwd' to ensure the script starts in the project folder
            subprocess.Popen(
                command, 
                shell=True, 
                cwd=PROJECT_ROOT,
                preexec_fn=os.setpgrp
            )
            return {"status": "launching"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    util = nvmlDeviceGetUtilizationRates(handle).gpu
    mem = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()

    rows = ""
    for port, name in MONITORED_SERVICES.items():
        s = get_service_info(port)
        is_online = s['status'] == "ONLINE"
        status_bg = "#006400" if is_online else "#8b0000"
        
        if is_online:
            action_btn = f"<button class='kill-btn' onclick='killProc(event, {s['pid']})'>KILL</button>"
        else:
            action_btn = f"<button class='launch-btn' onclick='launchProc(event, {port})'>LAUNCH</button>"
        
        rows += f"""
        <tr>
            <td class="name-cell">{name}</td>
            <td class="port-cell">{port}</td>
            <td><span class="status-pill" style="background:{status_bg}">{s['status']}</span></td>
            <td class="torch-cell">{s['v_torch']}</td>
            <td class="version-cell">{s['v_server']}</td>
            <td>{action_btn}</td>
        </tr>"""

    return f"""
    <html>
        <head>
            <style>
                body {{ font-family: 'Inter', sans-serif; background: #050505; color: #e0e0e0; padding: 20px; }}
                .header-area {{ margin-bottom: 30px; border-left: 10px solid #00d4ff; padding-left: 20px; }}
                .gpu-banner {{ font-size: 22px; color: #ffa500; font-weight: bold; background: #111; padding: 15px; border-radius: 8px; }}
                table {{ width: 100%; border-collapse: separate; border-spacing: 0 10px; }}
                td {{ padding: 20px 15px; background: #1a1a1a; vertical-align: middle; }}
                .port-cell {{ font-size: 22px; font-weight: 800; color: #ff00ff; font-family: monospace; }}
                .status-pill {{ padding: 10px 25px; border-radius: 50px; font-size: 22px; font-weight: 900; color: white; display: inline-block; min-width: 140px; text-align: center; }}
                .torch-cell {{ font-size: 22px; font-weight: 700; color: #00ff00; }}
                .version-cell {{ font-size: 22px; font-family: monospace; color: #888; max-width: 300px; word-break: break-all; }}
                .name-cell {{ font-size: 22px; font-weight: 600; color: #fff; }}
                .kill-btn {{ background: #ff4444; color: white; border: none; padding: 15px 25px; font-size: 22px; border-radius: 8px; cursor: pointer; font-weight: bold; }}
                .launch-btn {{ background: #008CBA; color: white; border: none; padding: 15px 25px; font-size: 22px; border-radius: 8px; cursor: pointer; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header-area">
                <h1 style="margin:0; font-size: 22px;">pxgpu03 CONTROL</h1>
                <div class="gpu-banner">GPU: {util}% | VRAM: {mem.used//1024**2}MB / {mem.total//1024**2}MB</div>
            </div>
            <table>
                <thead><tr><th>Service</th><th>Port</th><th>Status</th><th>Torch</th><th>Build Version</th><th>Action</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
            <script>
                async function killProc(e, pid) {{
                    if (confirm("TERMINATE PROCESS " + pid + "?")) {{
                        e.target.innerText = "KILLING...";
                        await fetch('/terminate/' + pid, {{ method: 'POST' }});
                        setTimeout(() => location.reload(), 800);
                    }}
                }}

                async function launchProc(e, port) {{
                    e.target.innerText = "LAUNCHING...";
                    await fetch('/launch/' + port, {{ method: 'POST' }});
                    // AI models take time to load onto GPU, wait 3 seconds
                    setTimeout(() => location.reload(), 12000);
                }}
            </script>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
