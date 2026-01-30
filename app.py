"""
app.py
CLASSIFICATION: V11.0 Control Hub (Unified & Governed)
GOAL: Merges V11 Dashboard Telemetry with V13 Governance Hooks.
STATUS: GOLD MASTER (Restores UI Endpoints + Adds V13 Sockets)
"""
import os
import json
import logging
import threading
import shutil
import base64
import io
import time
from flask import Flask, render_template, jsonify, request, send_from_directory
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# V12 Visualization Libs
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
import h5py
import numpy as np
import settings
import core_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ControlHub] - %(message)s')
PROVENANCE_DIR = settings.PROVENANCE_DIR
STATUS_FILE = settings.STATUS_FILE
HUNT_RUNNING_LOCK = threading.Lock()
g_hunt_in_progress = False

app = Flask(__name__, template_folder="templates")

# --- V13 GOVERNANCE SOCKET (The New Logic) ---
def check_governance(command_type: str, payload: dict) -> bool:
    """
    V13 Middleware Stub: 'Umbra'
    In V13, this will call the Ethical Sentinel LLM.
    In V11, it acts as a pass-through.
    """
    # Future Safety Logic goes here
    return True

# --- V11 DASHBOARD LOGIC (Restored) ---
def update_status(new_data: dict = None):
    if new_data is None: new_data = {}
    with HUNT_RUNNING_LOCK:
        status = {settings.API_KEY_HUNT_STATUS: "Idle"}
        if os.path.exists(STATUS_FILE):
            try:
                with open(STATUS_FILE, 'r') as f: status.update(json.load(f))
            except: pass
        status.update(new_data)
        with open(STATUS_FILE, 'w') as f: json.dump(status, f, indent=2)

class ProvenanceWatcher(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            try:
                # Short delay to ensure write complete
                time.sleep(0.1) 
                with open(event.src_path, 'r') as f: data = json.load(f)
                sentinel = data.get("sentinel_code", data.get("validation_status", 0))
                if sentinel == 0 or sentinel == settings.SENTINEL_SUCCESS:
                    metrics = data.get("metrics", {})
                    status_update = {
                        settings.API_KEY_LAST_EVENT: f"Analyzed {data.get(settings.HASH_KEY, 'Unknown')[:8]}",
                        settings.API_KEY_LAST_SSE: f"{metrics.get(settings.SSE_METRIC_KEY, 0):.4f}",
                        settings.API_KEY_LAST_STABILITY: f"{metrics.get(settings.STABILITY_METRIC_KEY, 0):.4f}"
                    }
                    update_status(status_update)
            except Exception as e:
                logging.error(f"Watcher Read Error: {e}")

def start_watcher_service():
    os.makedirs(PROVENANCE_DIR, exist_ok=True)
    observer = Observer()
    observer.schedule(ProvenanceWatcher(), str(PROVENANCE_DIR), recursive=False)
    observer.daemon = True
    observer.start()

def run_hunt_in_background(gens, pop, grid, steps, dt):
    global g_hunt_in_progress
    with HUNT_RUNNING_LOCK:
        if g_hunt_in_progress: return
        g_hunt_in_progress = True

    try:
        update_status({settings.API_KEY_HUNT_STATUS: "Running", "current_gen": 0, "total_gens": gens})
        # Ensure config/data dirs exist
        for p in [settings.CONFIG_DIR, settings.DATA_DIR, settings.PROVENANCE_DIR]:
            os.makedirs(p, exist_ok=True)
            
        res = core_engine.execute_hunt(gens, pop, grid, steps, dt)
        update_status({settings.API_KEY_HUNT_STATUS: "Completed", settings.API_KEY_FINAL_RESULT: res})
    except Exception as e:
        logging.error(f"Hunt failed: {e}")
        update_status({settings.API_KEY_HUNT_STATUS: f"Error: {str(e)}"})
    finally:
        with HUNT_RUNNING_LOCK: g_hunt_in_progress = False

# --- API ENDPOINTS ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start-hunt', methods=['POST'])
def api_start_hunt():
    # 1. Governance Check (V13)
    d = request.get_json(silent=True) or {}
    if not check_governance("START_HUNT", d):
        return jsonify({"error": "Governance Veto"}), 403

    # 2. Concurrency Check
    if g_hunt_in_progress: return jsonify({"error": "Busy"}), 409
    
    evo = d.get('evolutionary', {})
    phys = d.get('physics', {})
    
    args = (
        evo.get('generations', settings.DEFAULT_NUM_GENERATIONS),
        evo.get('population', settings.DEFAULT_POPULATION_SIZE),
        phys.get('grid_size', settings.DEFAULT_GRID_SIZE),
        phys.get('t_steps', settings.DEFAULT_T_STEPS),
        phys.get('dt', settings.DEFAULT_DT)
    )

    # Clean previous run data
    if os.path.exists(settings.STATUS_FILE): 
        os.remove(settings.STATUS_FILE)
    if os.path.exists("aste_telemetry_history.jsonl"): 
        os.remove("aste_telemetry_history.jsonl")

    t = threading.Thread(target=run_hunt_in_background, args=args)
    t.daemon = True; t.start()
    return jsonify({"status": "started"}), 202

@app.route('/api/steer', methods=['POST'])
def api_steer():
    """God Mode: Inject parameters into the running engine."""
    if not check_governance("STEER", request.get_json()): return jsonify({"error": "Veto"}), 403
    
    payload = request.get_json(silent=True)
    if not payload: return jsonify({"error": "No payload"}), 400
    core_engine.update_steering_config(payload)
    return jsonify({"status": "Steering Parameters Updated", "config": payload})

@app.route('/api/render-artifact/<job_uuid>')
def api_render_artifact(job_uuid):
    """Generates Heatmap for UI Inspection."""
    if not job_uuid.isalnum(): return jsonify({"error": "Invalid UUID"}), 400
    
    path = settings.DATA_DIR / f"rho_history_{job_uuid}.h5"
    if not os.path.exists(path):
        return jsonify({"error": "Artifact not found"}), 404

    try:
        with h5py.File(path, 'r') as f:
            psi = f['final_psi'][()]
        
        # Render Magnitude
        magnitude = np.abs(psi)
        
        plt.figure(figsize=(4, 4), dpi=100)
        plt.imshow(magnitude, cmap='inferno', origin='lower')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return jsonify({"image": b64_str, "uuid": job_uuid})
        
    except Exception as e:
        return jsonify({"error": "Render failed"}), 500

@app.route('/api/get-status')
def api_get_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f: return jsonify(json.load(f))
    return jsonify({})

@app.route('/api/get-progress')
def api_get_progress():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f: 
            d = json.load(f)
            return jsonify({"current_gen": d.get("current_gen",0), "total_gens": d.get("total_gens",0)})
    return jsonify({"current_gen": 0, "total_gens": 0})

@app.route('/api/get-constants')
def api_get_constants():
    return jsonify({
        "HUNT_STATUS": settings.API_KEY_HUNT_STATUS,
        "LAST_EVENT": settings.API_KEY_LAST_EVENT,
        "LAST_SSE": settings.API_KEY_LAST_SSE,
        "LAST_STABILITY": settings.API_KEY_LAST_STABILITY,
        "FINAL_RESULT": settings.API_KEY_FINAL_RESULT
    })

@app.route('/api/get-artifact/<job_uuid>')
def api_get_artifact(job_uuid):
    if not job_uuid.isalnum(): return jsonify({"error": "Invalid UUID"}), 400
    return send_from_directory(settings.PROVENANCE_DIR, f"provenance_{job_uuid}.json")

if __name__ == '__main__':
    if not os.path.exists("templates"): os.makedirs("templates")
    update_status() 
    start_watcher_service()
    app.run(host='0.0.0.0', port=8080)
