"""
core_engine.py
CLASSIFICATION: V12.0 Data Plane Orchestrator (Network Bridge Active)
GOAL: Encapsulates blocking hunt logic with Dynamic Steering and V12 Remote Dispatch.
"""
import os
import sys
import json
import subprocess
import hashlib
import logging
import time
import datetime
import threading
from typing import Dict, Any

# Try to import paramiko for V12 Bridge
try:
    import paramiko
except ImportError:
    paramiko = None

try:
    import settings
except ImportError:
    pass

from aste_hunter import Hunter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CoreEngine] - %(message)s')

# --- STEERING ---
STEERING_LOCK = threading.Lock()
STEERING_OVERRIDE = {}

def update_steering_config(new_params: Dict[str, Any]):
    with STEERING_LOCK:
        STEERING_OVERRIDE.update(new_params)
    logging.info(f"Steering Interceptor Updated: {new_params}")

def generate_deterministic_hash(params: dict) -> str:
   param_str = json.dumps(params, sort_keys=True).encode('utf-8')
   return hashlib.sha1(param_str).hexdigest()

# --- V12 DCO: RUNNER ABSTRACTION ---
class JobRunner:
    def run_worker(self, job_uuid, config_path): raise NotImplementedError
    def run_validator(self, job_uuid): raise NotImplementedError

class LocalRunner(JobRunner):
    """V11 Legacy Runner (Same Machine)"""
    def run_worker(self, job_uuid, config_path):
        cmd = [sys.executable, settings.WORKER_SCRIPT, "--job_uuid", job_uuid, "--config_path", config_path]
        subprocess.run(cmd, check=True, timeout=settings.JOB_TIMEOUT_SECONDS, capture_output=True)

    def run_validator(self, job_uuid):
        cmd = [sys.executable, settings.VALIDATOR_SCRIPT, "--job_uuid", job_uuid]
        subprocess.run(cmd, check=True, timeout=settings.VALIDATOR_TIMEOUT_SECONDS, capture_output=True)

class RemoteRunner(JobRunner):
    """V12 DCO Runner (Headless PC via SSH/SCP)"""
    def __init__(self, host, user, key_path, remote_dir):
        if not paramiko:
            raise ImportError("Paramiko required for RemoteRunner. pip install paramiko")
        self.host = host
        self.user = user
        self.key = key_path
        self.remote_dir = remote_dir
        self.ssh = self._connect()

    def _connect(self):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.host, username=self.user, key_filename=self.key)
        return client

    def run_worker(self, job_uuid, config_path):
        sftp = self.ssh.open_sftp()
        
        # 1. Push Config
        remote_cfg = f"{self.remote_dir}/input_configs/config_{job_uuid}.json"
        sftp.put(config_path, remote_cfg)
        
        # 2. Execute Worker Remotely
        # We use the remote python in the venv created by deploy_lifecycle.sh
        cmd = f"{self.remote_dir}/venv/bin/python3 {self.remote_dir}/{settings.WORKER_SCRIPT} --job_uuid {job_uuid} --config_path {remote_cfg}"
        stdin, stdout, stderr = self.ssh.exec_command(cmd, timeout=settings.JOB_TIMEOUT_SECONDS)
        
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            err = stderr.read().decode()
            raise Exception(f"Remote Worker Failed: {err}")

        # 3. Pull Artifact (The Bridge)
        remote_h5 = f"{self.remote_dir}/simulation_data/rho_history_{job_uuid}.h5"
        local_h5 = os.path.join(settings.DATA_DIR, f"rho_history_{job_uuid}.h5")
        
        try:
            sftp.get(remote_h5, local_h5)
        except FileNotFoundError:
             raise Exception("Remote Worker finished but produced no HDF5 artifact.")
        
        sftp.close()

    def run_validator(self, job_uuid):
        # Validation happens LOCALLY on the "Brain" PC using the pulled artifact
        LocalRunner().run_validator(job_uuid)

def get_runner():
    # Toggle based on settings
    conf = getattr(settings, 'V12_DCO_CONFIG', {})
    if conf.get("USE_REMOTE", False):
        logging.info(f"[DCO] Using Network Bridge to {conf['HOST']}")
        return RemoteRunner(conf["HOST"], conf["USER"], conf["KEY_PATH"], conf["REMOTE_DIR"])
    return LocalRunner()

def _generate_config_file(job_uuid, params, gen, i, grid, steps, dt):
   config = {
       settings.HASH_KEY: job_uuid,
       "generation": gen,
       "seed": (gen * 1000) + i,
       "N_grid": grid,
       "T_steps": steps,
       "dt": dt,
       **params
   }
   path = os.path.join(settings.CONFIG_DIR, f"config_{job_uuid}.json")
   with open(path, 'w') as f: json.dump(config, f, indent=2)
   return path

def execute_hunt(num_generations, population_size, grid_size, t_steps, dt):
   runner = get_runner()
   
   for d in [settings.CONFIG_DIR, settings.DATA_DIR, settings.PROVENANCE_DIR]:
       os.makedirs(d, exist_ok=True)

   hunter = Hunter()
   
   for gen in range(num_generations):
       logging.info(f"--- GENERATION {gen}/{num_generations-1} ---")
       
       current_dt = dt
       with STEERING_LOCK:
           if "dt" in STEERING_OVERRIDE: current_dt = float(STEERING_OVERRIDE["dt"])
           
       param_batch = hunter.breed_next_generation(population_size)
       jobs = []

       for i, params in enumerate(param_batch):
           with STEERING_LOCK: params.update(STEERING_OVERRIDE)
           
           job_uuid = generate_deterministic_hash(params)
           cfg_path = _generate_config_file(job_uuid, params, gen, i, grid_size, t_steps, current_dt)
           jobs.append((job_uuid, cfg_path))
           
           if not any(r[settings.HASH_KEY] == job_uuid for r in hunter.population):
               hunter.population.append({"generation": gen, settings.HASH_KEY: job_uuid, **params})

       completed = 0
       for uuid, cfg in jobs:
           try:
               runner.run_worker(uuid, cfg)
               runner.run_validator(uuid)
               completed += 1
           except Exception as e:
               logging.error(f"Job {uuid} failed: {e}")
       
       hunter.process_generation_results()
       
       best = hunter.get_best_run()
       if best:
           logging.info(f"Gen {gen} Best: {best.get('fitness', 0):.4f}")
           _write_telemetry(gen, best)
           
   return hunter.get_best_run()

def _write_telemetry(gen, best_run):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "gen": gen,
        "sse": best_run.get(settings.SSE_METRIC_KEY),
        "uuid": best_run.get(settings.HASH_KEY)
    }
    with open("aste_telemetry_history.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")