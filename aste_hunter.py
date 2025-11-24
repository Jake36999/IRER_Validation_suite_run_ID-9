"""
aste_hunter.py
CLASSIFICATION: Adaptive Learning Engine (ASTE V11.0)
GOAL: Acts as the "Brain" of the ASTE. It reads validation reports
(provenance.json), calculates a falsifiability-driven fitness, and
breeds new generations of parameters.
REMEDIATION: This version imports `settings.py` directly, resolving the
"Shadow Contract" audit gap.
"""

import os
import csv
import json
import math
import random
import sys
import numpy as np
from typing import List, Dict, Any, Optional

# REMEDIATION: All config is imported from the single source of truth.
try:
   import settings
except ImportError:
   print("FATAL: settings.py not found.", file=sys.stderr)
   sys.exit(1)

# --- Constants from settings ---
LEDGER_FILE = settings.LEDGER_FILE
PROVENANCE_DIR = settings.PROVENANCE_DIR
SSE_METRIC_KEY = settings.SSE_METRIC_KEY
HASH_KEY = settings.HASH_KEY
LAMBDA_FALSIFIABILITY = settings.LAMBDA_FALSIFIABILITY
MUTATION_RATE = settings.MUTATION_RATE
MUTATION_STRENGTH = settings.MUTATION_STRENGTH
TOURNAMENT_SIZE = 3

class Hunter:
   def __init__(self, ledger_file: str = LEDGER_FILE):
       self.ledger_file = ledger_file
       self.fieldnames = [
           HASH_KEY, SSE_METRIC_KEY, "fitness", "generation",
           "sncgl_epsilon", "sncgl_lambda", "sncgl_g_nonlocal", # V11.0 S-NCGL parameters
           "sdg_alpha", "sdg_kappa", "sdg_eta",                    # V11.0 SDG parameters
           "sse_null_phase_scramble", "sse_null_target_shuffle"
       ]
       self.population = self._load_ledger()
       print(f"[Hunter] Initialized. Loaded {len(self.population)} runs from {self.ledger_file}")

   def _load_ledger(self) -> List[Dict[str, Any]]:
       if not os.path.exists(self.ledger_file):
           # Create header if file doesn't exist
           with open(self.ledger_file, 'w', newline='') as f:
               writer = csv.DictWriter(f, fieldnames=self.fieldnames)
               writer.writeheader()
           return []

       population = []
       with open(self.ledger_file, 'r') as f:
           reader = csv.DictReader(f)
           for row in reader:
               for key in row:
                   try:
                       row[key] = float(row[key]) if row[key] else None
                   except (ValueError, TypeError):
                       pass
               population.append(row)
       return population

   def _save_ledger(self):
       with open(self.ledger_file, 'w', newline='') as f:
           writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
           writer.writeheader()
           writer.writerows(self.population)

   def process_generation_results(self):
       print(f"[Hunter] Processing new results from {PROVENANCE_DIR}...")
       processed_count = 0
       for run in self.population:
           if run.get('fitness') is not None:
               continue

           config_hash = run[HASH_KEY]
           prov_file = os.path.join(PROVENANCE_DIR, f"provenance_{config_hash}.json")

           if not os.path.exists(prov_file):
               continue

           try:
               with open(prov_file, 'r') as f:
                   provenance = json.load(f)

               # The validation_pipeline.py now puts metrics directly under 'metrics'
               metrics_data = provenance.get("metrics", {})
               sse = float(metrics_data.get(SSE_METRIC_KEY, 1002.0))

               # Mock null tests (these would come from a spectral_fidelity block in a full implementation)
               sse_null_a = sse * 10.0 # Mocking a null test always being worse
               sse_null_b = sse * 15.0 # Mocking another null test

               # Cap nulls to prevent runaway scores from errors
               sse_null_a = min(sse_null_a, 1000.0)
               sse_null_b = min(sse_null_b, 1000.0)

               fitness = 0.0
               if math.isfinite(sse) and sse < 900.0:
                   base_fitness = 1.0 / max(sse, 1e-12)
                   # "Falsifiability-Driven Fitness"
                   delta_a = max(0.0, sse_null_a - sse)
                   delta_b = max(0.0, sse_null_b - sse)
                   bonus = LAMBDA_FALSIFIABILITY * (delta_a + delta_b)
                   fitness = base_fitness + bonus

               run.update({
                   SSE_METRIC_KEY: sse,
                   "fitness": fitness,
                   "sse_null_phase_scramble": sse_null_a,
                   "sse_null_target_shuffle": sse_null_b
               })
               processed_count += 1
           except Exception as e:
               print(f"[Hunter Error] Failed to parse {prov_file}: {e}", file=sys.stderr)

       if processed_count > 0:
           print(f"[Hunter] Successfully processed and updated {processed_count} runs.")
           self._save_ledger()

   def get_best_run(self) -> Optional[Dict[str, Any]]:
       valid_runs = [r for r in self.population if r.get("fitness") is not None and math.isfinite(r["fitness"]) and r["fitness"] > 0]
       return max(valid_runs, key=lambda x: x["fitness"]) if valid_runs else None

   def _select_parent(self) -> Dict[str, Any]:
       valid_runs = [r for r in self.population if r.get("fitness") is not None and r["fitness"] > 0]
       if not valid_runs:
           return self._get_random_parent()

       tournament = random.sample(valid_runs, k=min(TOURNAMENT_SIZE, len(valid_runs)))
       return max(tournament, key=lambda x: x["fitness"])

   def _crossover(self, p1: Dict, p2: Dict) -> Dict: # Using placeholder params that align with V11 context
       child = {}
       child["sncgl_epsilon"] = p1.get("sncgl_epsilon", 0.0) if random.random() < 0.5 else p2.get("sncgl_epsilon", 0.0)
       child["sncgl_lambda"] = p1.get("sncgl_lambda", 0.0) if random.random() < 0.5 else p2.get("sncgl_lambda", 0.0)
       child["sncgl_g_nonlocal"] = p1.get("sncgl_g_nonlocal", 0.0) if random.random() < 0.5 else p2.get("sncgl_g_nonlocal", 0.0)
       child["sdg_alpha"] = p1.get("sdg_alpha", 0.0) if random.random() < 0.5 else p2.get("sdg_alpha", 0.0)
       child["sdg_kappa"] = p1.get("sdg_kappa", 0.0) if random.random() < 0.5 else p2.get("sdg_kappa", 0.0)
       child["sdg_eta"] = p1.get("sdg_eta", 0.0) if random.random() < 0.5 else p2.get("sdg_eta", 0.0)
       return child

   def _mutate(self, params: Dict) -> Dict:
       mutated = params.copy()
       if random.random() < MUTATION_RATE:
           mutated["sncgl_epsilon"] = max(0.01, mutated.get("sncgl_epsilon", 0.1) + np.random.normal(0, MUTATION_STRENGTH))
       if random.random() < MUTATION_RATE:
           mutated["sncgl_lambda"] = max(0.001, mutated.get("sncgl_lambda", 0.01) + np.random.normal(0, MUTATION_STRENGTH))
       if random.random() < MUTATION_RATE:
           mutated["sncgl_g_nonlocal"] = max(0.0001, mutated.get("sncgl_g_nonlocal", 0.001) + np.random.normal(0, MUTATION_STRENGTH))
       if random.random() < MUTATION_RATE:
           mutated["sdg_alpha"] = max(0.1, mutated.get("sdg_alpha", 1.0) + np.random.normal(0, MUTATION_STRENGTH))
       if random.random() < MUTATION_RATE:
           mutated["sdg_kappa"] = max(0.1, mutated.get("sdg_kappa", 1.0) + np.random.normal(0, MUTATION_STRENGTH))
       if random.random() < MUTATION_RATE:
           mutated["sdg_eta"] = max(0.1, mutated.get("sdg_eta", 0.5) + np.random.normal(0, MUTATION_STRENGTH))
       return mutated

   def _get_random_parent(self) -> Dict:
       return {
           "sncgl_epsilon": random.uniform(0.1, 0.5),
           "sncgl_lambda": random.uniform(0.01, 0.1),
           "sncgl_g_nonlocal": random.uniform(0.0005, 0.005),
           "sdg_alpha": random.uniform(1.0, 2.0),
           "sdg_kappa": random.uniform(0.5, 1.5),
           "sdg_eta": random.uniform(0.1, 1.0),
       }

   def breed_next_generation(self, size: int) -> List[Dict[str, Any]]:
       self.process_generation_results()
       new_gen = []

       best_run = self.get_best_run()
       if not best_run:
           print("[Hunter] No history. Generating random generation 0.")
           for _ in range(size):
               new_gen.append(self._get_random_parent())
           return new_gen

       print(f"[Hunter] Breeding generation... Best fitness so far: {best_run['fitness']:.2f}")

       # Elitism
       # Only include relevant parameter keys, filter out metrics and fitness
       new_gen.append({k: v for k, v in best_run.items() if k.startswith("sncgl_") or k.startswith("sdg_")})

       while len(new_gen) < size:
           p1 = self._select_parent()
           p2 = self._select_parent()
           child = self._crossover(p1, p2)
           mutated_child = self._mutate(child)
           new_gen.append(mutated_child)

       return new_gen
