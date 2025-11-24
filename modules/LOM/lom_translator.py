"""
MODULE: lom_translator.py
CLASSIFICATION: Group E (Infrastructure / Physical)
GOAL: Translates Density Fields (Rho) into G-Code for LOM Etching.
"""
import h5py
import numpy as np
import argparse
import os
import settings

def generate_gcode(density_map, threshold=0.5):
    """
    Converts a 2D density map into G-Code instructions.
    - X, Y: Coordinates
    - S: Laser Power (mapped from Density)
    """
    H, W = density_map.shape
    lines = []
    lines.append("G21 ; Metric units")
    lines.append("G90 ; Absolute positioning")
    lines.append("M3 S0 ; Laser on, zero power")
    
    # Scanline approach
    for y in range(H):
        for x in range(W):
            val = density_map[y, x]
            if val > threshold:
                # Map density 0.0-1.0 to Spindle Speed 0-1000
                power = int(np.clip(val * 1000, 0, 1000))
                lines.append(f"G1 X{x*0.1:.3f} Y{y*0.1:.3f} S{power}")
    
    lines.append("M5 ; Laser off")
    lines.append("G0 X0 Y0 ; Home")
    return "\n".join(lines)

def translate_artifact(job_uuid):
    h5_path = os.path.join(settings.DATA_DIR, f"rho_history_{job_uuid}.h5")
    if not os.path.exists(h5_path):
        print(f"Artifact not found: {h5_path}")
        return

    with h5py.File(h5_path, 'r') as f:
        psi = f['final_psi'][()]
        rho = np.abs(psi)**2
        
        # Slice middle layer for 2D printing
        mid = rho.shape[0] // 2
        layer = rho[mid, :, :]
        
        # Normalize
        layer = layer / (np.max(layer) + 1e-9)
        
        gcode = generate_gcode(layer)
        
        out_path = f"lom_output_{job_uuid}.gcode"
        with open(out_path, 'w') as out:
            out.write(gcode)
            
        print(f"LOM G-Code generated: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_uuid", required=True)
    args = parser.parse_args()
    translate_artifact(args.job_uuid)