"""
Example: Using TrajectoryPadder to convert variable-particle trajectories for VMD.
"""

import pandas as pd
import numpy as np
from trajectory_padding import TrajectoryPadder

# ============================================================================
# Example 1: Create synthetic trajectory with varying particle counts
# ============================================================================

def create_example_trajectory():
    """
    Create a synthetic trajectory where particle counts vary per frame.
    
    Simulates a system with:
    - Type 0 particles: varies 50-60
    - Type 1 particles: varies 30-35
    - Type 2 particles: varies 20-25
    """
    frames_data = []
    
    np.random.seed(42)
    
    for frame in range(10):
        # Varying counts per type per frame
        n_type0 = np.random.randint(50, 61)
        n_type1 = np.random.randint(30, 36)
        n_type2 = np.random.randint(20, 26)
        
        # Generate random positions
        for _ in range(n_type0):
            frames_data.append({
                'time': frame,
                'x': np.random.uniform(0, 10),
                'y': np.random.uniform(0, 10),
                'z': np.random.uniform(0, 10),
                'type': 0
            })
        
        for _ in range(n_type1):
            frames_data.append({
                'time': frame,
                'x': np.random.uniform(0, 10),
                'y': np.random.uniform(0, 10),
                'z': np.random.uniform(0, 10),
                'type': 1
            })
        
        for _ in range(n_type2):
            frames_data.append({
                'time': frame,
                'x': np.random.uniform(0, 10),
                'y': np.random.uniform(0, 10),
                'z': np.random.uniform(0, 10),
                'type': 2
            })
    
    return pd.DataFrame(frames_data)


# ============================================================================
# Example 2: Use the padder
# ============================================================================

if __name__ == "__main__":
    # Load or create trajectory
    print("Creating example trajectory...")
    df = create_example_trajectory()
    print(f"DataFrame shape: {df.shape}")
    print(f"Particle types: {sorted(df['type'].unique())}")
    
    # Define type-to-element mapping
    type_to_element = {
        0: 'C',  # Carbon
        1: 'N',  # Nitrogen
        2: 'O'   # Oxygen
    }
    
    # Create padder
    print("\n" + "="*60)
    print("Initializing TrajectoryPadder...")
    padder = TrajectoryPadder(
        df=df,
        type_to_element=type_to_element,
        parking_position=(999.0, 999.0, 999.0)
    )
    
    # Analyze trajectory
    print("\n" + "="*60)
    max_counts = padder.analyze()
    
    # Print summary
    print("\n" + "="*60)
    print(padder.summary())
    
    # Write outputs
    print("\n" + "="*60)
    print("Writing outputs...")
    
    # PDB topology (always write this first)
    padder.write_pdb('topology.pdb')
    
    # DCD trajectory (requires mdtraj; will use the PDB we just wrote)
    try:
        padder.write_dcd('trajectory.dcd', topology_pdb='topology.pdb')
    except ImportError as e:
        print(f"Note: {e}")
        print("To enable DCD writing, install mdtraj: pip install mdtraj")
    
    # ========================================================================
    # Example 3: Inspect the padded data
    # ========================================================================
    print("\n" + "="*60)
    print("Inspecting padded data...")
    
    # Get data for first frame
    positions, types = padder.get_frame_data(0)
    print(f"\nFrame 0:")
    print(f"  Shape: {positions.shape}")
    print(f"  Unique types: {np.unique(types)}")
    print(f"  Type counts: {np.bincount(types)}")
    
    # Show which particles are at parking position (dummy particles)
    parking_mask = np.all(positions == (999.0, 999.0, 999.0), axis=1)
    n_real = np.sum(~parking_mask)
    n_dummy = np.sum(parking_mask)
    print(f"  Real particles: {n_real}")
    print(f"  Dummy particles: {n_dummy}")
    
    print("\n" + "="*60)
    print("Done! Files written:")
    print("  - topology.pdb (VMD topology)")
    print("  - trajectory.dcd (VMD trajectory)")
    print("\nYou can now load these into VMD!")