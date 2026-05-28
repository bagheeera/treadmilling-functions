"""
Trajectory padding tool: convert variable-particle-count trajectories to fixed-topology format.

Handles:
- Multiple particle types with separate max counts
- Padding frames with dummy particles at a parking position
- Writing PDB topology file
- Writing DCD trajectory compatible with VMD
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

try:
    import mdtraj as md
    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False
    warnings.warn("mdtraj not installed. DCD and PDB writing will not work.")


class TrajectoryPadder:
    """
    Convert trajectories with variable particle counts to fixed-topology format.
    
    Example:
        >>> df = pd.DataFrame({
        ...     'time': [0, 0, 0, 1, 1],
        ...     'x': [1.0, 2.0, 3.0, 1.5, 2.5],
        ...     'y': [1.0, 2.0, 3.0, 1.5, 2.5],
        ...     'z': [1.0, 2.0, 3.0, 1.5, 2.5],
        ...     'type': [0, 0, 1, 0, 1]
        ... })
        >>> padder = TrajectoryPadder(df, type_to_element={0: 'C', 1: 'N'})
        >>> padder.analyze()
        >>> padder.write_pdb('topology.pdb')
        >>> padder.write_dcd('trajectory.dcd')
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        type_to_element: Dict[int, str],
        parking_position: Tuple[float, float, float] = (999.0, 999.0, 999.0),
        time_column: str = 'time',
        x_column: str = 'x',
        y_column: str = 'y',
        z_column: str = 'z',
        type_column: str = 'type'
    ):
        """
        Initialize the padder.
        
        Args:
            df: DataFrame with columns [time, x, y, z, type, ...]
            type_to_element: Dict mapping type (int) to element symbol (str)
                            e.g., {0: 'C', 1: 'N', 2: 'O'}
            parking_position: (x, y, z) coord for dummy particles
            time_column: Name of time/frame column
            x_column, y_column, z_column: Names of coordinate columns
            type_column: Name of type column
        """
        self.df = df.copy()
        self.type_to_element = type_to_element
        self.parking_position = parking_position
        self.time_col = time_column
        self.x_col = x_column
        self.y_col = y_column
        self.z_col = z_column
        self.type_col = type_column
        
        # Will be computed in analyze()
        self.max_counts: Dict[int, int] = {}
        self.total_particles: int = 0
        self.n_frames: int = 0
        self.frames: List[int] = []
        self.padded_trajectory: Optional[np.ndarray] = None
        self.topology_df: Optional[pd.DataFrame] = None
        
    def analyze(self) -> Dict[int, int]:
        """
        Analyze trajectory to find max particle count per type.
        
        Returns:
            Dictionary of {type: max_count}
        """
        # Get unique frames
        self.frames = sorted(self.df[self.time_col].unique())
        self.n_frames = len(self.frames)
        
        # Find max count per type across all frames
        for frame in self.frames:
            frame_data = self.df[self.df[self.time_col] == frame]
            type_counts = frame_data[self.type_col].value_counts()
            
            for ptype, count in type_counts.items():
                if ptype not in self.max_counts:
                    self.max_counts[ptype] = count
                else:
                    self.max_counts[ptype] = max(self.max_counts[ptype], count)
        
        # Total particles in padded system
        self.total_particles = sum(self.max_counts.values())
        
        print(f"Analysis complete:")
        print(f"  Frames: {self.n_frames}")
        print(f"  Particle types: {sorted(self.max_counts.keys())}")
        for ptype, count in sorted(self.max_counts.items()):
            print(f"    Type {ptype}: max {count} particles")
        print(f"  Total padded size: {self.total_particles} particles")
        
        return self.max_counts
    
    def _build_padded_trajectory(self) -> np.ndarray:
        """
        Build padded trajectory array of shape (n_frames, n_particles, 3).
        
        Returns:
            Array with padding particles at parking_position
        """
        if not self.max_counts:
            raise ValueError("Call analyze() first")
        
        padded = np.zeros((self.n_frames, self.total_particles, 3), dtype=np.float32)
        
        # Fill with parking position as default
        padded[:, :] = self.parking_position
        
        # Track insertion index per type
        type_indices = {ptype: 0 for ptype in self.max_counts.keys()}
        
        # Assign particle indices by type (deterministic ordering)
        particle_to_type = {}
        particle_idx = 0
        for ptype in sorted(self.max_counts.keys()):
            for _ in range(self.max_counts[ptype]):
                particle_to_type[particle_idx] = ptype
                particle_idx += 1
        
        # Fill frames
        for frame_idx, frame in enumerate(self.frames):
            frame_data = self.df[self.df[self.time_col] == frame]
            
            # Reset type indices for this frame
            type_indices = {ptype: 0 for ptype in self.max_counts.keys()}
            
            # Place real particles
            for _, row in frame_data.iterrows():
                ptype = int(row[self.type_col])
                # Find the next available slot for this type
                particle_idx = sum(self.max_counts[t] for t in sorted(self.max_counts.keys()) if t < ptype)
                particle_idx += type_indices[ptype]
                
                padded[frame_idx, particle_idx] = [
                    float(row[self.x_col]),
                    float(row[self.y_col]),
                    float(row[self.z_col])
                ]
                type_indices[ptype] += 1
        
        return padded
    
    def _build_topology_dataframe(self) -> pd.DataFrame:
        """
        Build a topology DataFrame compatible with biopandas/PDB format.
        
        Returns:
            DataFrame with columns for PDB structure
        """
        if not self.max_counts:
            raise ValueError("Call analyze() first")
        
        rows = []
        particle_idx = 1  # PDB uses 1-indexed atom numbers
        
        for ptype in sorted(self.max_counts.keys()):
            element = self.type_to_element.get(ptype, 'X')
            
            for i in range(self.max_counts[ptype]):
                rows.append({
                    'record_name': 'ATOM',
                    'atom_number': particle_idx,
                    'blank_1': '',
                    'atom_name': f'{element}{i:03d}',
                    'alt_loc': '',
                    'residue_name': f'T{ptype:02d}',
                    'blank_2': '',
                    'chain_id': 'A',
                    'residue_number': ptype + 1,
                    'code_for_insertion': '',
                    'blank_3': '',
                    'x_coord': self.parking_position[0],
                    'y_coord': self.parking_position[1],
                    'z_coord': self.parking_position[2],
                    'occupancy': 1.0,
                    'b_factor': 0.0,
                    'blank_4': '',
                    'segment_id': '',
                    'element_symbol': element,
                    'charge': ''
                })
                particle_idx += 1
        
        return pd.DataFrame(rows)
    
    def write_pdb(self, filename: str) -> None:
        """
        Write topology as PDB file (hand-crafted, no dependencies).
        
        Args:
            filename: Output PDB file path
        """
        if not self.max_counts:
            raise ValueError("Call analyze() first")
        
        filename = Path(filename)
        
        with open(filename, 'w') as f:
            f.write("TITLE     Padded trajectory topology\n")
            
            atom_idx = 1
            
            for ptype in sorted(self.max_counts.keys()):
                element = self.type_to_element.get(ptype, 'X')
                res_num = ptype + 1
                res_name = f'T{ptype:02d}'
                
                for i in range(self.max_counts[ptype]):
                    atom_name = f'{element}{i:03d}'
                    x, y, z = self.parking_position
                    
                    # PDB ATOM record format (strict column positions)
                    # Cols: 1-6 (ATOM), 7-11 (atom#), 13-16 (atom name),
                    # 18-20 (res name), 22 (chain), 23-26 (res#),
                    # 31-38 (x), 39-46 (y), 47-54 (z),
                    # 55-60 (occupancy), 61-66 (b-factor), 77-78 (element)
                    line = (
                        f"ATOM  {atom_idx:5d}  {atom_name:3s} "
                        f"{res_name:3s} A{res_num:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}"
                        f"  1.00  0.00           {element:>2s}\n"
                    )
                    f.write(line)
                    atom_idx += 1
            
            f.write("END\n")
        
        print(f"Wrote PDB topology to {filename}")
    
    def write_dcd(self, filename: str, topology_pdb: Optional[str] = None) -> None:
        """
        Write padded trajectory as DCD file.
        
        Args:
            filename: Output DCD file path
            topology_pdb: Optional PDB file path to write alongside
        """
        if not HAS_MDTRAJ:
            raise ImportError("mdtraj required for DCD output. Install: pip install mdtraj")
        
        # Build padded trajectory if not done
        if self.padded_trajectory is None:
            self.padded_trajectory = self._build_padded_trajectory()
        
        # Build topology directly in mdtraj
        top = md.Topology()
        chain = top.add_chain()
        
        for ptype in sorted(self.max_counts.keys()):
            element = self.type_to_element.get(ptype, 'X')
            # Convert element symbol to mdtraj element
            try:
                elem = md.core.element.Element.by_symbol(element)
            except KeyError:
                # Fallback to virtual site if element not recognized
                elem = md.core.element.virtual_site
            
            residue = top.add_residue(f'T{ptype:02d}', chain, resSeq=ptype+1)
            
            for i in range(self.max_counts[ptype]):
                atom_name = f'{element}{i:03d}'
                top.add_atom(atom_name, elem, residue)
        
        # Create trajectory object
        traj = md.Trajectory(
            self.padded_trajectory,
            topology=top,
            time=np.array(self.frames)  # Frame numbers as time
        )
        
        # Write DCD
        traj.save_dcd(filename)
        print(f"Wrote DCD trajectory to {filename}")
        print(f"  Frames: {len(traj)}")
        print(f"  Particles: {traj.n_atoms}")
        
        # Optionally write PDB topology file for reference
        if topology_pdb is not None:
            self.write_pdb(topology_pdb)
    
    def get_frame_data(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get padded positions and type array for a single frame.
        
        Args:
            frame_idx: Frame index (0-based)
        
        Returns:
            (positions, types) where positions is (n_particles, 3) and types is (n_particles,)
        """
        if self.padded_trajectory is None:
            self.padded_trajectory = self._build_padded_trajectory()
        
        positions = self.padded_trajectory[frame_idx]
        
        # Reconstruct type array
        types = np.zeros(self.total_particles, dtype=int)
        particle_idx = 0
        for ptype in sorted(self.max_counts.keys()):
            for _ in range(self.max_counts[ptype]):
                types[particle_idx] = ptype
                particle_idx += 1
        
        return positions, types
    
    def summary(self) -> str:
        """Return a text summary of the padding configuration."""
        if not self.max_counts:
            return "Not yet analyzed. Call analyze() first."
        
        summary_lines = [
            "=== Trajectory Padding Summary ===",
            f"Total frames: {self.n_frames}",
            f"Total padded size: {self.total_particles} particles",
            f"Parking position: {self.parking_position}",
            "",
            "Particle counts per type:"
        ]
        
        for ptype in sorted(self.max_counts.keys()):
            element = self.type_to_element.get(ptype, '?')
            count = self.max_counts[ptype]
            summary_lines.append(f"  Type {ptype} ({element}): {count} particles")
        
        return "\n".join(summary_lines)


if __name__ == "__main__":
    # Example usage
    print("Trajectory Padding Tool")
    print("See docstrings for usage examples.")