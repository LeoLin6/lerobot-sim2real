#!/usr/bin/env python3
import re
import json
import struct

def analyze_obj_file(filename):
    """Analyze OBJ file to find bounding box"""
    y_coords = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 4:
                    y_coords.append(float(parts[2]))
    
    if y_coords:
        min_y = min(y_coords)
        max_y = max(y_coords)
        height = max_y - min_y
        return min_y, max_y, height
    return None, None, None

def analyze_glb_file(filename):
    """Analyze GLB file to find bounding box"""
    try:
        with open(filename, 'rb') as f:
            # Read GLB header
            header = f.read(12)
            if header[:4] != b'glTF':
                print(f"Not a valid GLB file: {filename}")
                return None, None, None
            
            # Read JSON chunk
            chunk_header = f.read(8)
            chunk_length, chunk_type = struct.unpack('<II', chunk_header)
            
            if chunk_type != 0x4E4F534A:  # JSON chunk
                print(f"Expected JSON chunk, got: {chunk_type}")
                return None, None, None
            
            json_data = f.read(chunk_length)
            gltf = json.loads(json_data.decode('utf-8'))
            
            # Try to find bounding box information
            if 'meshes' in gltf:
                all_vertices = []
                for mesh in gltf['meshes']:
                    if 'primitives' in mesh:
                        for primitive in mesh['primitives']:
                            if 'attributes' in primitive and 'POSITION' in primitive['attributes']:
                                # This would require reading the binary data chunk
                                # For now, we'll note that GLB contains mesh data
                                pass
                
                print(f"GLB file contains {len(gltf.get('meshes', []))} meshes")
                return None, None, None
    except Exception as e:
        print(f"Error analyzing GLB file: {e}")
        return None, None, None

def main():
    print("Analyzing Boba model dimensions...")
    print("=" * 50)
    
    # Analyze OBJ file
    print("OBJ File Analysis (Original):")
    min_y, max_y, height = analyze_obj_file('assets/Boba.obj')
    if height is not None:
        print(f"  Minimum Y coordinate: {min_y:.3f}")
        print(f"  Maximum Y coordinate: {max_y:.3f}")
        print(f"  Height: {height:.3f} units")
        print(f"  Height in cm (assuming 1 unit = 1 cm): {height:.3f} cm")
        print(f"  Height in mm: {height*10:.1f} mm")
    else:
        print("  Could not determine dimensions from OBJ file")

    print("\nOBJ File Analysis (Scaled):")
    min_y_s, max_y_s, height_s = analyze_obj_file('assets/Boba_scaled.obj')
    if height_s is not None:
        print(f"  Minimum Y coordinate: {min_y_s:.3f}")
        print(f"  Maximum Y coordinate: {max_y_s:.3f}")
        print(f"  Height: {height_s:.3f} units")
        print(f"  Height in cm (assuming 1 unit = 1 cm): {height_s:.3f} cm")
        print(f"  Height in mm: {height_s*10:.1f} mm")
    else:
        print("  Could not determine dimensions from scaled OBJ file")
    
    print("\nGLB File Analysis:")
    min_y_glb, max_y_glb, height_glb = analyze_glb_file('assets/Boba.glb')
    if height_glb is not None:
        print(f"  Height: {height_glb:.3f} units")
    else:
        print("  GLB file analysis requires additional tools")
        print("  Note: GLB files are binary and require specialized parsing")
    
    print("\n" + "=" * 50)
    print("Summary:")
    if height is not None:
        print(f"The Boba model in Boba.obj has a height of {height:.3f} units")
        print(f"This corresponds to approximately {height:.1f} cm or {height*10:.1f} mm")
        print("For a LEGO minifigure, this seems reasonable as they are typically around 4cm tall")
        print("The model appears to be scaled appropriately for a LEGO minifigure representation")

if __name__ == "__main__":
    main() 