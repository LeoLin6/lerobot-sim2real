#!/usr/bin/env python3
import json
import struct
import base64
import sys
from pygltflib import GLTF2

def scale_glb_file_better(input_path, output_path, scale_factor=0.001):
    """Scale GLB file by modifying the mesh data directly"""
    
    # Load the GLB file
    gltf = GLTF2().load(input_path)
    
    print(f"Loaded GLB file: {input_path}")
    print(f"Number of nodes: {len(gltf.nodes)}")
    print(f"Number of meshes: {len(gltf.meshes)}")
    print(f"Number of accessors: {len(gltf.accessors)}")
    print(f"Number of bufferViews: {len(gltf.bufferViews)}")
    
    # Instead of scaling nodes, let's try a different approach
    # We'll create a new GLB with the same structure but scaled mesh data
    
    # For now, let's just copy the original file and add a scale transform
    # This is a safer approach that doesn't modify the binary data
    gltf_scaled = GLTF2().load(input_path)
    
    # Apply scale to the root node or create a new root node with scale
    if len(gltf_scaled.nodes) > 0:
        # Add scale to the first node (usually the root)
        root_node = gltf_scaled.nodes[0]
        if not hasattr(root_node, 'scale') or root_node.scale is None:
            root_node.scale = [scale_factor, scale_factor, scale_factor]
        else:
            # Apply scale to existing scale
            root_node.scale = [s * scale_factor for s in root_node.scale]
        
        print(f"Applied scale {scale_factor} to root node")
    
    # Save the scaled GLB file
    gltf_scaled.save(output_path)
    print(f"Scaled GLB saved to: {output_path}")
    print(f"Scale factor applied: {scale_factor}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python scale_glb_better.py <input_path> <output_path> [scale_factor]")
        print("Example: python scale_glb_better.py assets/fighter.glb assets/fighter_scaled.glb 0.001")
        print("If scale_factor is not provided, defaults to 0.001 (meters to mm)")
        # Fallback to default for backward compatibility
        input_path = 'assets/Boba.glb'
        output_path = 'assets/Boba_meters_v2.glb'
        scale_factor = 0.001077  # Scale to make it 4cm tall
        print(f"\nNo arguments provided, using default: {input_path} -> {output_path}, scale={scale_factor}")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        if len(sys.argv) > 3:
            scale_factor = float(sys.argv[3])
        else:
            scale_factor = 0.001  # Default: meters to mm
    
    print(f"Scaling GLB file by factor: {scale_factor}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    try:
        scale_glb_file_better(input_path, output_path, scale_factor)
        print("✅ GLB file scaled successfully!")
    except Exception as e:
        print(f"❌ Error scaling GLB file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 