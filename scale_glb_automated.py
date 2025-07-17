#!/usr/bin/env python3
import json
import struct
from pygltflib import GLTF2

def scale_glb_file_automated(input_path, output_path, scale_factor=0.001077):
    """Scale GLB file by modifying the node transforms"""
    
    # Load the GLB file
    gltf = GLTF2().load(input_path)
    
    print(f"Loaded GLB file: {input_path}")
    print(f"Number of nodes: {len(gltf.nodes)}")
    print(f"Number of meshes: {len(gltf.meshes)}")
    
    # Apply scale to all nodes
    for i, node in enumerate(gltf.nodes):
        print(f"Processing node {i}: {node.name if hasattr(node, 'name') else 'unnamed'}")
        
        # Initialize scale if it doesn't exist
        if not hasattr(node, 'scale') or node.scale is None:
            node.scale = [scale_factor, scale_factor, scale_factor]
        else:
            # Apply scale to existing scale
            node.scale = [s * scale_factor for s in node.scale]
        
        # Initialize translation if it doesn't exist
        if not hasattr(node, 'translation') or node.translation is None:
            node.translation = [0.0, 0.0, 0.0]
    
    # Save the scaled GLB file
    gltf.save(output_path)
    print(f"Scaled GLB saved to: {output_path}")
    print(f"Scale factor applied: {scale_factor}")

def main():
    input_path = 'assets/Boba.glb'
    output_path = 'assets/Boba_meters.glb'
    scale_factor = 0.001077  # Scale to make it 4cm tall
    
    print(f"Scaling GLB file by factor: {scale_factor}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    try:
        scale_glb_file_automated(input_path, output_path, scale_factor)
        print("✅ GLB file scaled successfully!")
    except Exception as e:
        print(f"❌ Error scaling GLB file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 