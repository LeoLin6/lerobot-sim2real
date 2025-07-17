#!/usr/bin/env python3
import sys
import json
import struct

def scale_glb_by_factor(input_path, output_path, scale_factor):
    """Scale GLB file by a given factor (e.g., 1000 for meters to mm)"""
    print(f"Scaling GLB file by factor {scale_factor}...")
    
    with open(input_path, 'rb') as f:
        # Read GLB header (12 bytes)
        header = f.read(12)
        magic, version, length = struct.unpack('<4sII', header)
        
        if magic != b'glTF':
            print("Error: Not a valid GLB file!")
            return
        
        # Read JSON chunk
        chunk_length = struct.unpack('<I', f.read(4))[0]
        chunk_type = f.read(4)
        json_data = f.read(chunk_length).decode('utf-8')
        
        # Parse JSON
        gltf = json.loads(json_data)
        
        # Scale all mesh positions
        for mesh in gltf.get('meshes', []):
            for primitive in mesh.get('primitives', []):
                if 'attributes' in primitive and 'POSITION' in primitive['attributes']:
                    accessor_index = primitive['attributes']['POSITION']
                    accessor = gltf['accessors'][accessor_index]
                    buffer_view_index = accessor['bufferView']
                    buffer_view = gltf['bufferViews'][buffer_view_index]
                    
                    # Read binary data
                    f.seek(20 + chunk_length + buffer_view['byteOffset'])
                    binary_data = f.read(buffer_view['byteLength'])
                    
                    # Scale positions
                    positions = []
                    for i in range(0, len(binary_data), 12):  # 3 floats * 4 bytes each
                        x, y, z = struct.unpack('<fff', binary_data[i:i+12])
                        positions.extend([x * scale_factor, y * scale_factor, z * scale_factor])
                    
                    # Update binary data
                    scaled_binary = struct.pack('<%df' % len(positions), *positions)
                    
                    # Update buffer view length
                    buffer_view['byteLength'] = len(scaled_binary)
                    
                    # Update accessor min/max if present
                    if 'min' in accessor:
                        accessor['min'] = [x * scale_factor for x in accessor['min']]
                    if 'max' in accessor:
                        accessor['max'] = [x * scale_factor for x in accessor['max']]
        
        # Write scaled GLB
        with open(output_path, 'wb') as out_f:
            # Write header
            out_f.write(header)
            
            # Write JSON chunk
            json_str = json.dumps(gltf, separators=(',', ':'))
            json_bytes = json_str.encode('utf-8')
            json_padding = (4 - (len(json_bytes) % 4)) % 4
            json_bytes += b' ' * json_padding
            
            out_f.write(struct.pack('<I', len(json_bytes)))
            out_f.write(b'JSON')
            out_f.write(json_bytes)
            
            # Write binary chunk (if any)
            f.seek(20 + chunk_length)
            remaining_data = f.read()
            if remaining_data:
                out_f.write(struct.pack('<I', len(remaining_data)))
                out_f.write(b'BIN ')
                out_f.write(remaining_data)
    
    print(f"Scaled GLB saved to: {output_path}")

def main():
    if len(sys.argv) < 4:
        print("Usage: python scale_glb_by_factor.py <input_path> <output_path> <scale_factor>")
        print("Example: python scale_glb_by_factor.py assets/fighter.glb assets/fighter_scaled.glb 0.001")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    scale_factor = float(sys.argv[3])
    
    scale_glb_by_factor(input_path, output_path, scale_factor)
    print("✅ Done!")

if __name__ == "__main__":
    main() 