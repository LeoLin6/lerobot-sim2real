#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import re

def scale_dae_file(input_path, output_path, scale_factor=0.108):
    """Scale all vertex coordinates in a DAE file by the given factor"""
    
    # Parse the DAE file
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    # Find all float_array elements that contain position data
    for float_array in root.findall('.//{http://www.collada.org/2005/11/COLLADASchema}float_array'):
        if 'pos' in float_array.get('id', ''):
            # Parse the float array data
            data = float_array.text.strip()
            values = [float(x) for x in data.split()]
            
            # Scale all coordinates (X, Y, Z)
            scaled_values = []
            for i in range(0, len(values), 3):
                if i + 2 < len(values):
                    # Scale X, Y, Z coordinates
                    scaled_values.extend([
                        values[i] * scale_factor,      # X
                        values[i + 1] * scale_factor,  # Y
                        values[i + 2] * scale_factor   # Z
                    ])
            
            # Update the float array with scaled values
            float_array.text = ' '.join([f"{x:.6f}" for x in scaled_values])
    
    # Write the scaled DAE file
    tree.write(output_path, encoding='UTF-8', xml_declaration=True)
    print(f"Scaled DAE file saved to: {output_path}")

def main():
    input_path = 'assets/Boba.dae'
    output_path = 'assets/Boba_scaled.dae'
    scale_factor = 0.108  # Same factor used for OBJ file
    
    print(f"Scaling DAE file by factor: {scale_factor}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    try:
        scale_dae_file(input_path, output_path, scale_factor)
        print("✅ DAE file scaled successfully!")
    except Exception as e:
        print(f"❌ Error scaling DAE file: {e}")

if __name__ == "__main__":
    main() 