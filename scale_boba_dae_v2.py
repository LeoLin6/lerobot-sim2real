#!/usr/bin/env python3
import re

def scale_dae_file_text(input_path, output_path, scale_factor=0.108):
    """Scale DAE file by directly manipulating the text, preserving original structure"""
    
    with open(input_path, 'r') as f:
        content = f.read()
    
    # Find all position arrays and scale them
    def scale_position_array(match):
        # Extract the float array data
        array_id = match.group(1)
        count = match.group(2)
        data = match.group(3)
        
        # Parse and scale the values
        values = [float(x) for x in data.split()]
        scaled_values = []
        
        for i in range(0, len(values), 3):
            if i + 2 < len(values):
                # Scale X, Y, Z coordinates
                scaled_values.extend([
                    values[i] * scale_factor,      # X
                    values[i + 1] * scale_factor,  # Y
                    values[i + 2] * scale_factor   # Z
                ])
        
        # Format the scaled values back to string
        scaled_data = ' '.join([f"{x:.6f}" for x in scaled_values])
        
        return f'<float_array id="{array_id}" count="{count}">{scaled_data}</float_array>'
    
    # Use regex to find and scale all position arrays
    pattern = r'<float_array id="(pos_\d+-array)" count="(\d+)">([^<]+)</float_array>'
    scaled_content = re.sub(pattern, scale_position_array, content)
    
    # Write the scaled DAE file
    with open(output_path, 'w') as f:
        f.write(scaled_content)
    
    print(f"Scaled DAE file saved to: {output_path}")

def main():
    input_path = 'assets/Boba.dae'
    output_path = 'assets/Boba_scaled_v2.dae'
    scale_factor = 0.108  # Same factor used for OBJ file
    
    print(f"Scaling DAE file by factor: {scale_factor}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    try:
        scale_dae_file_text(input_path, output_path, scale_factor)
        print("✅ DAE file scaled successfully!")
    except Exception as e:
        print(f"❌ Error scaling DAE file: {e}")

if __name__ == "__main__":
    main() 