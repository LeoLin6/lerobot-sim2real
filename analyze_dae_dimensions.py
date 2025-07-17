#!/usr/bin/env python3
import re
import xml.etree.ElementTree as ET

def analyze_dae_file(filename):
    """Analyze DAE file to find bounding box from all position arrays"""
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
        
        # Find all position arrays
        y_coords = []
        
        # Look for all float_array elements that contain position data
        for float_array in root.findall('.//{http://www.collada.org/2005/11/COLLADASchema}float_array'):
            if 'pos' in float_array.get('id', ''):
                # Parse the float array data
                data = float_array.text.strip()
                values = [float(x) for x in data.split()]
                
                # Every 3rd value (index 1, 4, 7, ...) is the Y coordinate
                for i in range(1, len(values), 3):
                    if i < len(values):
                        y_coords.append(values[i])
        
        if y_coords:
            min_y = min(y_coords)
            max_y = max(y_coords)
            height = max_y - min_y
            return min_y, max_y, height
        return None, None, None
        
    except Exception as e:
        print(f"Error analyzing DAE file: {e}")
        return None, None, None

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

def main():
    print("Analyzing Boba model dimensions...")
    print("=" * 50)
    
    # Analyze DAE file
    print("Original DAE File Analysis:")
    min_y_dae, max_y_dae, height_dae = analyze_dae_file('assets/Boba.dae')
    if height_dae is not None:
        print(f"  Minimum Y coordinate: {min_y_dae:.3f}")
        print(f"  Maximum Y coordinate: {max_y_dae:.3f}")
        print(f"  Height: {height_dae:.3f} units")
        print(f"  Height in cm (assuming 1 unit = 1 cm): {height_dae:.3f} cm")
        print(f"  Height in mm: {height_dae*10:.1f} mm")
    else:
        print("  Could not determine dimensions from DAE file")
    
    # Analyze scaled DAE file
    print("\nScaled DAE File Analysis:")
    min_y_dae_scaled, max_y_dae_scaled, height_dae_scaled = analyze_dae_file('assets/Boba_scaled.dae')
    if height_dae_scaled is not None:
        print(f"  Minimum Y coordinate: {min_y_dae_scaled:.3f}")
        print(f"  Maximum Y coordinate: {max_y_dae_scaled:.3f}")
        print(f"  Height: {height_dae_scaled:.3f} units")
        print(f"  Height in cm (assuming 1 unit = 1 cm): {height_dae_scaled:.3f} cm")
        print(f"  Height in mm: {height_dae_scaled*10:.1f} mm")
    else:
        print("  Could not determine dimensions from scaled DAE file")
    
    # Analyze scaled DAE file v2
    print("\nScaled DAE File v2 Analysis:")
    min_y_dae_scaled_v2, max_y_dae_scaled_v2, height_dae_scaled_v2 = analyze_dae_file('assets/Boba_scaled_v2.dae')
    if height_dae_scaled_v2 is not None:
        print(f"  Minimum Y coordinate: {min_y_dae_scaled_v2:.3f}")
        print(f"  Maximum Y coordinate: {max_y_dae_scaled_v2:.3f}")
        print(f"  Height: {height_dae_scaled_v2:.3f} units")
        print(f"  Height in cm (assuming 1 unit = 1 cm): {height_dae_scaled_v2:.3f} cm")
        print(f"  Height in mm: {height_dae_scaled_v2*10:.1f} mm")
    else:
        print("  Could not determine dimensions from scaled DAE file v2")
    
    # Analyze scaled OBJ file
    print("\nScaled OBJ File Analysis:")
    min_y_obj, max_y_obj, height_obj = analyze_obj_file('assets/Boba_scaled.obj')
    if height_obj is not None:
        print(f"  Minimum Y coordinate: {min_y_obj:.3f}")
        print(f"  Maximum Y coordinate: {max_y_obj:.3f}")
        print(f"  Height: {height_obj:.3f} units")
        print(f"  Height in cm (assuming 1 unit = 1 cm): {height_obj:.3f} cm")
        print(f"  Height in mm: {height_obj*10:.1f} mm")
    else:
        print("  Could not determine dimensions from scaled OBJ file")
    
    print("\n" + "=" * 50)
    print("Comparison:")
    if height_dae_scaled is not None and height_obj is not None:
        print(f"Scaled DAE file height: {height_dae_scaled:.3f} units")
        print(f"Scaled OBJ file height: {height_obj:.3f} units")
        if abs(height_dae_scaled - height_obj) < 0.1:
            print("✅ Dimensions match! Both scaled files have similar heights.")
        else:
            print("❌ Dimensions don't match. The scaled files have different heights.")
            print(f"Difference: {abs(height_dae_scaled - height_obj):.3f} units")
    else:
        print("Could not compare dimensions due to analysis errors.")

if __name__ == "__main__":
    main() 