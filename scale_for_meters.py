#!/usr/bin/env python3
import re
import sys


def scale_obj_by_factor(input_path, output_path, scale_factor):
    """Scale OBJ file by a given factor (e.g., 1000 for meters to mm)"""
    print(f"Scaling all vertices by factor {scale_factor}...")
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = [float(parts[i]) * scale_factor for i in range(1, 4)]
                fout.write(f'v {x} {y} {z}\n')
            else:
                fout.write(line)
    print(f"Scaled OBJ saved to: {output_path}")

def main():
    if len(sys.argv) < 4:
        print("Usage: python scale_for_meters.py <input_path> <output_path> --factor <scale_factor>")
        print("Example: python scale_for_meters.py assets/fighter.obj assets/fighter_mm.obj --factor 0.001")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == '--factor':
        if len(sys.argv) < 5:
            print("Error: --factor requires a scale factor argument.")
            return
        scale_factor = float(sys.argv[4])
        scale_obj_by_factor(input_path, output_path, scale_factor)
    else:
        print("Error: Only --factor mode is supported.")
        print("Usage: python scale_for_meters.py <input_path> <output_path> --factor <scale_factor>")
        return
    print("✅ Done!")

if __name__ == "__main__":
    main() 