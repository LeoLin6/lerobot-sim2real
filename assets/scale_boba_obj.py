scale_factor = 0.108  # 4cm / 37.1cm
input_path = 'Boba.obj'
output_path = 'Boba_scaled.obj'

with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
    for line in fin:
        if line.startswith('v '):
            parts = line.strip().split()
            x, y, z = [float(parts[i]) * scale_factor for i in range(1, 4)]
            fout.write(f'v {x} {y} {z}\n')
        else:
            fout.write(line) 