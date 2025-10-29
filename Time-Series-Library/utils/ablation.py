#!/usr/bin/env python3

import sys
import random
import csv

def ablate_csv(input_file, removal_percentage):
    if not (0.0 <= removal_percentage <= 1.0):
        raise ValueError("Removal percentage must be between 0.0 and 1.0")

    ablation_rate = round(removal_percentage * 100)
    output_file = input_file.replace('.csv', f'_ablated_by_{ablation_rate}.csv')

    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        data_lines = list(reader)

    total_lines = len(data_lines)
    lines_to_keep = []

    for line in data_lines:
        if random.random() > removal_percentage:
            lines_to_keep.append(line)

    kept_count = len(lines_to_keep)
    removed_count = total_lines - kept_count
    actual_removal_rate = removed_count / total_lines if total_lines > 0 else 0

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(lines_to_keep)

    print(f"Original lines: {total_lines}")
    print(f"Lines kept: {kept_count}")
    print(f"Lines removed: {removed_count}")
    print(f"Actual removal rate: {actual_removal_rate:.2%}")
    print(f"Output written to: {output_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python ablation.py <csv_file> <removal_percentage>")
        print("Example: python ablation.py electricity/electricity.csv 0.75")
        sys.exit(1)

    input_file = sys.argv[1]
    try:
        removal_percentage = float(sys.argv[2])
    except ValueError:
        print("Error: Removal percentage must be a number")
        sys.exit(1)

    try:
        ablate_csv(input_file, removal_percentage)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
