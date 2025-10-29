#!/usr/bin/env python3

import sys
import csv
import statistics
import math
from datetime import datetime

def parse_datetime(date_str):
    """Parse datetime string in multiple formats"""
    formats = [
        '%Y-%m-%d %H:%M:%S',  # 2016-07-01 02:00:00
        '%Y/%m/%d %H:%M',     # 1990/1/8 0:00
        '%Y-%m-%d',           # 2016-07-01
        '%Y/%m/%d',           # 1990/1/8
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse datetime: {date_str}")

def calculate_time_uniformity(csv_file):
    """
    Calculate time uniformity score for a time series CSV file.

    Returns a score between 0 and 1:
    - 1.0: Perfectly uniform time intervals
    - 0.0: Completely random time intervals

    Args:
        csv_file (str): Path to CSV file with datetime in first column

    Returns:
        dict: Contains uniformity score and statistics
    """
    timestamps = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header

        for row in reader:
            if row:  # Skip empty rows
                try:
                    timestamp = parse_datetime(row[0])
                    timestamps.append(timestamp)
                except (ValueError, IndexError):
                    continue

    if len(timestamps) < 2:
        return {
            'uniformity_score': 0.0,
            'total_points': len(timestamps),
            'error': 'Need at least 2 valid timestamps'
        }

    # Sort timestamps to ensure chronological order
    timestamps.sort()

    # Calculate time intervals in hours
    intervals = []
    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i-1]
        interval_hours = delta.total_seconds() / 3600.0
        intervals.append(interval_hours)

    if len(intervals) == 0:
        return {
            'uniformity_score': 0.0,
            'total_points': len(timestamps),
            'error': 'No valid intervals found'
        }

    # Calculate statistics
    mean_interval = statistics.mean(intervals)
    std_deviation = statistics.stdev(intervals) if len(intervals) > 1 else 0
    min_interval = min(intervals)
    max_interval = max(intervals)

    # Calculate coefficient of variation (CV)
    # CV = std_dev / mean, normalized measure of variability
    cv = std_deviation / mean_interval if mean_interval > 0 else float('inf')

    # Calculate uniformity score
    # For perfectly uniform data: CV = 0, score = 1
    # For very variable data: CV >> 1, score approaches 0
    # Using exponential decay: score = exp(-CV)
    uniformity_score = math.exp(-cv) if cv != float('inf') else 0.0

    # Alternative scoring method: based on relative standard deviation
    # Clamp to [0, 1] range
    uniformity_score = max(0.0, min(1.0, uniformity_score))

    return {
        'uniformity_score': uniformity_score,
        'total_points': len(timestamps),
        'total_intervals': len(intervals),
        'mean_interval_hours': mean_interval,
        'std_deviation_hours': std_deviation,
        'min_interval_hours': min_interval,
        'max_interval_hours': max_interval,
        'coefficient_of_variation': cv,
        'first_timestamp': timestamps[0].strftime('%Y-%m-%d %H:%M:%S'),
        'last_timestamp': timestamps[-1].strftime('%Y-%m-%d %H:%M:%S')
    }

def format_duration(hours):
    """Format duration in hours to human readable format"""
    if hours < 1:
        minutes = hours * 60
        return f"{minutes:.1f} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = hours / 24
        return f"{days:.1f} days"

def main():
    if len(sys.argv) != 2:
        print("Usage: python time_variance.py <csv_file>")
        print("Example: python time_variance.py electricity/electricity.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    try:
        result = calculate_time_uniformity(csv_file)

        print(f"Time Series Uniformity Analysis for: {csv_file}")
        print("=" * 60)

        if 'error' in result:
            print(f"Error: {result['error']}")
            print(f"Total points analyzed: {result['total_points']}")
            return

        print(f"Uniformity Score: {result['uniformity_score']:.4f}")
        print(f"  (1.0 = perfectly uniform, 0.0 = completely random)")
        print()

        print("Dataset Statistics:")
        print(f"  Total data points: {result['total_points']:,}")
        print(f"  Total intervals: {result['total_intervals']:,}")
        print(f"  Time range: {result['first_timestamp']} to {result['last_timestamp']}")
        print()

        print("Interval Statistics:")
        print(f"  Mean interval: {format_duration(result['mean_interval_hours'])}")
        print(f"  Standard deviation: {format_duration(result['std_deviation_hours'])}")
        print(f"  Min interval: {format_duration(result['min_interval_hours'])}")
        print(f"  Max interval: {format_duration(result['max_interval_hours'])}")
        print(f"  Coefficient of variation: {result['coefficient_of_variation']:.4f}")
        print()

        # Interpretation
        score = result['uniformity_score']
        if score > 0.9:
            interpretation = "Highly uniform (regular time series)"
        elif score > 0.7:
            interpretation = "Moderately uniform (some irregularity)"
        elif score > 0.3:
            interpretation = "Somewhat irregular (significant gaps)"
        else:
            interpretation = "Highly irregular (random/sparse sampling)"

        print(f"Interpretation: {interpretation}")

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
