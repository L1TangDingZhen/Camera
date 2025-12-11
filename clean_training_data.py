#!/usr/bin/env python3
"""
Training Data Cleaning Tool

Features:
1. View data statistics
2. Delete data by time range
3. Filter data by quality
4. Visualize data distribution
5. Backup data
"""

import os
import json
import argparse
import shutil
from datetime import datetime
from typing import List, Dict
import numpy as np


class TrainingDataCleaner:
    """Training data cleaning tool"""

    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = data_dir
        self.labels = ['sitting', 'standing', 'lying']
        self.data = {}

    def load_data(self):
        """Load all training data"""
        print("\n" + "=" * 60)
        print("Loading training data...")
        print("=" * 60)

        for label in self.labels:
            filepath = os.path.join(self.data_dir, f"{label}_samples.json")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.data[label] = json.load(f)
                print(f"✓ {label:10s}: {len(self.data[label]):5d} samples")
            else:
                self.data[label] = []
                print(f"✗ {label:10s}: File not found")

        print("=" * 60)

    def show_statistics(self):
        """Display data statistics"""
        print("\n" + "=" * 60)
        print("Data Statistics")
        print("=" * 60)

        total = 0
        for label in self.labels:
            samples = self.data[label]
            total += len(samples)

            if len(samples) == 0:
                print(f"\n{label.upper()}:")
                print("  No data")
                continue

            # Time range
            timestamps = [s['timestamp'] for s in samples if 'timestamp' in s]
            if timestamps:
                # Handle both Unix timestamp (float) and ISO string formats
                times = []
                for t in timestamps:
                    if isinstance(t, (int, float)):
                        times.append(datetime.fromtimestamp(t))
                    else:
                        times.append(datetime.fromisoformat(t))
                earliest = min(times)
                latest = max(times)
                print(f"\n{label.upper()}:")
                print(f"  Sample count: {len(samples)}")
                print(f"  Time range: {earliest.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"              to {latest.strftime('%Y-%m-%d %H:%M:%S')}")

                # Statistics by date
                date_counts = {}
                for t in times:
                    date = t.strftime('%Y-%m-%d')
                    date_counts[date] = date_counts.get(date, 0) + 1

                print(f"  Recording dates:")
                for date in sorted(date_counts.keys()):
                    print(f"    {date}: {date_counts[date]:4d} samples")
            else:
                print(f"\n{label.upper()}:")
                print(f"  Sample count: {len(samples)}")
                print(f"  No timestamp information")

        print(f"\nTotal: {total} samples")
        print("=" * 60)

    def backup_data(self, backup_name: str = None):
        """Backup current data"""
        if backup_name is None:
            backup_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        backup_dir = os.path.join(self.data_dir, f"backup_{backup_name}")

        if os.path.exists(backup_dir):
            print(f"❌ Backup directory already exists: {backup_dir}")
            response = input("Overwrite? (yes/no): ").strip().lower()
            if response != 'yes':
                print("Backup cancelled")
                return False
            shutil.rmtree(backup_dir)

        os.makedirs(backup_dir, exist_ok=True)

        print(f"\nBacking up to: {backup_dir}")
        for label in self.labels:
            src = os.path.join(self.data_dir, f"{label}_samples.json")
            if os.path.exists(src):
                dst = os.path.join(backup_dir, f"{label}_samples.json")
                shutil.copy2(src, dst)
                print(f"  ✓ {label}_samples.json")

        print(f"✅ Backup completed: {backup_dir}")
        return True

    def delete_by_time_range(self, label: str, start_time: str = None, end_time: str = None):
        """Delete data by time range

        Args:
            label: Data label (sitting/standing/lying)
            start_time: Start time (YYYY-MM-DD HH:MM:SS or YYYY-MM-DD)
            end_time: End time (YYYY-MM-DD HH:MM:SS or YYYY-MM-DD)
        """
        if label not in self.data or len(self.data[label]) == 0:
            print(f"❌ {label} has no data")
            return

        samples = self.data[label]
        original_count = len(samples)

        # Parse time
        def parse_time(time_str):
            if time_str is None:
                return None
            # Try both formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                try:
                    return datetime.strptime(time_str, fmt)
                except:
                    continue
            # Try ISO format
            try:
                return datetime.fromisoformat(time_str)
            except:
                return None

        start_dt = parse_time(start_time) if start_time else None
        end_dt = parse_time(end_time) if end_time else None

        # Filter data
        filtered = []
        deleted_count = 0

        for sample in samples:
            if 'timestamp' not in sample:
                filtered.append(sample)  # Keep samples without timestamp
                continue

            # Handle both Unix timestamp and ISO string formats
            t = sample['timestamp']
            if isinstance(t, (int, float)):
                sample_time = datetime.fromtimestamp(t)
            else:
                sample_time = datetime.fromisoformat(t)

            should_delete = False
            if start_dt and end_dt:
                if start_dt <= sample_time <= end_dt:
                    should_delete = True
            elif start_dt:
                if sample_time >= start_dt:
                    should_delete = True
            elif end_dt:
                if sample_time <= end_dt:
                    should_delete = True

            if should_delete:
                deleted_count += 1
            else:
                filtered.append(sample)

        self.data[label] = filtered

        print(f"\n{label.upper()}:")
        print(f"  Original samples: {original_count}")
        print(f"  Deleted samples: {deleted_count}")
        print(f"  Remaining samples: {len(filtered)}")

        return deleted_count

    def delete_by_date(self, label: str, date: str):
        """Delete data from a specific date

        Args:
            label: Data label
            date: Date (YYYY-MM-DD)
        """
        start_time = f"{date} 00:00:00"
        end_time = f"{date} 23:59:59"
        return self.delete_by_time_range(label, start_time, end_time)

    def delete_samples_without_timestamp(self, label: str):
        """Delete all samples without timestamp

        Args:
            label: Data label
        """
        if label not in self.data or len(self.data[label]) == 0:
            print(f"❌ {label} has no data")
            return

        samples = self.data[label]
        original_count = len(samples)

        # Keep only samples with timestamp
        filtered = [s for s in samples if 'timestamp' in s]
        deleted_count = original_count - len(filtered)

        self.data[label] = filtered

        print(f"\n{label.upper()}:")
        print(f"  Original samples: {original_count}")
        print(f"  Deleted samples (no timestamp): {deleted_count}")
        print(f"  Remaining samples: {len(filtered)}")

        return deleted_count

    def keep_recent_n_samples(self, label: str, n: int):
        """Keep only the most recent N samples

        Args:
            label: Data label
            n: Number of samples to keep
        """
        if label not in self.data or len(self.data[label]) == 0:
            print(f"❌ {label} has no data")
            return

        samples = self.data[label]
        original_count = len(samples)

        if original_count <= n:
            print(f"{label}: Current sample count {original_count} <= {n}, no deletion needed")
            return

        # Sort by timestamp (newest last)
        samples_with_time = []
        for s in samples:
            if 'timestamp' in s:
                t = s['timestamp']
                if isinstance(t, (int, float)):
                    dt = datetime.fromtimestamp(t)
                else:
                    dt = datetime.fromisoformat(t)
                samples_with_time.append((s, dt))
        samples_without_time = [s for s in samples if 'timestamp' not in s]

        # Sort and keep the most recent N samples
        samples_with_time.sort(key=lambda x: x[1])
        kept = [s for s, _ in samples_with_time[-n:]] + samples_without_time

        self.data[label] = kept
        deleted = original_count - len(kept)

        print(f"\n{label.upper()}:")
        print(f"  Original samples: {original_count}")
        print(f"  Deleted samples: {deleted}")
        print(f"  Remaining samples: {len(kept)}")

    def save_data(self):
        """Save data"""
        print("\n" + "=" * 60)
        print("Saving data...")
        print("=" * 60)

        for label in self.labels:
            filepath = os.path.join(self.data_dir, f"{label}_samples.json")
            with open(filepath, 'w') as f:
                json.dump(self.data[label], f)
            print(f"✓ {label:10s}: {len(self.data[label]):5d} samples → {filepath}")

        print("=" * 60)
        print("✅ Save completed")

    def interactive_clean(self):
        """Interactive cleaning"""
        self.load_data()
        self.show_statistics()

        while True:
            print("\n" + "=" * 60)
            print("Cleaning Operations Menu")
            print("=" * 60)
            print("1. View statistics")
            print("2. Backup current data")
            print("3. Delete data from specific date")
            print("4. Delete data from time range")
            print("5. Delete samples without timestamp")
            print("6. Keep only recent N samples")
            print("7. Save and exit")
            print("8. Exit without saving")
            print("=" * 60)

            choice = input("Select operation (1-8): ").strip()

            if choice == '1':
                self.show_statistics()

            elif choice == '2':
                backup_name = input("Backup name (leave empty for timestamp): ").strip()
                if backup_name == '':
                    backup_name = None
                self.backup_data(backup_name)

            elif choice == '3':
                label = input("Label (sitting/standing/lying): ").strip()
                if label not in self.labels:
                    print("❌ Invalid label")
                    continue
                date = input("Date (YYYY-MM-DD): ").strip()

                # Preview
                print(f"\nWill delete all {label} data from {date}")
                confirm = input("Confirm deletion? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    self.delete_by_date(label, date)
                else:
                    print("Cancelled")

            elif choice == '4':
                label = input("Label (sitting/standing/lying): ").strip()
                if label not in self.labels:
                    print("❌ Invalid label")
                    continue
                start = input("Start time (YYYY-MM-DD HH:MM:SS or YYYY-MM-DD, empty for no limit): ").strip()
                end = input("End time (YYYY-MM-DD HH:MM:SS or YYYY-MM-DD, empty for no limit): ").strip()

                if start == '':
                    start = None
                if end == '':
                    end = None

                print(f"\nWill delete {label} data from {start or 'beginning'} to {end or 'now'}")
                confirm = input("Confirm deletion? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    self.delete_by_time_range(label, start, end)
                else:
                    print("Cancelled")

            elif choice == '5':
                label = input("Label (sitting/standing/lying): ").strip()
                if label not in self.labels:
                    print("❌ Invalid label")
                    continue

                confirm = input(f"Will delete all {label} samples without timestamp, confirm? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    self.delete_samples_without_timestamp(label)
                else:
                    print("Cancelled")

            elif choice == '6':
                label = input("Label (sitting/standing/lying): ").strip()
                if label not in self.labels:
                    print("❌ Invalid label")
                    continue
                n = input("Number of samples to keep: ").strip()
                try:
                    n = int(n)
                    confirm = input(f"Will keep only the most recent {n} samples of {label}, confirm? (yes/no): ").strip().lower()
                    if confirm == 'yes':
                        self.keep_recent_n_samples(label, n)
                    else:
                        print("Cancelled")
                except ValueError:
                    print("❌ Invalid number")

            elif choice == '7':
                confirm = input("\nConfirm saving changes? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    self.save_data()
                    break
                else:
                    print("Save cancelled")

            elif choice == '8':
                confirm = input("\nConfirm exit without saving? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    print("Exited without saving changes")
                    break
                else:
                    print("Continue operations")

            else:
                print("❌ Invalid selection")


def main():
    parser = argparse.ArgumentParser(description='Training data cleaning tool')
    parser.add_argument('--data-dir', default='training_data', help='Data directory')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')

    # Quick operations
    parser.add_argument('--backup', metavar='NAME', help='Backup data')
    parser.add_argument('--stats', action='store_true', help='Display statistics')
    parser.add_argument('--label', choices=['sitting', 'standing', 'lying'], help='Label to operate on')
    parser.add_argument('--delete-date', metavar='DATE', help='Delete data from specific date (YYYY-MM-DD)')
    parser.add_argument('--delete-start', metavar='TIME', help='Delete start time (YYYY-MM-DD)')
    parser.add_argument('--delete-end', metavar='TIME', help='Delete end time (YYYY-MM-DD)')
    parser.add_argument('--delete-no-timestamp', action='store_true', help='Delete samples without timestamp')
    parser.add_argument('--keep-recent', type=int, metavar='N', help='Keep only the most recent N samples')
    parser.add_argument('--save', action='store_true', help='Save changes')

    args = parser.parse_args()

    cleaner = TrainingDataCleaner(args.data_dir)

    # Interactive mode
    if args.interactive or len(vars(args)) == 1:  # Default to interactive mode if no arguments
        cleaner.interactive_clean()
        return

    # Command line mode
    cleaner.load_data()

    modified = False

    if args.backup:
        cleaner.backup_data(args.backup)

    if args.stats:
        cleaner.show_statistics()

    if args.delete_date and args.label:
        cleaner.delete_by_date(args.label, args.delete_date)
        modified = True

    if (args.delete_start or args.delete_end) and args.label:
        cleaner.delete_by_time_range(args.label, args.delete_start, args.delete_end)
        modified = True

    if args.delete_no_timestamp and args.label:
        cleaner.delete_samples_without_timestamp(args.label)
        modified = True

    if args.keep_recent and args.label:
        cleaner.keep_recent_n_samples(args.label, args.keep_recent)
        modified = True

    if modified and args.save:
        cleaner.save_data()
    elif modified:
        print("\n⚠️  Changes not saved! Use --save parameter to save changes")


if __name__ == '__main__':
    main()
