import os
import wfdb
from collections import Counter, defaultdict


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) % 3600 // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def main():
    dataset_dir = "dataset"
    records_file = os.path.join(dataset_dir, "RECORDS")

    with open(records_file, 'r') as f:
        record_names = [line.strip() for line in f.readlines()]

    total_duration = 0
    durations = []
    sampling_freqs = Counter()
    channel_combinations = Counter()
    channel_names = Counter()

    for record in record_names:
        record_path = os.path.join(dataset_dir, record)
        try:
            header = wfdb.rdheader(record_path)
            duration = header.sig_len / header.fs
            total_duration += duration
            durations.append(duration)

            sampling_freqs[header.fs] += 1
            combination = tuple(header.sig_name)
            channel_combinations[combination] += 1

            for ch in header.sig_name:
                channel_names[ch] += 1

        except Exception as e:
            print(f"Error reading {record_path}: {e}")

    print(f"Total Recordings: {len(record_names)}")
    print(f"Total Duration: {format_duration(total_duration)}")
    avg_duration = total_duration / len(record_names)
    print(f"Average Duration: {format_duration(avg_duration)}")
    print(f"Min Duration: {format_duration(min(durations))}")
    print(f"Max Duration: {format_duration(max(durations))}\n")

    print("Sampling Frequencies:")
    for fs, count in sampling_freqs.items():
        print(f"- {fs} Hz ({count} recordings)")

    print("\nSignal Channels Used:")
    for ch, count in channel_names.items():
        print(f"- {ch} ({count} recordings)")

    print("\nCommon Channel Combinations:")
    for combo, count in channel_combinations.items():
        print(f"- {', '.join(combo)} ({count} recordings)")


if __name__ == "__main__":
    main()
