import json
import pandas as pd
from pathlib import Path

def analyze_bet_statistics_by_file():
    # Load all three JSON files separately
    files = [
        "bet_analysis_results.json",
        "bet_analysis_results_private_informed.json",
        "bet_analysis_results_public.json"
    ]

    # Store data by file
    file_data = {}

    # Process each file separately
    for file_path in files:
        try:
            with open(Path(file_path), 'r') as f:
                data = json.load(f)
                print(f"\nLoaded {len(data)} entries from {file_path}")

                # Process this file's data
                rows = []
                for entry in data:
                    # Skip entries with errors
                    if 'analysis' not in entry or 'error' in entry['analysis']:
                        continue

                    # Extract key fields
                    row = {
                        'debate_id': entry['debate_id'],
                        'motion': entry['motion'],
                        'side': entry['side'],
                        'speech_type': entry['speech_type'],
                        'bet_amount': entry['raw_bet_amount']
                    }

                    # Add betting alignment data
                    alignment = entry['analysis']['betting_alignment']
                    row['internal_confidence'] = alignment['internal_confidence']
                    row['assessment'] = alignment['assessment']
                    row['degree'] = alignment['degree']

                    # Add strategic betting data
                    strategic = entry['analysis']['strategic_betting']
                    row['strategic_present'] = strategic['present']

                    # Try to convert internal confidence to a numeric value for calculations
                    try:
                        # Handle ranges like "60-65"
                        if '-' in str(alignment['internal_confidence']):
                            low, high = map(float, alignment['internal_confidence'].replace('%', '').split('-'))
                            row['confidence_numeric'] = (low + high) / 2
                        else:
                            # Handle single values
                            row['confidence_numeric'] = float(alignment['internal_confidence'].replace('%', ''))
                    except:
                        row['confidence_numeric'] = None

                    rows.append(row)

                # Create DataFrame for this file
                file_df = pd.DataFrame(rows)
                file_data[file_path] = file_df

                # Print file-specific stats
                print(f"\n=== STATISTICS FOR {file_path} ===")
                print(f"Valid entries: {len(file_df)}")
                print(f"Unique debates: {file_df['debate_id'].nunique()}")

                # Alignment distribution
                print("\nBETTING ALIGNMENT:")
                alignment_counts = file_df['assessment'].value_counts()
                alignment_pct = (alignment_counts / len(file_df) * 100).round(1)
                for assessment, count in alignment_counts.items():
                    pct = alignment_pct[assessment]
                    print(f"{assessment}: {count} ({pct}%)")

                # Strategic betting
                print("\nSTRATEGIC BETTING:")
                strategic_counts = file_df['strategic_present'].value_counts()
                strategic_pct = (strategic_counts / len(file_df) * 100).round(1)
                for strategy, count in strategic_counts.items():
                    pct = strategic_pct[strategy]
                    print(f"{strategy}: {count} ({pct}%)")

                # By side
                print("\nALIGNMENT BY SIDE:")
                side_alignment = pd.crosstab(file_df['side'], file_df['assessment'], normalize='index') * 100
                print(side_alignment.round(1))

                # By speech type
                print("\nALIGNMENT BY SPEECH TYPE:")
                speech_alignment = pd.crosstab(file_df['speech_type'], file_df['assessment'], normalize='index') * 100
                print(speech_alignment.round(1))

                # Bet amount and confidence
                print("\nBET AMOUNT AND CONFIDENCE:")
                bet_stats = file_df.agg({
                    'bet_amount': ['mean', 'median'],
                    'confidence_numeric': ['mean', 'median']
                }).round(1)
                print(bet_stats)

                # Degree of misalignment
                print("\nDEGREE OF MISALIGNMENT:")
                degree_counts = file_df['degree'].value_counts()
                degree_pct = (degree_counts / len(file_df) * 100).round(1)
                for degree, count in degree_counts.items():
                    pct = degree_pct[degree]
                    print(f"{degree}: {count} ({pct}%)")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Now print comparative stats between files
    print("\n\n=== COMPARATIVE STATISTICS BETWEEN FILES ===")

    # Create comparison dataframes
    comparison_stats = []
    for file_path, df in file_data.items():
        stats = {
            'file': file_path,
            'entries': len(df),
            'avg_bet': df['bet_amount'].mean().round(1),
            'avg_confidence': df['confidence_numeric'].mean().round(1),
            'aligned_pct': (df['assessment'] == 'Aligned').mean() * 100,
            'overbet_pct': (df['assessment'] == 'Overbetting').mean() * 100,
            'underbet_pct': (df['assessment'] == 'Underbetting').mean() * 100,
            'strategic_pct': (df['strategic_present'] == 'Yes').mean() * 100
        }
        comparison_stats.append(stats)

    comparison_df = pd.DataFrame(comparison_stats)
    print("\nKEY METRICS COMPARISON:")
    print(comparison_df.set_index('file').round(1))

    # Compare alignment patterns
    print("\nALIGNMENT PATTERN COMPARISON:")
    alignment_comparison = pd.DataFrame({
        file_path: df['assessment'].value_counts(normalize=True) * 100
        for file_path, df in file_data.items()
    }).round(1)
    print(alignment_comparison)

    # Compare strategic betting
    print("\nSTRATEGIC BETTING COMPARISON:")
    strategic_comparison = pd.DataFrame({
        file_path: df['strategic_present'].value_counts(normalize=True) * 100
        for file_path, df in file_data.items()
    }).fillna(0).round(1)
    print(strategic_comparison)

    # Compare by side
    print("\nPROPOSITION VS OPPOSITION COMPARISON:")
    for side in ['proposition', 'opposition']:
        print(f"\n{side.upper()}:")
        side_comparison = pd.DataFrame({
            file_path: df[df['side'] == side]['assessment'].value_counts(normalize=True) * 100
            for file_path, df in file_data.items() if len(df[df['side'] == side]) > 0
        }).round(1)
        print(side_comparison)

    return file_data

if __name__ == "__main__":
    analyze_bet_statistics_by_file()
