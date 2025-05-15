# Enhanced Wichita Public Schools Improvement Planner

This script provides an enhanced version of the Wichita Public Schools improvement planning system. It allows for multiple optimization strategies, KNN clustering analysis, and generates comprehensive reports to help decision makers.

## Features

- **7 Optimization Strategies**:
  1. **Status Quo**: The baseline approach balanced for priority
  2. **Cheapest Option**: Focuses on minimizing costs
  3. **Within Entire Scope Option**: Balanced quality improvements within budget
  4. **Speed**: Prioritizes fastest improvements
  5. **Cheapest+Fastest**: Optimizes for low cost and quick implementation
  6. **Cheapest+Quality**: Balances cost savings with quality
  7. **Fastest+Quality**: Premium option focusing on speed and quality

- **Advanced Analysis**:
  - KNN clustering to find similarities between strategies
  - Identification of common schools across strategies
  - Visualization of strategy comparisons

## Usage

### Command Line Arguments

```
python wichita_ps_enhanced.py -f <csv_file> [options]

Required:
  -f, --file FILE           Path to the CSV file with school data

Options:
  -s, --strategy STRATEGY   Strategy number (1-7)
  -b, --budget BUDGET       Total budget (default: 450,000,000)
  -c, --compare             Run all strategies and compare
  -k, --cluster             Perform KNN clustering analysis
```

### Interactive Mode

If you run the script without specifying a strategy, it enters interactive mode where you can select:
- Individual strategies (1-7)
- Comparison of all strategies (8)
- Clustering analysis (9)

## Input Data Requirements

The CSV file should contain the following columns:
- School Name
- FCI (Facility Condition Index)
- 2023/24 Official Enroll (Current enrollment)
- Full Use Capacity
- % CRs <700 (Percentage of classrooms smaller than 700 sq ft)
- % SPED CRs <700 (Percentage of special education classrooms smaller than 700 sq ft)
- 2028/29 Utilization (Projected utilization)
- Deficiencies (Cost to fix all deficiencies)
- FMP (Facility Master Plan recommendation)

## How It Works

### Optimization Process

1. **Data Processing**: Loads and cleans CSV data, calculates key metrics
2. **Strategy Selection**: Applies different optimization weights based on strategy
3. **Linear Programming**: Uses PuLP to optimize for objectives within constraints
4. **Phasing Plan**: Creates phased implementation plan based on budget
5. **Report Generation**: Outputs detailed reports and comparisons

### KNN Clustering

The script uses KMeans clustering to group similar strategies based on:
- Total cost as percentage of budget
- Percentage of schools covered
- Quality metric (average priority score)
- Speed metric (average speed score)
- Number of phases

This helps identify patterns and common threads between different optimization approaches.

## Example Output

When running the comparison mode, you'll see:
1. A table comparing all 7 strategies
2. Common schools recommended across multiple strategies
3. Cluster analysis showing which strategies are similar
4. Visualization of how strategies compare on different metrics

## Notes

- The script handles missing data and invalid values
- Budget adjustments are made for certain strategies (e.g., 120% for quality focus)
- Phase planning is tailored to each strategy's priorities

For best results, ensure your CSV data is complete and accurate.
