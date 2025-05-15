import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pulp
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

class WichitaPSModel:
    def __init__(self, total_budget=450000000):
        self.df = None
        self.total_budget = total_budget
        self.scaler = StandardScaler()
        self.optimization_results = {}
        self.phasing_plans = {}
        
    def load_and_clean_data(self, filepath: str) -> pd.DataFrame:
        """Load and clean the school data from CSV file"""
        try:
            # Load data
            self.df = pd.read_csv(filepath)
            
            # Check for missing columns
            required_columns = ['School Name', 'FCI', '2023/24 Official Enroll', 
                               'Full Use Capacity', '% CRs <700', '% SPED CRs <700',
                               '2028/29 Utilization', 'Deficiencies', 'FMP']
            
            for col in required_columns:
                if col not in self.df.columns:
                    print(f"Warning: Column '{col}' not found in dataset")
            
            # Handle missing or invalid values
            self.df['FCI'] = pd.to_numeric(self.df['FCI'], errors='coerce')
            self.df['2023/24 Official Enroll'] = pd.to_numeric(self.df['2023/24 Official Enroll'], errors='coerce')
            self.df['Full Use Capacity'] = pd.to_numeric(self.df['Full Use Capacity'], errors='coerce')
            self.df['% CRs <700'] = pd.to_numeric(self.df['% CRs <700'], errors='coerce')
            self.df['% SPED CRs <700'] = pd.to_numeric(self.df['% SPED CRs <700'], errors='coerce') 
            self.df['2028/29 Utilization'] = pd.to_numeric(self.df['2028/29 Utilization'], errors='coerce')
            self.df['Deficiencies'] = pd.to_numeric(self.df['Deficiencies'], errors='coerce')
            
            # Fill NaN values with appropriate defaults
            self.df['FCI'] = self.df['FCI'].fillna(0)
            self.df['2023/24 Official Enroll'] = self.df['2023/24 Official Enroll'].fillna(0)
            self.df['Full Use Capacity'] = self.df['Full Use Capacity'].fillna(1)  # Avoid division by zero
            self.df['% CRs <700'] = self.df['% CRs <700'].fillna(0)
            self.df['% SPED CRs <700'] = self.df['% SPED CRs <700'].fillna(0)
            self.df['2028/29 Utilization'] = self.df['2028/29 Utilization'].fillna(0.5)  # Default to 50% utilization
            self.df['Deficiencies'] = self.df['Deficiencies'].fillna(0)
            self.df['FMP'] = self.df['FMP'].fillna('No recommendation')
            
            # Calculated fields
            self.df['Utilization'] = self.df['2023/24 Official Enroll'] / self.df['Full Use Capacity']
            self.df['Utilization'] = self.df['Utilization'].clip(0, 2)  # Limit to reasonable range
            
            self.df['CR_Size_Score'] = (self.df['% CRs <700'] + self.df['% SPED CRs <700']) / 2
            self.df['CR_Size_Score'] = self.df['CR_Size_Score'].clip(0, 1)  # Ensure valid range
            
            # Calculate Priority Score - avoid NaN/inf
            self.df['Priority_Score'] = (
                self.df['FCI'] * 0.4 +  # Condition
                self.df['CR_Size_Score'] * 0.3 +  # Classroom adequacy
                (1 - self.df['2028/29 Utilization'].clip(0, 1)) * 0.3  # Future capacity needs
            )
            
            # Calculate Speed Score - schools with less deficiencies can be fixed faster
            self.df['Speed_Score'] = 1 - (self.df['Deficiencies'] / self.df['Deficiencies'].max())
            
            # Calculate Cost Efficiency Score - ratio of priority to cost
            self.df['CostEfficiency_Score'] = (
                self.df['Priority_Score'] / 
                (self.df['Deficiencies'] / self.df['Deficiencies'].max())
            ).fillna(0)
            
            # Final check for any remaining NaN/inf values
            for col in ['Priority_Score', 'Speed_Score', 'CostEfficiency_Score']:
                self.df[col] = self.df[col].fillna(0)
                self.df[col] = self.df[col].replace([np.inf, -np.inf], 0)
            
            # Remove rows with zero deficiencies (can't be improved)
            self.df = self.df[self.df['Deficiencies'] > 0].copy()
            
            print(f"Successfully processed {len(self.df)} schools with valid data")
            return self.df
            
        except Exception as e:
            print(f"Error in data loading/cleaning: {str(e)}")
            raise
    
    def optimize_improvements(self, strategy: int = 3) -> pd.DataFrame:
        """
        Optimize school improvements based on selected strategy
        
        strategy options:
        1. Status Quo (baseline - original algorithm)
        2. Cheapest Option (minimize cost)
        3. Within Entire Scope Option (balance of quality within budget)
        4. Done Quickly Option (maximize speed)
        5. Cheapest+Fastest (low quality)
        6. Cheapest+Quality (low priority)
        7. Fastest+Quality (the most expensive)
        """
        try:
            if self.df is None or len(self.df) == 0:
                raise ValueError("No valid data to optimize")
            
            # Store the strategy for later reference
            strategy_name = {
                1: "Status_Quo",
                2: "Cheapest_Option",
                3: "Within_Scope",
                4: "Done_Quickly",
                5: "Cheapest_Fastest",
                6: "Cheapest_Quality",
                7: "Fastest_Quality"
            }.get(strategy, "Unknown")
            
            prob = pulp.LpProblem(f"School_Improvements_{strategy_name}", pulp.LpMaximize)
            
            # Get list of schools with valid data
            schools = self.df['School Name'].tolist()
            
            # Decision variables - whether to include each school (0 or 1)
            x = pulp.LpVariable.dicts("include", schools, 0, 1, pulp.LpBinary)
            
            # Prepare coefficients for objective function based on strategy
            priority_coeffs = {}
            cost_coeffs = {}
            speed_coeffs = {}
            
            for s in schools:
                school_data = self.df[self.df['School Name'] == s]
                if len(school_data) > 0:
                    priority_coeffs[s] = float(school_data['Priority_Score'].iloc[0])
                    cost_coeffs[s] = float(school_data['Deficiencies'].iloc[0])
                    speed_coeffs[s] = float(school_data['Speed_Score'].iloc[0])
                else:
                    priority_coeffs[s] = 0
                    cost_coeffs[s] = 0
                    speed_coeffs[s] = 0
            
            # Define objective based on strategy
            if strategy == 1:  # Status Quo
                prob += pulp.lpSum([x[s] * priority_coeffs[s] for s in schools])
            
            elif strategy == 2:  # Cheapest Option
                # Invert costs so we maximize the negative of cost (minimize cost)
                prob += pulp.lpSum([x[s] * (1 / (cost_coeffs[s] + 0.1)) for s in schools])
            
            elif strategy == 3:  # Within Entire Scope Option (Quality within budget)
                # Balance priority and cost efficiency
                prob += pulp.lpSum([x[s] * self.df[self.df['School Name'] == s]['CostEfficiency_Score'].iloc[0] for s in schools])
            
            elif strategy == 4:  # Done Quickly Option
                # Maximize speed score
                prob += pulp.lpSum([x[s] * speed_coeffs[s] for s in schools])
                # Use 110% of budget to allow for faster implementation
                self.total_budget *= 1.1
            
            elif strategy == 5:  # Cheapest+Fastest (low quality)
                # Combine inverse of cost and speed score
                prob += pulp.lpSum([x[s] * (0.6 * (1 / (cost_coeffs[s] + 0.1)) + 0.4 * speed_coeffs[s]) for s in schools])
                # Use 80% of budget to emphasize cost savings
                self.total_budget *= 0.8
            
            elif strategy == 6:  # Cheapest+Quality (low priority)
                # Combine inverse of cost with priority
                prob += pulp.lpSum([x[s] * (0.5 * (1 / (cost_coeffs[s] + 0.1)) + 0.5 * priority_coeffs[s]) for s in schools])
                # Use 90% of budget
                self.total_budget *= 0.9
            
            elif strategy == 7:  # Fastest+Quality (most expensive)
                # Combine speed and quality
                prob += pulp.lpSum([x[s] * (0.4 * speed_coeffs[s] + 0.6 * priority_coeffs[s]) for s in schools])
                # Allow for 130% of budget for this premium option
                self.total_budget *= 1.3
            
            # Budget constraint
            prob += pulp.lpSum([x[s] * cost_coeffs[s] for s in schools]) <= self.total_budget
            
            # Add a minimum number of schools constraint for some strategies
            if strategy in [4, 7]:  # Faster strategies should ensure broader coverage
                min_schools = max(5, int(len(schools) * 0.15))  # At least 15% of schools or 5 schools
                prob += pulp.lpSum([x[s] for s in schools]) >= min_schools
            
            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            
            # Get results
            selected_schools = [s for s in schools if x[s].value() > 0.5]
            
            # Create recommendations DataFrame
            if not selected_schools:
                print(f"Warning: No schools were selected in the optimization for strategy {strategy_name}")
                recommendations = pd.DataFrame()
            else:
                recommendations = self.df[self.df['School Name'].isin(selected_schools)].copy()
                recommendations = recommendations.sort_values('Priority_Score', ascending=False)
                
                # Calculate metrics for this strategy
                total_cost = recommendations['Deficiencies'].sum()
                avg_priority = recommendations['Priority_Score'].mean()
                avg_speed = recommendations['Speed_Score'].mean()
                school_count = len(recommendations)
                
                print(f"Strategy {strategy} ({strategy_name}): Selected {school_count} schools")
                print(f"  - Total Cost: ${total_cost:,.2f}")
                print(f"  - Avg Priority: {avg_priority:.4f}")
                print(f"  - Avg Speed: {avg_speed:.4f}")
            
            # Reset budget if it was modified for this strategy
            if strategy in [4, 5, 6, 7]:
                if strategy == 4:
                    self.total_budget /= 1.1
                elif strategy == 5:
                    self.total_budget /= 0.8
                elif strategy == 6:
                    self.total_budget /= 0.9
                elif strategy == 7:
                    self.total_budget /= 1.3
            
            # Store results for comparison
            self.optimization_results[strategy_name] = {
                'recommendations': recommendations,
                'strategy': strategy,
                'strategy_name': strategy_name
            }
            
            return recommendations
            
        except Exception as e:
            print(f"Error in optimization for strategy {strategy}: {str(e)}")
            raise
    
    def generate_phasing_plan(self, recommendations: pd.DataFrame, strategy: int = 3) -> Dict:
        """Generate phasing plan for selected improvements"""
        try:
            strategy_name = {
                1: "Status_Quo",
                2: "Cheapest_Option",
                3: "Within_Scope",
                4: "Done_Quickly",
                5: "Cheapest_Fastest",
                6: "Cheapest_Quality",
                7: "Fastest_Quality"
            }.get(strategy, "Unknown")
            
            if recommendations.empty:
                empty_plan = {
                    'total_phases': 0,
                    'total_cost': 0,
                    'phases': [],
                    'remaining_total_budget': self.total_budget,
                    'strategy': strategy,
                    'strategy_name': strategy_name
                }
                self.phasing_plans[strategy_name] = empty_plan
                return empty_plan
                
            # Apply sanity checks to numerical values
            recommendations['Deficiencies'] = pd.to_numeric(recommendations['Deficiencies'], errors='coerce').fillna(0)
            recommendations['FCI'] = pd.to_numeric(recommendations['FCI'], errors='coerce').fillna(0)
            recommendations['Utilization'] = pd.to_numeric(recommendations['Utilization'], errors='coerce').fillna(0.5)
            recommendations['2028/29 Utilization'] = pd.to_numeric(recommendations['2028/29 Utilization'], errors='coerce').fillna(0.5)
            recommendations['Speed_Score'] = pd.to_numeric(recommendations['Speed_Score'], errors='coerce').fillna(0.5)
            
            total_cost = 0
            phases = []
            current_phase = []
            
            # Adjust phase budget based on strategy
            if strategy == 4:  # Done Quickly - fewer phases with larger budgets
                phase_budget = self.total_budget * 0.5  # 50% of total budget per phase
                max_phases = 2
            elif strategy == 5:  # Cheapest+Fastest - more phases with smaller budgets
                phase_budget = self.total_budget * 0.15  # 15% of total budget per phase
                max_phases = 6
            elif strategy == 7:  # Fastest+Quality - fewer phases with larger budgets
                phase_budget = self.total_budget * 0.4  # 40% of total budget per phase
                max_phases = 3
            else:
                phase_budget = self.total_budget * 0.25  # 25% of total budget per phase
                max_phases = 4
            
            # Sort schools differently based on strategy
            if strategy == 4:  # Done Quickly
                recommendations = recommendations.sort_values('Speed_Score', ascending=False)
            elif strategy == 5:  # Cheapest+Fastest
                recommendations = recommendations.sort_values(['Speed_Score', 'Deficiencies'], 
                                                           ascending=[False, True])
            elif strategy == 7:  # Fastest+Quality
                recommendations = recommendations.sort_values(['Priority_Score', 'Speed_Score'], 
                                                           ascending=[False, False])
            elif strategy == 2:  # Cheapest
                recommendations = recommendations.sort_values('Deficiencies', ascending=True)
            else:
                # Default sorting by priority score
                recommendations = recommendations.sort_values('Priority_Score', ascending=False)
            
            for _, school in recommendations.iterrows():
                cost = float(school['Deficiencies'])
                
                if cost <= 0:
                    continue  # Skip schools with zero or negative cost
                
                # Check if adding this school exceeds phase budget
                if total_cost + cost <= phase_budget:
                    current_phase.append({
                        'School': str(school['School Name']),
                        'Cost': cost,
                        'FCI': float(school['FCI']),
                        'Priority_Score': float(school['Priority_Score']),
                        'Speed_Score': float(school['Speed_Score']),
                        'Current_Utilization': float(school['Utilization']),
                        'Future_Utilization': float(school['2028/29 Utilization']),
                        'FMP_Recommendation': str(school['FMP']),
                        'CR_Size_Issue': float(school['CR_Size_Score']) > 0.5
                    })
                    total_cost += cost
                
                # Start new phase when 90% of budget is used or max schools per phase reached
                phase_threshold = 0.9
                if strategy == 4:  # Done Quickly - fill phases more completely
                    phase_threshold = 0.95
                elif strategy == 5:  # Cheapest+Fastest - more smaller phases
                    phase_threshold = 0.85
                
                if total_cost >= phase_budget * phase_threshold or len(current_phase) >= 15:
                    phases.append({
                        'schools': current_phase,
                        'phase_cost': total_cost,
                        'remaining_budget': phase_budget - total_cost,
                        'school_count': len(current_phase),
                        'avg_priority': np.mean([s['Priority_Score'] for s in current_phase]),
                        'avg_speed': np.mean([s['Speed_Score'] for s in current_phase])
                    })
                    current_phase = []
                    total_cost = 0
                    
                    # Stop if we've reached max phases
                    if len(phases) >= max_phases:
                        break
                    
            # Add final phase if there are schools left
            if current_phase:
                phases.append({
                    'schools': current_phase,
                    'phase_cost': total_cost,
                    'remaining_budget': phase_budget - total_cost,
                    'school_count': len(current_phase),
                    'avg_priority': np.mean([s['Priority_Score'] for s in current_phase]),
                    'avg_speed': np.mean([s['Speed_Score'] for s in current_phase])
                })
            
            # Calculate overall metrics
            total_schools = sum(len(phase['schools']) for phase in phases)
            overall_avg_priority = np.mean([s['Priority_Score'] for phase in phases for s in phase['schools']])
            overall_avg_speed = np.mean([s['Speed_Score'] for phase in phases for s in phase['schools']])
            
            phasing_plan = {
                'total_phases': len(phases),
                'total_cost': sum(phase['phase_cost'] for phase in phases),
                'total_schools': total_schools,
                'phases': phases,
                'remaining_total_budget': self.total_budget - sum(phase['phase_cost'] for phase in phases),
                'overall_avg_priority': overall_avg_priority,
                'overall_avg_speed': overall_avg_speed,
                'strategy': strategy,
                'strategy_name': strategy_name
            }
            
            # Store the plan
            self.phasing_plans[strategy_name] = phasing_plan
            
            return phasing_plan
            
        except Exception as e:
            print(f"Error in generating phasing plan for strategy {strategy}: {str(e)}")
            raise
    
    def run_all_strategies(self) -> Dict:
        """Run all 7 optimization strategies and generate phasing plans"""
        results = {}
        
        for strategy in range(1, 8):
            print(f"\nRunning strategy {strategy}...")
            
            # Backup original budget as some strategies modify it temporarily
            original_budget = self.total_budget
            
            # Run optimization
            recommendations = self.optimize_improvements(strategy)
            
            # Generate phasing plan if we have recommendations
            if not recommendations.empty:
                phasing_plan = self.generate_phasing_plan(recommendations, strategy)
                results[strategy] = {
                    'recommendations': recommendations,
                    'phasing_plan': phasing_plan
                }
            else:
                results[strategy] = {
                    'recommendations': pd.DataFrame(),
                    'phasing_plan': None
                }
            
            # Restore original budget
            self.total_budget = original_budget
            
        return results
    
    def cluster_strategies(self, n_clusters=3) -> Dict:
        """
        Use KNN clustering to group similar strategies and find common threads
        """
        try:
            # If we don't have results from all strategies, run them
            if len(self.phasing_plans) < 7:
                self.run_all_strategies()
            
            # Extract features for clustering
            strategy_features = []
            strategy_names = []
            
            for strategy_name, plan in self.phasing_plans.items():
                if plan['total_phases'] == 0:
                    continue  # Skip empty plans
                
                features = [
                    plan['total_cost'] / self.total_budget,  # Normalized cost
                    plan['total_schools'] / len(self.df),    # Percentage of schools covered
                    plan['overall_avg_priority'],            # Quality metric
                    plan['overall_avg_speed'],               # Speed metric
                    plan['total_phases']                     # Number of phases
                ]
                
                strategy_features.append(features)
                strategy_names.append(strategy_name)
            
            if not strategy_features:
                return {'error': 'No valid strategies to cluster'}
            
            # Normalize features
            X = self.scaler.fit_transform(strategy_features)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(X)), random_state=42)
            clusters = kmeans.fit_predict(X)
            
            # Map strategies to clusters
            strategy_clusters = {}
            for i, (strategy, cluster) in enumerate(zip(strategy_names, clusters)):
                if cluster not in strategy_clusters:
                    strategy_clusters[cluster] = []
                strategy_clusters[cluster].append({
                    'strategy': strategy,
                    'features': strategy_features[i]
                })
            
            # Identify common threads within clusters
            cluster_analysis = {}
            for cluster, strategies in strategy_clusters.items():
                # Calculate average metrics for this cluster
                avg_cost = np.mean([s['features'][0] for s in strategies]) * self.total_budget
                avg_coverage = np.mean([s['features'][1] for s in strategies]) * 100  # percentage
                avg_quality = np.mean([s['features'][2] for s in strategies])
                avg_speed = np.mean([s['features'][3] for s in strategies])
                avg_phases = np.mean([s['features'][4] for s in strategies])
                
                cluster_analysis[cluster] = {
                    'strategies': [s['strategy'] for s in strategies],
                    'avg_metrics': {
                        'cost': avg_cost,
                        'budget_percentage': avg_cost / self.total_budget * 100,
                        'school_coverage': avg_coverage,
                        'quality': avg_quality,
                        'speed': avg_speed,
                        'phases': avg_phases
                    },
                    'common_threads': self._identify_common_threads(strategies)
                }
            
            # Create PCA visualization for clusters
            pca_results = None
            if len(X) >= 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                
                pca_results = {
                    'points': X_pca.tolist(),
                    'labels': clusters.tolist(),
                    'strategies': strategy_names
                }
            
            return {
                'clusters': cluster_analysis,
                'pca': pca_results
            }
            
        except Exception as e:
            print(f"Error in clustering strategies: {str(e)}")
            return {'error': str(e)}
    
    def _identify_common_threads(self, strategies: List[Dict]) -> Dict:
        """Identify common characteristics among a group of strategies"""
        # Extract strategy names
        strategy_names = [s['strategy'] for s in strategies]
        
        # Check for common characteristics
        common_threads = {}
        
        # Check if all are high quality strategies
        if all(s in ['Highest_Quality', 'Fastest_Quality', 'Within_Scope'] for s in strategy_names):
            common_threads['focus'] = 'Quality-focused strategies'
        
        # Check if all are budget-conscious strategies  
        elif all(s in ['Cheapest_Option', 'Cheapest_Fastest', 'Cheapest_Quality'] for s in strategy_names):
            common_threads['focus'] = 'Cost-effective strategies'
        
        # Check if all emphasize speed
        elif all(s in ['Cheapest_Fastest', 'Fastest_Quality'] for s in strategy_names):
            common_threads['focus'] = 'Speed-focused strategies'
        
        # Look for common schools across these strategies
        common_schools = self._find_common_schools(strategy_names)
        if common_schools:
            common_threads['common_schools'] = common_schools
        
        # Check if plans have similar phasing
        phase_counts = [self.phasing_plans[s]['total_phases'] for s in strategy_names]
        if max(phase_counts) - min(phase_counts) <= 1:
            common_threads['similar_phasing'] = f"All have {min(phase_counts)}-{max(phase_counts)} phases"
            
        return common_threads
    
    def _find_common_schools(self, strategy_names: List[str], threshold=0.7) -> List[str]:
        """Find schools that appear in multiple strategies"""
        # Get all schools selected in each strategy
        strategy_schools = {}
        for strategy in strategy_names:
            if strategy not in self.optimization_results:
                continue
                
            df = self.optimization_results[strategy]['recommendations']
            if df.empty:
                strategy_schools[strategy] = set()
            else:
                strategy_schools[strategy] = set(df['School Name'].tolist())
        
        if not strategy_schools:
            return []
            
        # Find schools that appear in at least threshold % of strategies
        all_schools = set().union(*strategy_schools.values())
        common_schools = []
        
        for school in all_schools:
            count = sum(1 for schools in strategy_schools.values() if school in schools)
            if count / len(strategy_schools) >= threshold:
                common_schools.append(school)
                
        return common_schools
                
    def print_analysis_report(self, phasing_plan: Dict):
        """Print detailed analysis report for a single strategy"""
        try:
            print(f"\nWichita Public Schools Improvement Plan - Strategy: {phasing_plan['strategy_name']}")
            print("="*80)
            print(f"Total Budget: ${self.total_budget:,.2f}")
            print(f"Total Cost: ${phasing_plan['total_cost']:,.2f}")
            print(f"Remaining Budget: ${phasing_plan['remaining_total_budget']:,.2f}")
            print(f"Number of Phases: {phasing_plan['total_phases']}")
            print(f"Number of Schools: {phasing_plan.get('total_schools', sum(len(p['schools']) for p in phasing_plan['phases']))}")
            print(f"Average Priority Score: {phasing_plan.get('overall_avg_priority', 0):.4f}")
            print(f"Average Speed Score: {phasing_plan.get('overall_avg_speed', 0):.4f}")
            
            for i, phase in enumerate(phasing_plan['phases'], 1):
                print(f"\nPhase {i}")
                print("-"*40)
                print(f"Phase Budget: ${self.total_budget * 0.25:,.2f}")
                print(f"Phase Cost: ${phase['phase_cost']:,.2f}")
                print(f"Remaining Phase Budget: ${phase['remaining_budget']:,.2f}")
                print(f"Schools in this phase: {len(phase['schools'])}")
                
                for school in phase['schools']:
                    print(f"\n{school['School']}")
                    print(f"  Cost: ${school['Cost']:,.2f}")
                    print(f"  FCI: {school['FCI']:.2%}")
                    print(f"  Priority Score: {school['Priority_Score']:.4f}")
                    print(f"  Speed Score: {school['Speed_Score']:.4f}")
                    print(f"  Current Utilization: {school['Current_Utilization']:.1%}")
                    print(f"  2028/29 Utilization: {school['Future_Utilization']:.1%}")
                    print(f"  Classroom Size Issues: {'Yes' if school['CR_Size_Issue'] else 'No'}")
                    print(f"  FMP Recommendation: {school['FMP_Recommendation']}")
                    
        except Exception as e:
            print(f"Error in generating report: {str(e)}")
            raise
    
    def print_comparison_report(self):
        """Print comparative report of all strategies"""
        try:
            # If we don't have results from all strategies, run them
            if len(self.phasing_plans) < 7:
                self.run_all_strategies()
            
            print("\nWichita Public Schools Strategy Comparison")
            print("="*100)
            
            # Prepare data for tabular comparison
            comparison_data = []
            headers = ["Strategy", "Total Cost", "Budget %", "Schools", "Phases", "Avg Priority", "Avg Speed"]
            
            for strategy in range(1, 8):
                strategy_name = {
                    1: "1-Status Quo",
                    2: "2-Cheapest Option",
                    3: "3-Within Scope",
                    4: "4-Highest Quality",
                    5: "5-Cheapest+Fastest",
                    6: "6-Cheapest+Quality",
                    7: "7-Fastest+Quality"
                }.get(strategy, f"Strategy {strategy}")
                
                plan_key = strategy_name.split("-")[1]
                if plan_key in self.phasing_plans:
                    plan = self.phasing_plans[plan_key]
                    
                    # Skip empty plans
                    if plan['total_phases'] == 0:
                        comparison_data.append([
                            strategy_name, "$0", "0%", "0", "0", "0.00", "0.00"
                        ])
                        continue
                    
                    comparison_data.append([
                        strategy_name,
                        f"${plan['total_cost']:,.2f}",
                        f"{plan['total_cost']/self.total_budget*100:.1f}%",
                        str(plan.get('total_schools', sum(len(p['schools']) for p in plan['phases']))),
                        str(plan['total_phases']),
                        f"{plan.get('overall_avg_priority', 0):.4f}",
                        f"{plan.get('overall_avg_speed', 0):.4f}"
                    ])
                else:
                    comparison_data.append([
                        strategy_name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
                    ])
            
            # Print table
            print(tabulate(comparison_data, headers=headers, tablefmt="grid"))
            
            # Print common schools across all strategies
            common_schools = self._find_common_schools([
                "Status_Quo", "Cheapest_Option", "Within_Scope", 
                "Highest_Quality", "Cheapest_Fastest", "Cheapest_Quality",
                "Fastest_Quality"
            ], threshold=0.5)
            
            if common_schools:
                print("\nSchools recommended in multiple strategies:")
                for school in common_schools:
                    strategies_with_school = []
                    for strategy, results in self.optimization_results.items():
                        if not results['recommendations'].empty and school in results['recommendations']['School Name'].values:
                            strategies_with_school.append(strategy)
                    
                    print(f"- {school}: Found in {len(strategies_with_school)}/7 strategies")
                    school_data = self.df[self.df['School Name'] == school].iloc[0]
                    print(f"  Priority: {school_data['Priority_Score']:.4f}, Cost: ${school_data['Deficiencies']:,.2f}, FCI: {school_data['FCI']:.2%}")
            
        except Exception as e:
            print(f"Error in generating comparison report: {str(e)}")
            raise
    
    def visualize_clusters(self, cluster_results: Dict):
        """Create visualizations for the clustering results"""
        try:
            plt.figure(figsize=(12, 10))
            
            # 2D PCA plot of strategies
            if 'pca' in cluster_results and cluster_results['pca']:
                plt.subplot(2, 2, 1)
                
                pca_data = cluster_results['pca']
                points = np.array(pca_data['points'])
                labels = np.array(pca_data['labels'])
                strategy_names = pca_data['strategies']
                
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                for i, label in enumerate(np.unique(labels)):
                    mask = labels == label
                    plt.scatter(points[mask, 0], points[mask, 1], c=colors[i % len(colors)], label=f'Cluster {label}')
                    
                    # Annotate points with strategy names
                    for j, strategy in enumerate(np.array(strategy_names)[mask]):
                        plt.annotate(strategy, (points[mask, 0][j], points[mask, 1][j]))
                
                plt.title('Strategy Clusters')
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')
                plt.legend()
            
            # Cost comparison
            plt.subplot(2, 2, 2)
            strategy_names = []
            costs = []
            
            for strategy in range(1, 8):
                strategy_name = {
                    1: "Status Quo",
                    2: "Cheapest",
                    3: "Within Scope",
                    4: "Highest Quality",
                    5: "Cheap+Fast",
                    6: "Cheap+Quality",
                    7: "Fast+Quality"
                }.get(strategy, f"S{strategy}")
                
                plan_key = strategy_name.replace(" ", "_")
                if plan_key in self.phasing_plans:
                    plan = self.phasing_plans[plan_key]
                    if plan['total_phases'] > 0:
                        strategy_names.append(strategy_name)
                        costs.append(plan['total_cost'])
            
            plt.bar(strategy_names, costs)
            plt.title('Total Cost by Strategy')
            plt.ylabel('Cost ($)')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Schools covered
            plt.subplot(2, 2, 3)
            strategy_names = []
            schools = []
            
            for strategy in range(1, 8):
                strategy_name = {
                    1: "Status Quo",
                    2: "Cheapest",
                    3: "Within Scope",
                    4: "Highest Quality",
                    5: "Cheap+Fast",
                    6: "Cheap+Quality",
                    7: "Fast+Quality"
                }.get(strategy, f"S{strategy}")
                
                plan_key = strategy_name.replace(" ", "_")
                if plan_key in self.phasing_plans:
                    plan = self.phasing_plans[plan_key]
                    if plan['total_phases'] > 0:
                        strategy_names.append(strategy_name)
                        schools.append(plan.get('total_schools', sum(len(p['schools']) for p in plan['phases'])))
            
            plt.bar(strategy_names, schools)
            plt.title('Schools Covered by Strategy')
            plt.ylabel('Number of Schools')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Quality vs. Speed
            plt.subplot(2, 2, 4)
            quality = []
            speed = []
            names = []
            
            for strategy in range(1, 8):
                strategy_name = {
                    1: "Status Quo",
                    2: "Cheapest",
                    3: "Within Scope",
                    4: "Highest Quality",
                    5: "Cheap+Fast",
                    6: "Cheap+Quality",
                    7: "Fast+Quality"
                }.get(strategy, f"S{strategy}")
                
                plan_key = strategy_name.replace(" ", "_")
                if plan_key in self.phasing_plans:
                    plan = self.phasing_plans[plan_key]
                    if plan['total_phases'] > 0:
                        quality.append(plan.get('overall_avg_priority', 0))
                        speed.append(plan.get('overall_avg_speed', 0))
                        names.append(strategy_name)
            
            plt.scatter(quality, speed)
            for i, name in enumerate(names):
                plt.annotate(name, (quality[i], speed[i]))
            plt.title('Quality vs. Speed by Strategy')
            plt.xlabel('Average Priority Score')
            plt.ylabel('Average Speed Score')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Return as base64 string - would need to be implemented
            # return convert_plot_to_base64(plt)
            
        except Exception as e:
            print(f"Error in visualizing clusters: {str(e)}")
    
    def get_strategy_recommendation(self, user_preferences: Optional[Dict] = None) -> Dict:
        """Provide a recommendation based on user preferences"""
        try:
            # Default preferences
            preferences = {
                'cost_importance': 0.33,
                'quality_importance': 0.33,
                'speed_importance': 0.33
            }
            
            # Update with user preferences if provided
            if user_preferences:
                preferences.update(user_preferences)
            
            # Normalize preference weights
            total_weight = sum(preferences.values())
            for key in preferences:
                preferences[key] /= total_weight
            
            # Score each strategy
            strategy_scores = {}
            
            for strategy_name, plan in self.phasing_plans.items():
                if plan['total_phases'] == 0:
                    continue
                
                # Calculate normalized metrics
                cost_score = 1 - (plan['total_cost'] / self.total_budget)  # Lower cost is better
                quality_score = plan.get('overall_avg_priority', 0)
                speed_score = plan.get('overall_avg_speed', 0)
                
                # Calculate weighted score
                total_score = (
                    preferences['cost_importance'] * cost_score +
                    preferences['quality_importance'] * quality_score +
                    preferences['speed_importance'] * speed_score
                )
                
                strategy_scores[strategy_name] = total_score
            
            if not strategy_scores:
                return {'recommendation': None, 'message': 'No valid strategies found'}
            
            # Get the best strategy
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
            
            # Map back to numeric strategy
            strategy_to_num = {
                'Status_Quo': 1,
                'Cheapest_Option': 2,
                'Within_Scope': 3,
                'Highest_Quality': 4,
                'Cheapest_Fastest': 5,
                'Cheapest_Quality': 6,
                'Fastest_Quality': 7
            }
            
            return {
                'recommendation': strategy_to_num.get(best_strategy[0], 0),
                'strategy_name': best_strategy[0],
                'score': best_strategy[1],
                'all_scores': strategy_scores
            }
            
        except Exception as e:
            print(f"Error in getting strategy recommendation: {str(e)}")
            return {'recommendation': None, 'message': f'Error: {str(e)}'}


def main():
    try:
        # Get user input for filepath and strategy
        import argparse
        import os
        import sys
        
        parser = argparse.ArgumentParser(description='Wichita Public Schools Improvement Planner')
        parser.add_argument('-f', '--file', type=str, help='Path to CSV file')
        parser.add_argument('-s', '--strategy', type=int, choices=range(1, 8), help='Strategy (1-7)', default=0)
        parser.add_argument('-b', '--budget', type=float, help='Total budget', default=450000000)
        parser.add_argument('-c', '--compare', action='store_true', help='Run all strategies and compare')
        parser.add_argument('-k', '--cluster', action='store_true', help='Perform KNN clustering')
        
        # Check if script was called with no arguments - if so, enter interactive mode
        if len(sys.argv) == 1:
            # Go straight to interactive mode
            args = parser.parse_args([])
            args.file = None
        else:
            args = parser.parse_args()
        
        # Initialize model with budget
        model = WichitaPSModel(total_budget=args.budget)
        
        # Interactive file selection if not provided
        if args.file is None:
            print("\nWichita Public Schools Improvement Planner")
            print("="*50)
            
            # Ask for file path
            while True:
                filepath = input("\nEnter the path to your CSV file (or type 'exit' to quit): ")
                
                if filepath.lower() == 'exit':
                    print("Exiting program.")
                    return
                
                # Check if file exists
                if os.path.exists(filepath) and filepath.lower().endswith('.csv'):
                    args.file = filepath
                    break
                else:
                    print(f"Error: File not found or not a CSV file: {filepath}")
                    print("Please enter a valid path to a CSV file.")
                    continue
        
        # Load and process data
        print("\nLoading and processing data...")
        try:
            df = model.load_and_clean_data(args.file)
        except Exception as e:
            print(f"Error loading data from {args.file}: {e}")
            print("Please check that the file exists and contains the required columns.")
            return
        
        # Handle user choice
        if args.compare or args.cluster:
            # Run all strategies
            print("\nRunning all strategies for comparison...")
            model.run_all_strategies()
            
            # Print comparison report
            model.print_comparison_report()
            
            if args.cluster:
                # Perform clustering analysis
                print("\nPerforming clustering analysis...")
                cluster_results = model.cluster_strategies()
                
                # Print cluster analysis results
                print("\nCluster Analysis Results:")
                for cluster_id, cluster_info in cluster_results['clusters'].items():
                    print(f"\nCluster {cluster_id}")
                    print("-" * 40)
                    print(f"Strategies in this cluster: {', '.join(cluster_info['strategies'])}")
                    print("\nAverage Metrics:")
                    metrics = cluster_info['avg_metrics']
                    print(f"  Cost: ${metrics['cost']:,.2f} ({metrics['budget_percentage']:.1f}% of budget)")
                    print(f"  School Coverage: {metrics['school_coverage']:.1f}%")
                    print(f"  Quality Score: {metrics['quality']:.4f}")
                    print(f"  Speed Score: {metrics['speed']:.4f}")
                    print(f"  Average Phases: {metrics['phases']:.1f}")
                    
                    if cluster_info['common_threads']:
                        print("\nCommon Characteristics:")
                        for key, value in cluster_info['common_threads'].items():
                            if key == 'common_schools' and isinstance(value, list):
                                print(f"  Common Schools: {', '.join(value[:5])}" + 
                                      (f" and {len(value)-5} more..." if len(value) > 5 else ""))
                            else:
                                print(f"  {key.replace('_', ' ').title()}: {value}")
                
                # Create visualization
                print("\nGenerating cluster visualizations...")
                model.visualize_clusters(cluster_results)
                print("Visualizations generated. (Note: In a GUI environment, these would be displayed)")
                
                # Get recommendation based on balanced preferences
                print("\nGenerating balanced recommendation...")
                recommendation = model.get_strategy_recommendation()
                if recommendation['recommendation']:
                    print(f"Recommended strategy: {recommendation['recommendation']} - {recommendation['strategy_name']}")
                    print(f"Score: {recommendation['score']:.4f}")
                    print("\nAll strategy scores:")
                    for strategy, score in sorted(recommendation['all_scores'].items(), key=lambda x: x[1], reverse=True):
                        print(f"  {strategy}: {score:.4f}")
                else:
                    print(f"No recommendation available: {recommendation['message']}")
        
        elif args.strategy > 0:
            # Run specific strategy
            print(f"\nRunning strategy {args.strategy}...")
            recommendations = model.optimize_improvements(args.strategy)
            
            # Generate phasing plan
            print("\nGenerating phasing plan...")
            phasing_plan = model.generate_phasing_plan(recommendations, args.strategy)
            
            # Print results
            model.print_analysis_report(phasing_plan)
        
        else:
            # Interactive mode
            print("\nWichita Public Schools Improvement Planner")
            print("="*50)
            print("Please select a strategy:")
            print("1. Status Quo")
            print("2. Cheapest Option")
            print("3. Within Entire Scope Option (Quality)")
            print("4. Done Quickly Option")
            print("5. Cheapest+Fastest (low quality)")
            print("6. Cheapest+Quality (low priority)")
            print("7. Fastest+Quality (the most expensive)")
            print("8. Run all strategies and compare")
            print("9. Run clustering analysis")
            
            while True:
                choice = input("\nEnter your choice (1-9) or 'exit' to quit: ")
                
                if choice.lower() == 'exit':
                    print("Exiting program.")
                    break
                
                try:
                    choice = int(choice)
                    if choice == 8:
                        # Run comparison
                        model.run_all_strategies()
                        model.print_comparison_report()
                    elif choice == 9:
                        # Run clustering
                        model.run_all_strategies()
                        cluster_results = model.cluster_strategies()
                        model.visualize_clusters(cluster_results)
                        model.print_comparison_report()
                    elif 1 <= choice <= 7:
                        # Run specific strategy
                        recommendations = model.optimize_improvements(choice)
                        phasing_plan = model.generate_phasing_plan(recommendations, choice)
                        model.print_analysis_report(phasing_plan)
                    else:
                        print("Invalid choice. Please select 1-9.")
                        continue
                    
                    # Ask if user wants to continue
                    another = input("\nWould you like to run another analysis? (y/n): ")
                    if another.lower() != 'y':
                        print("Exiting program.")
                        break
                        
                except ValueError:
                    print("Invalid input. Please enter a number from 1 to 9 or 'exit' to quit.")
        
    except FileNotFoundError:
        print(f"Error: Could not find file at the specified path")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
