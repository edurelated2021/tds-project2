# /// script
# requires-python = "==3.12.8"
# dependencies = [
#   "httpx",
#   "requests",
#   "networkx",
#   "folium",
#   "pandas",
#   "fsspec",
#   "chardet",
#   "seaborn",
#   "scipy",
#   "scikit-learn",
#   "statsmodels"
# ]
# ///
import numpy as np
import pandas as pd
import sys
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import requests
import networkx as nx
from datetime import datetime
import folium
from io import BytesIO
import base64
import os
import chardet
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import json

class DataAnalyzer:
    def __init__(self):
        self.analysis_results = {
            'outliers': {},
            'correlations': {},
            'regression': {},
            'time_series': {},
            'clustering': {},
            'geographic': {},
            'network': {},
            'summary_stats': {},
            'quality_metrics': {}
        }
    
    def detect_outliers_and_anomalies(self, df):
        """
        Comprehensive outlier and anomaly detection
        """
        results = {}
        
        # Only analyze numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Z-score method
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            z_score_outliers = len(z_scores[z_scores > 3])
            
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = len(df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))])
            
            results[col] = {
                'z_score_outliers': int(z_score_outliers),
                'iqr_outliers': int(iqr_outliers),
                'z_score_percentage': round((z_score_outliers / len(df)) * 100, 2),
                'iqr_percentage': round((iqr_outliers / len(df)) * 100, 2)
            }
        
        # Isolation Forest for multivariate anomaly detection
        if len(numeric_cols) >= 2:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(df[numeric_cols].fillna(df[numeric_cols].mean()))
            results['multivariate_anomalies'] = {
                'count': int(sum(anomalies == -1)),
                'percentage': round((sum(anomalies == -1) / len(df)) * 100, 2)
            }
        
        self.analysis_results['outliers'] = results
        return results

    def analyze_correlations(self, df):
        """
        Analyze correlations between numeric variables
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results = {}
        
        if len(numeric_cols) >= 2:
            # Correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Find significant correlations
            significant_corr = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.5:  # Threshold for significant correlation
                        significant_corr.append({
                            'variables': (numeric_cols[i], numeric_cols[j]),
                            'correlation': round(corr, 3)
                        })
            
            results['significant_correlations'] = significant_corr
            results['correlation_matrix'] = corr_matrix.to_dict()
            
            # PCA to understand overall data structure
            if len(numeric_cols) > 2:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[numeric_cols].fillna(df[numeric_cols].mean()))
                pca = PCA()
                pca.fit(scaled_data)
                
                results['pca_explained_variance'] = {
                    'ratios': pca.explained_variance_ratio_.tolist(),
                    'cumulative': np.cumsum(pca.explained_variance_ratio_).tolist()
                }
        
        self.analysis_results['correlations'] = results
        return results

    # Continue with additional analysis methods
    def analyze_time_series(self, df):
        """
        Analyze time series patterns in the data
        """
        results = {}
        
        # Identify potential date columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) == 0:
            # Try to convert columns that might contain dates
            for col in df.columns:
                try:
                    pd.to_datetime(df[col])
                    date_cols = [col]
                    break
                except:
                    continue
        
        if len(date_cols) > 0:
            date_col = date_cols[0]
            df = df.sort_values(date_col)
            
            # Analyze numeric columns over time
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col == date_col:
                    continue
                    
                # Basic time series metrics
                try:
                    # Decomposition
                    series = df[col].fillna(method='ffill')
                    if len(series) > 2:  # Need at least 2 periods for decomposition
                        decomposition = seasonal_decompose(series, period=min(len(series)//2, 12))
                        results[col] = {
                            'trend': decomposition.trend.dropna().tolist(),
                            'seasonal': decomposition.seasonal.dropna().tolist(),
                            'residual': decomposition.resid.dropna().tolist()
                        }
                        
                        # Stationarity test
                        adf_test = adfuller(series.dropna())
                        results[col]['stationarity'] = {
                            'adf_statistic': adf_test[0],
                            'p_value': adf_test[1],
                            'is_stationary': adf_test[1] < 0.05
                        }
                except:
                    continue
        
        self.analysis_results['time_series'] = results
        return results

    def analyze_clusters(self, df):
        """
        Perform cluster analysis on the data
        """
        results = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Prepare data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_cols].fillna(df[numeric_cols].mean()))
            
            # K-means clustering
            max_clusters = min(5, len(df)//2)
            inertias = []
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)
            
            # Find optimal number of clusters using elbow method
            optimal_clusters = 2  # default
            for i in range(1, len(inertias)):
                if (inertias[i-1] - inertias[i]) / inertias[i-1] < 0.2:  # 20% improvement threshold
                    optimal_clusters = i + 1
                    break
            
            # Final clustering with optimal number
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Analyze cluster characteristics
            cluster_stats = {}
            df_with_clusters = df.copy()
            df_with_clusters['Cluster'] = clusters
            
            for cluster in range(optimal_clusters):
                cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
                cluster_stats[f'cluster_{cluster}'] = {
                    'size': len(cluster_data),
                    'percentage': round(len(cluster_data) / len(df) * 100, 2),
                    'characteristics': {}
                }
                
                # Calculate mean values for each numeric column in the cluster
                for col in numeric_cols:
                    cluster_mean = cluster_data[col].mean()
                    overall_mean = df[col].mean()
                    difference = round(((cluster_mean - overall_mean) / overall_mean) * 100, 2)
                    
                    cluster_stats[f'cluster_{cluster}']['characteristics'][col] = {
                        'mean': round(cluster_mean, 2),
                        'difference_from_overall': difference
                    }
            
            results['kmeans'] = {
                'optimal_clusters': optimal_clusters,
                'inertias': inertias,
                'cluster_stats': cluster_stats
            }
            
            # DBSCAN for detecting natural clusters
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_clusters = dbscan.fit_predict(scaled_data)
            
            results['dbscan'] = {
                'n_clusters': len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0),
                'noise_points': sum(dbscan_clusters == -1)
            }
        
        self.analysis_results['clustering'] = results
        return results

    # Continue with geographic and network analysis methods
    def analyze_geographic(self, df):
        """
        Analyze geographic patterns in the data
        """
        results = {}
        
        # Look for common geographic column names
        geo_columns = [col for col in df.columns if any(term in col.lower() 
                    for term in ['country', 'state', 'city', 'region', 'latitude', 'longitude', 'lat', 'long', 'zip'])]
        
        if geo_columns:
            results['geo_columns'] = geo_columns
            
            # Analyze distribution by geographic regions
            for col in geo_columns:
                value_counts = df[col].value_counts()
                results[f'{col}_distribution'] = {
                    'top_locations': value_counts.head(10).to_dict(),
                    'unique_locations': len(value_counts)
                }
                
                # If we have lat/long pairs, calculate geographic clusters
                if ('lat' in col.lower() or 'latitude' in col.lower()) and any('long' in c.lower() for c in geo_columns):
                    lat_col = col
                    long_col = [c for c in geo_columns if 'long' in c.lower()][0]
                    
                    # Calculate geographic centroids and dispersion
                    valid_coords = df[[lat_col, long_col]].dropna()
                    if len(valid_coords) > 0:
                        results['geographic_metrics'] = {
                            'centroid': {
                                'latitude': float(valid_coords[lat_col].mean()),
                                'longitude': float(valid_coords[long_col].mean())
                            },
                            'dispersion': {
                                'lat_std': float(valid_coords[lat_col].std()),
                                'long_std': float(valid_coords[long_col].std())
                            }
                        }
        
        self.analysis_results['geographic'] = results
        return results

    def analyze_network(self, df):
        """
        Analyze network relationships in the data
        """
        results = {}
        
        # Look for potential relationship columns (e.g., ID pairs, categories that could be connected)
        potential_id_cols = [col for col in df.columns if 'id' in col.lower() or 'category' in col.lower()]
        
        if len(potential_id_cols) >= 2:
            # Create a network graph
            G = nx.Graph()
            
            # Add edges between related items
            for i in range(len(potential_id_cols)):
                for j in range(i+1, len(potential_id_cols)):
                    col1, col2 = potential_id_cols[i], potential_id_cols[j]
                    
                    # Create edges between unique pairs
                    unique_pairs = df[[col1, col2]].drop_duplicates()
                    edges = list(map(tuple, unique_pairs.values))
                    G.add_edges_from(edges)
            
            # Calculate network metrics
            results['network_metrics'] = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'connected_components': nx.number_connected_components(G)
            }
            
            # Identify important nodes
            if G.number_of_nodes() > 0:
                degree_centrality = nx.degree_centrality(G)
                results['central_nodes'] = {
                    str(k): round(v, 3) 
                    for k, v in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                }
        
        self.analysis_results['network'] = results
        return results

    def generate_summary_report(self):
        """
        Generate a comprehensive summary of all analyses
        """
        summary = {
            'dataset_overview': {
                'timestamp': datetime.now().isoformat(),
                'analyses_performed': list(self.analysis_results.keys())
            }
        }
        
        # Add summaries from each analysis type
        for analysis_type, results in self.analysis_results.items():
            if results:  # Only include non-empty results
                summary[analysis_type] = {
                    'key_findings': self._extract_key_findings(analysis_type, results)
                }
        
        return summary

    def _extract_key_findings(self, analysis_type, results):
        """
        Extract key findings from each analysis type
        """
        findings = []
        
        if analysis_type == 'outliers':
            for col, stats in results.items():
                if isinstance(stats, dict) and 'z_score_outliers' in stats:
                    if stats['z_score_percentage'] > 5:  # Significant outliers
                        findings.append(f"Column {col} has {stats['z_score_percentage']}% outliers")
        
        elif analysis_type == 'correlations':
            if 'significant_correlations' in results:
                for corr in results['significant_correlations'][:3]:  # Top 3 correlations
                    findings.append(f"Strong correlation ({corr['correlation']}) between {corr['variables'][0]} and {corr['variables'][1]}")
        
        elif analysis_type == 'clustering':
            if 'kmeans' in results:
                findings.append(f"Identified {results['kmeans']['optimal_clusters']} distinct clusters")
                
        elif analysis_type == 'geographic':
            if 'geographic_metrics' in results:
                findings.append(f"Data centered around lat: {results['geographic_metrics']['centroid']['latitude']}, long: {results['geographic_metrics']['centroid']['longitude']}")
        
        elif analysis_type == 'network':
            if 'network_metrics' in results:
                findings.append(f"Network contains {results['network_metrics']['nodes']} nodes and {results['network_metrics']['edges']} edges")
        
        return findings


    def read_csv_file(self, file_path):
        """
        Reads a CSV file with automatic encoding detection and error handling.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame or None: DataFrame if successful, None if failed
        """
        try:
            # First check if file exists
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} does not exist")
                return None
            
            # Read a sample of the file to detect encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read(10000)  # Read first 10000 bytes
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                print("encoging", encoding)
            
            # Try reading with detected encoding
            try:
                df = pd.read_csv(file_path, encoding=encoding)
            except:
                # If detected encoding fails, try common encodings
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for enc in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=enc)
                        print(f"Successfully read file using {enc} encoding")
                        break
                    except:
                        continue
                else:
                    raise Exception("Failed to read with all attempted encodings")
            
            print(f"File successfully read with {len(df)} rows and {len(df.columns)} columns")
            return df
        
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return None
        
    # Add summary statistics calculation
    def generate_summary_statistics1(self, df):
        """
        Generate comprehensive summary statistics for a DataFrame
        """
        summary_stats = {
            'numeric_summary': {},
            'categorical_summary': {},
            'overall': {
                'total_rows': len(df),
                'total_columns': len(df.columns)
            }
        }
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary_stats['numeric_summary'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'missing': int(df[col].isna().sum()),
                'missing_pct': float(df[col].isna().mean() * 100)
            }
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            summary_stats['categorical_summary'][col] = {
                'unique_values': int(df[col].nunique()),
                'most_common': str(value_counts.index[0]) if not value_counts.empty else None,
                'most_common_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
                'missing': int(df[col].isna().sum()),
                'missing_pct': float(df[col].isna().mean() * 100)
            }
        
        return summary_stats

   

    def generate_summary_statistics(self, df):
        """
        Generate comprehensive summary statistics for a DataFrame
        """
        summary_stats = {}
        
        # Basic dataset info
        summary_stats['basic_info'] = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # in MB
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Numeric statistics
        if len(numeric_cols) > 0:
            numeric_stats = {}
            for col in numeric_cols:
                stats = df[col].describe()
                numeric_stats[col] = {
                    'count': stats['count'],
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'median': stats['50%'],
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis()
                }
            summary_stats['numeric_stats'] = numeric_stats
        
        # Categorical statistics
        if len(categorical_cols) > 0:
            cat_stats = {}
            for col in categorical_cols:
                cat_stats[col] = {
                    'unique_values': df[col].nunique(),
                    'top_5_values': df[col].value_counts().nlargest(5).to_dict(),
                    'missing_values': df[col].isnull().sum()
                }
            summary_stats['categorical_stats'] = cat_stats
        
        # Additional statistics
        summary_stats['correlation'] = {}
        if len(numeric_cols) > 1:
            summary_stats['correlation'] = df[numeric_cols].corr().to_dict()
        self.analysis_results['summary_stats'] = summary_stats
        return summary_stats

    def analyze_csv(self, file_path):
        """
        Main function to analyze a single CSV file
        """
        analyzer = DataAnalyzer()
        print(f"\
    Analyzing {file_path}:")
        print("=" * 50)
        
        # Read the CSV file
        df = analyzer.read_csv_file(file_path)
        
        if df is not None:
            # Generate and print summary statistics
            stats = analyzer.generate_summary_statistics(df)
            
            # Print key insights
            print("\
    Key Insights:")
            print(f"Number of rows: {stats['basic_info']['rows']}")
            print(f"Number of columns: {stats['basic_info']['columns']}")
            print(f"Memory usage: {stats['basic_info']['memory_usage']:.2f} MB")
            
            if 'numeric_stats' in stats:
                print("\
    Numeric Columns Summary:")
                for col, metrics in stats['numeric_stats'].items():
                    print(f"\
    {col}:")
                    print(f"  Mean: {metrics['mean']:.2f}")
                    print(f"  Std: {metrics['std']:.2f}")
                    print(f"  Min: {metrics['min']}")
                    print(f"  Max: {metrics['max']}")
            
            if 'categorical_stats' in stats:
                print("\
    Categorical Columns Summary:")
                for col, metrics in stats['categorical_stats'].items():
                    print(f"\
    {col}:")
                    print(f"  Unique values: {metrics['unique_values']}")
                    print("  Top 5 values:")
                    for val, count in metrics['top_5_values'].items():
                        print(f"    {val}: {count}")
            
            return df, stats
        return None, None

    # Adding data quality analysis function
    def analyze_data_quality(self, df):
        """
        Analyze the quality of data in a DataFrame
        Returns a dictionary with quality metrics
        """
        quality_report = {}
        
        # Missing values analysis
        missing_vals = df.isnull().sum()
        missing_percentages = (missing_vals / len(df)) * 100
        quality_report['missing_values'] = {
            'total_missing': missing_vals.sum(),
            'missing_by_column': {col: {'count': int(count), 
                                    'percentage': round(pct, 2)} 
                                for col, count, pct in zip(df.columns, 
                                                        missing_vals, 
                                                        missing_percentages) 
                                if count > 0}
        }
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        quality_report['duplicates'] = {
            'total_duplicates': int(duplicates),
            'percentage_duplicates': round((duplicates / len(df)) * 100, 2)
        }
        
        # Data type analysis
        quality_report['data_types'] = df.dtypes.astype(str).to_dict()
        
        # Consistency checks
        quality_report['consistency'] = {}
        
        # Check numeric columns for outliers using IQR method
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        outliers_report = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            
            if len(outliers) > 0:
                outliers_report[col] = {
                    'count': len(outliers),
                    'percentage': round((len(outliers) / len(df)) * 100, 2),
                    'min_outlier': float(outliers.min()),
                    'max_outlier': float(outliers.max())
                }
        
        quality_report['outliers'] = outliers_report
        
        # Zero and negative values in numeric columns
        zero_neg_report = {}
        for col in numeric_cols:
            zeros = (df[col] == 0).sum()
            negatives = (df[col] < 0).sum()
            if zeros > 0 or negatives > 0:
                zero_neg_report[col] = {
                    'zeros': int(zeros),
                    'negatives': int(negatives)
                }
        
        quality_report['zero_negative_values'] = zero_neg_report
        
        return quality_report
    
    # Add the generate_report function to the DataAnalyzer class
    def generate_llm_report(self, img_base64):
        """
        Generate a comprehensive markdown report using OpenAI's API based on analysis results
        """
        # Prepare the system message with the report format
        system_message = """You are a data analysis expert. Generate a detailed report in markdown format following this structure:
        1. Brief description of the data (structure and types)
        2. Summary of analyses performed
        3. Key insights discovered
        4. Implications and recommended next steps
        
        Be specific, quantitative where possible, and focus on actionable insights."""
        
        # Prepare the analysis results as a structured message
        # analysis_content = json.dumps(self.analysis_results, indent=2)
        print(self.analysis_results['summary_stats'])
        
        user_message = f"Please analyze the following data analysis results and image (correlation matrix) and generate a report:\
    {self.analysis_results['summary_stats']}"
    
       
        messages = [
            {   "role": "system", 
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            },
        ]
        correlation_matrix_file = "correlation.png"
       
        
        images=[
            {"name": correlation_matrix_file, "data": img_base64}
        ]
        
        api_token = os.getenv("AIPROXY_TOKEN")
        
        try:
             # Query the Chat Completion API
            print("Invoking chat completion api")
            response = requests.post(
                "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_token}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": messages,
                        # "files": images,
                        "temperature": 0.8,
                        "max_tokens": 5000
                    },                    
                    
            )
            result = response.json()
            print(result)
            
            # Extract the generated report from the response
            report = result["choices"][0]["message"]["content"]
            
            # Save the report to a markdown file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = "README.md"
            with open(filename, 'w') as f:
                f.write(report)
                
            return report, filename
            
        except Exception as e:
            print("Error generating report:", str(e))
            return None, None

    def generate_correlation_heatmap(self, df):
        # Convert all columns to numeric values where possible, non-convertible values become NaN
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        
        # Compute the correlation matrix for numeric columns only
        correlation_matrix = df_numeric.corr()
        
        # Create and save the heatmap to an image file
        plt.figure(figsize=(12, 10))  # Increase the size to accommodate larger labels
        ax = sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
                        cbar_kws={"shrink": 0.8})  # Adjust the color bar size for better fitting
        
        # Increase the font size of annotations for better readability
        plt.title("Correlation Matrix", fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate column names and adjust their size
        plt.yticks(rotation=0, ha='right', fontsize=12)  # Adjust row label orientation and size
        
        # Save the image as 'correlation.png' in the current folder
        plt.tight_layout()  # Adjust layout to prevent label clipping
        plt.savefig("correlation.png", format='png', dpi=300)  # Save with higher resolution
        
        # Save image to a BytesIO object to send it to GPT as base64
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        plt.close()
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')  # base64-encoded image
        
        return img_base64

# Second cell: Main execution
def main():
    # Get the file path from command line argument
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <csv filename>")
        sys.exit(1)

    file = sys.argv[1]
    
    analyzer = DataAnalyzer()
    results = {}
    df, stats = analyzer.analyze_csv(file)
    analyzer.analysis_results['summary_stats'] = analyzer.generate_summary_statistics1(df)
    data_quality = analyzer.analyze_data_quality(df)
    print(data_quality)
    if df is not None and stats is not None:
         results[file] = {
            'dataframe': df,
            'stats': stats
         }
         
    # Run all analyses
    print("Running analyses on csv...")
    analyzer.detect_outliers_and_anomalies(df)
    analyzer.analyze_correlations(df)
    analyzer.analyze_time_series(df)
    analyzer.analyze_clusters(df)
    analyzer.analyze_geographic(df)
    analyzer.analyze_network(df)

    # Generate summary report
    summary = analyzer.generate_summary_report()
    
    # Print key findings
    print("\
    Key Findings from Happiness Dataset Analysis:")
    print("============================================")
    for analysis_type, results in summary.items():
        if analysis_type != 'dataset_overview':
            print(f"\
    {analysis_type.upper()}:")
            for finding in results['key_findings']:
                print(f"- {finding}")

    # Save results to JSON for future LLM processing
    #with open('happiness_analysis.json', 'w') as f:
        #json.dump(summary, f, indent=2)
        
    #print(stats)
    img_base64 = analyzer.generate_correlation_heatmap(df)
        
    analyzer.generate_llm_report(img_base64)


if __name__ == "__main__":
    main()

