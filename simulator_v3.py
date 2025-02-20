import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.colors import LinearSegmentedColormap

class MilitarySimulator:
    def __init__(self, data_path):
        self.df = self._load_and_clean(data_path)
        self._validate_data()
        self.global_avg = self._calculate_global_averages()
        self.general_stats = self._preprocess_general_stats()
        self.all_generals = self.general_stats['General'].tolist()
        
    def _load_and_clean(self, path):
        df = pd.read_csv(
            path,
            converters={
                'Year': pd.to_numeric,
                'Value': pd.to_numeric,
                'Battle': lambda x: str(x).replace('â€“', '-').strip(),
                'General': str.strip,
                'Outcome': lambda x: str(x).strip().upper()
            }
        )
        df = df.dropna(subset=['Year', 'Value', 'General'])
        return df

    def _validate_data(self):
        if self.df.empty:
            raise ValueError("No valid data remaining after cleaning")
            
        valid_outcomes = {'V', 'D', 'I'}
        invalid = set(self.df['Outcome']) - valid_outcomes
        if invalid:
            raise ValueError(f"Invalid outcomes: {invalid}")

    def _calculate_global_averages(self):
        victors = self.df[self.df['Outcome'] == 'V']['Value'].mean()
        defeated = self.df[self.df['Outcome'] == 'D']['Value'].mean()
        return {'V': victors, 'D': defeated, 'I': 0.0}

    def _preprocess_general_stats(self):
        stats = self.df.groupby('General').agg(
            battles=('Battle', 'size'),
            avg_lvb=('Value', 'mean'),
            first_year=('Year', 'min'),
            last_year=('Year', 'max')
        ).reset_index()

        stats['adj_lvb'] = stats.apply(self._bayesian_adjustment, axis=1)
        stats['lvb_std'] = stats.apply(
            lambda x: 1/(x['battles'] + 3),  # Simplified uncertainty model
            axis=1
        )
        return stats

    def _bayesian_adjustment(self, row):
        global_avg = self.global_avg['V'] if row['avg_lvb'] > 0 else self.global_avg['D']
        return (row['avg_lvb'] * row['battles'] + global_avg * 3) / (row['battles'] + 3)

    def _sample_team_strength(self, generals, n_samples=10000):
        """Monte Carlo sampling of team strength with uncertainty"""
        if not generals:
            return np.zeros(n_samples)
            
        df = self.general_stats[self.general_stats['General'].isin(generals)]
        samples = np.zeros(n_samples)
        
        for _, general in df.iterrows():
            samples += np.random.normal(general.adj_lvb, general.lvb_std, n_samples)
            
        era_mult = 1.2 if any(y >= 1967 for y in df['last_year']) else 1.0
        coalition_mult = max(0.75, 1 - 0.05 * len(generals))
        
        return samples * era_mult * coalition_mult

    def simulate_battle(self, team_a, team_b, n_samples=10000):
        """Returns battle outcome with Monte Carlo samples"""
        a_samples = self._sample_team_strength(team_a, n_samples)
        b_samples = self._sample_team_strength(team_b, n_samples)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            prob_a_samples = np.abs(a_samples) / (np.abs(a_samples) + np.abs(b_samples))
            
        return {
            'team_a': team_a,
            'team_b': team_b,
            'samples': prob_a_samples,
            'strength_a': np.nanmean(a_samples),
            'strength_b': np.nanmean(b_samples),
            'probability_a': np.nanmean(prob_a_samples),
            'ci_low': np.nanquantile(prob_a_samples, 0.05),
            'ci_high': np.nanquantile(prob_a_samples, 0.95)
        }

def main():
    st.set_page_config(page_title="Military Strategy Simulator", layout="wide")
    set_custom_css()
    
    @st.cache_resource
    def load_simulator():
        try:
            return MilitarySimulator("all_battle_war.csv")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    st.sidebar.title("Navigation")
    simulator = load_simulator()
    
    if not simulator:
        return
        
    st.title("🎯 Military Strategy Simulator")
    
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.multiselect(
            "Select Side A Commanders",
            simulator.all_generals,
            format_func=lambda x: f"{x} ({sim_attrs(x, simulator)})",
            key="team_a"
        )
        
    with col2:
        team_b = st.multiselect(
            "Select Side B Commanders",
            simulator.all_generals,
            format_func=lambda x: f"{x} ({sim_attrs(x, simulator)})",
            key="team_b"
        )
        
    if st.button("▶️ Simulate Battle", use_container_width=True, type="primary"):
        if not team_a or not team_b:
            st.error("Both sides need at least one commander")
        else:
            with st.spinner("Running 10,000 Monte Carlo simulations..."):
                try:
                    result = simulator.simulate_battle(team_a, team_b)
                    display_result(result, simulator)
                except Exception as e:
                    st.error(f"Simulation failed: {str(e)}")

def set_custom_css():
    st.markdown("""
    <style>
        html, body, .stApp { font-size: 18px; }
        h1 { font-size: 2.6rem !important; }
        h2 { font-size: 2rem !important; }
        h3 { font-size: 1.7rem !important; }
        .stMultiSelect [data-baseweb=select] span { 
            font-size: 1.15rem !important;
            padding: 8px !important;
        }
        [data-testid="stMetricLabel"] { font-size: 1.3rem !important; }
        [data-testid="stMetricValue"] { font-size: 2rem !important; }
        .stDataFrame { font-size: 1.05rem !important; }
        .stButton button { 
            font-size: 1.3rem !important;
            padding: 12px 24px !important;
        }
        [data-testid="stExpander"] .streamlit-expanderHeader { 
            font-size: 1.4rem !important;
        }
        .stProgress > div > div > div { 
            font-size: 1.05rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

def sim_attrs(general, simulator):
    stats = simulator.general_stats
    gen = stats[stats['General'] == general].iloc[0]
    return f"LVB: {gen.adj_lvb:+.2f} ±{gen.lvb_std:.2f} | Battles: {gen.battles}"

def display_result(result, simulator):
    st.divider()
    
    prob_a = result['probability_a'] * 100
    ci_low = result['ci_low'] * 100
    ci_high = result['ci_high'] * 100
    
    # Victory Probability Display
    col_probs = st.columns([1, 2, 1])
    with col_probs[1]:
        st.markdown(f"""
        <div style="text-align: center; margin: 30px 0;">
            <div style="font-size: 2.8em; font-weight: bold; color: {'#2ecc71' if prob_a > 50 else '#e74c3c'}">
                {prob_a:.1f}% 
                <span style="font-size: 0.6em; color: #95a5a6;">
                    (95% CI: {ci_low:.1f}% - {ci_high:.1f}%)
                </span>
            </div>
            <div style="font-size: 1.4em; color: #95a5a6; margin-top: 10px">
                Victory Probability for Side A
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Contour Visualization with Error Handling
    try:
        if len(result['samples']) < 2:
            raise ValueError("Insufficient samples for density estimation")
            
        fig, ax = plt.subplots(figsize=(10, 1))
        x = np.linspace(0, 100, 500)
        kde = stats.gaussian_kde(result['samples'] * 100, bw_method=0.15)
        density = kde(x)
        
        cmap = LinearSegmentedColormap.from_list('military', ['#e74c3c', '#f1c40f', '#2ecc71'])
        cs = ax.contourf(x, [0,1], [density, density], 
                        levels=100, cmap=cmap, alpha=0.6)
        
        ax.axvspan(ci_low, ci_high, color='#3498db', alpha=0.2)
        ax.plot(x, density, color='#2c3e50', linewidth=3)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        st.pyplot(fig)
        
    except ValueError as e:
        st.warning("Could not generate density plot: " + str(e))

    # Individual bar charts
    st.divider()
    
    prob_a = result['probability_a'] * 100
    prob_b = 100 - prob_a

    # Modified Probability Visualization with Dual Axis
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Create main bars
    ax.barh(['A'], [prob_a/100], color='#2ecc71', height=0.6)
    ax.barh(['B'], [prob_b/100], color='#e74c3c', height=0.6)
    
    # Set axis limits
    ax.set_xlim(0, 1)
    
    # Create mirrored top axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[::-1])
    
    # Format axes
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{x:.0f}%" for x in np.linspace(0, 100, 6)])
    ax2.set_xticks(np.linspace(0, 1, 6))
    ax2.set_xticklabels([f"{x:.0f}%" for x in np.linspace(100, 0, 6)])
    
    # Axis labels
    ax.set_xlabel("Side A Victory Probability", labelpad=15, fontsize=12)
    ax2.set_xlabel("Side B Victory Probability", labelpad=15, fontsize=12)
    
    # Remove y-axis and borders
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    st.pyplot(fig)

    # Strength Metrics
    col_strength = st.columns(2)
    with col_strength[0]:
        st.metric("Side A Strength Score", f"{result['strength_a']:.2f}")
    with col_strength[1]:
        st.metric("Side B Strength Score", f"{result['strength_b']:.2f}")

    # Detailed Analysis
    with st.expander("📊 Detailed Battle Analysis", expanded=True):
        col_teams = st.columns(2)
        with col_teams[0]:
            st.write("### Side A Composition")
            st.dataframe(
                get_general_stats(result['team_a'], simulator),
                column_config={
                    "adj_lvb": st.column_config.NumberColumn(
                        "Leadership Value (±)",
                        format="±%.2f"
                    ),
                    "battles": "Experience"
                },
                hide_index=True
            )
        
        with col_teams[1]:
            st.write("### Side B Composition")
            st.dataframe(
                get_general_stats(result['team_b'], simulator),
                column_config={
                    "adj_lvb": st.column_config.NumberColumn(
                        "Leadership Value (±)",
                        format="±%.2f"
                    ),
                    "battles": "Experience"
                },
                hide_index=True
            )

def get_general_stats(generals, simulator):
    return simulator.general_stats[simulator.general_stats['General'].isin(generals)][
        ['General', 'adj_lvb', 'battles', 'last_year']
    ].reset_index(drop=True)

if __name__ == "__main__":
    main()
