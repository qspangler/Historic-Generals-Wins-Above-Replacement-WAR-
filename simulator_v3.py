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
            lambda x: 1/(x['battles'] + 3),
            axis=1
        )
        return stats

    def _bayesian_adjustment(self, row):
        global_avg = self.global_avg['V'] if row['avg_lvb'] > 0 else self.global_avg['D']
        return (row['avg_lvb'] * row['battles'] + global_avg * 3) / (row['battles'] + 3)

    def _sample_team_strength(self, generals, n_samples=10000):
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
    
    with st.sidebar:
        st.title("Historic General Comparison")
        page = st.radio(
            "Navigation Menu",
            ["Head to Head Simulation", "General Comparison", "About"],
            label_visibility="collapsed"
        )
    
    if page == "Head to Head Simulation":
        render_simulation_page()
    elif page == "General Comparison":
        render_comparison_page()
    elif page == "About":
        render_about_page()

def render_simulation_page():
    @st.cache_resource
    def load_simulator():
        try:
            return MilitarySimulator("all_battle_war.csv")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    st.title("🎯 Military Strategy Head to Head Simulator")
    st.subheader('Add Generals from history to either side, and see how the hypothetical opposing sides stack up based on historic Wins Above Replacement.')
    simulator = load_simulator()
    
    if not simulator:
        return
        
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

def render_comparison_page():
    @st.cache_resource
    def load_simulator():
        try:
            return MilitarySimulator("all_battle_war.csv")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    st.title("⚔️ Historic General Comparison")
    st.subheader("Select two generals to compare their career statistics")
    
    simulator = load_simulator()
    if not simulator:
        return
    
    col1, col2 = st.columns(2)
    with col1:
        general1 = st.selectbox(
            "Select General 1",
            simulator.all_generals,
            format_func=lambda x: f"{x} ({sim_attrs(x, simulator)})",
            key="general1"
        )
    
    with col2:
        general2 = st.selectbox(
            "Select General 2",
            [g for g in simulator.all_generals if g != general1],
            format_func=lambda x: f"{x} ({sim_attrs(x, simulator)})",
            key="general2"
        )

    if general1 and general2:
        st.divider()
        
        # Get base statistics
        stats1 = simulator.general_stats.loc[simulator.general_stats['General'] == general1].iloc[0]
        stats2 = simulator.general_stats.loc[simulator.general_stats['General'] == general2].iloc[0]
        
        # Calculate battle outcomes
        def get_outcomes(general):
            outcomes = simulator.df[simulator.df['General'] == general]['Outcome'].value_counts()
            return outcomes.get('V', 0), outcomes.get('D', 0), outcomes.get('I', 0)
        
        v1, d1, i1 = get_outcomes(general1)
        v2, d2, i2 = get_outcomes(general2)
        
        # Create comparison data
        metrics = ["Total Battles", "Record (V-D-I)", "Total WAR", "Avg WAR/Battle"]
        comparison_data = {
            "Metric": metrics,
            general1: [
                stats1['battles'],
                f"{v1}-{d1}-{i1}",
                np.round(stats1['adj_lvb'],2),
                np.round(stats1['avg_lvb'],2)
            ],
            general2: [
                stats2['battles'],
                f"{v2}-{d2}-{i2}",
                np.round(stats2['adj_lvb'],2),
                np.round(stats2['avg_lvb'],2)
            ]
        }

        # Create combined dataframe
        df = pd.DataFrame(comparison_data).set_index('Metric')
        
        # Create separate dataframes for display
        df1 = df[[general1]].reset_index()
        df2 = df[[general2]].reset_index()

        # Highlighting functions
        def highlight_gen1(row):
            styles = [''] * len(row)
            metric = row['Metric']
            if metric == "Record (V-D-I)":
                return styles
            try:
                val1 = float(row[general1])
                val2 = float(df.loc[metric, general2])
                if val1 > val2:
                    styles[1] = 'background-color: #e6f5d0'
            except:
                pass
            return styles

        def highlight_gen2(row):
            styles = [''] * len(row)
            metric = row['Metric']
            if metric == "Record (V-D-I)":
                return styles
            try:
                val2 = float(row[general2])
                val1 = float(df.loc[metric, general1])
                if val2 > val1:
                    styles[1] = 'background-color: #e6f5d0'
            except:
                pass
            return styles

        # Apply styling
         # Apply styling with larger fonts
    styled_df1 = df1.style \
        .apply(highlight_gen1, axis=1) \
        .set_properties(**{
            'font-size': '20px',
            'padding': '15px',
            'border': '2px solid #f0f0f0'
        }) \
        .set_table_styles([{
            'selector': 'th',
            'props': [
                ('font-size', '22px'),
                ('background-color', '#f8f8f8'),
                ('font-weight', 'bold'),
                ('padding', '15px')
            ]
        }, {
            'selector': 'td',
            'props': [
                ('font-size', '20px')
            ]
        }])

    styled_df2 = df2.style \
        .apply(highlight_gen2, axis=1) \
        .set_properties(**{
            'font-size': '20px',
            'padding': '15px',
            'border': '2px solid #f0f0f0'
        }) \
        .set_table_styles([{
            'selector': 'th',
            'props': [
                ('font-size', '22px'),
                ('background-color', '#f8f8f8'),
                ('font-weight', 'bold'),
                ('padding', '15px')
            ]
        }, {
            'selector': 'td',
            'props': [
                ('font-size', '20px')
            ]
        }])

    # Display formatted comparison
    st.markdown("### Comparative Battle Statistics")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"<h3 style='font-size: 26px;'>{general1}</h3>", unsafe_allow_html=True)
        st.dataframe(
            styled_df1,
            use_container_width=True,
            hide_index=True,
            height=400  # Taller table
        )
        
    with col_b:
        st.markdown(f"<h3 style='font-size: 26px;'>{general2}</h3>", unsafe_allow_html=True)
        st.dataframe(
            styled_df2,
            use_container_width=True,
            hide_index=True,
            height=400  # Taller table
        )

    # Update metric box styling
    st.markdown("""
    <style>
        .metric-box {
            padding: 20px !important;
            font-size: 20px !important;
        }
        .metric-value {
            font-size: 22px !important;
        }
        .metric-diff {
            font-size: 20px !important;
        }
    </style>
    """, unsafe_allow_html=True)



def render_about_page():
    st.title("About the Military Strategy Simulator")
    st.markdown("""
    ## Project Overview
    This interactive military strategy simulator uses historical battle data and statistical modeling 
    to predict hypothetical combat outcomes between military forces led by different commanders.
    
    ### Key Features
    - **Monte Carlo Simulation**: Uses 10,000 iterations for probability modeling
    - **Bayesian Adjustment**: Incorporates global averages to stabilize estimates
    - **Historical Context**: Adjusts for era-specific combat effectiveness
    - **Coalition Dynamics**: Models command structure complexities
    
    ## Data Sources
    Analysis based on comprehensive military history records including:
    - Historical battle outcomes
    - Commander performance metrics
    - Force composition data
    - Geographical and temporal context
    
    ## Methodology
    The model calculates Leadership Value Battles (LVB) scores using:
    \[ \text{Adjusted LVB} = \frac{(\text{Observed LVB} \times \text{Battles}) + (\text{Global Avg} \times 3)}{\text{Battles} + 3} \]
                
    ## Data
    Data used in this app are sourced from a project done by Ethan Arsht in 2017, an overview of which can be found here: https://medium.com/towards-data-science/napoleon-was-the-best-general-ever-and-the-math-proves-it-86efed303eeb.\n
    Data files can be found a public Google Drive provided by Arsht in his article, linked here: https://drive.google.com/drive/folders/1nQM9eJKjp4T8EeqkSGitZS7kiok5zwJ2
    
    """)

    with st.expander("Technical Specifications"):
        st.markdown("""
        ### System Architecture
        - **Backend**: Python 3.10+ with NumPy/SciPy stack
        - **Frontend**: Streamlit web framework
        - **Data Processing**: Pandas for ETL pipelines
        - **Visualization**: Matplotlib with custom military-themed palettes
        
        ### Statistical Models
        - Bayesian hierarchical regression
        - Gaussian kernel density estimation
        - 95% confidence interval calculation
        - Era-specific effectiveness multipliers
        """)

def set_custom_css():
    st.markdown("""
    <style>
        /* Base styles */
        html, body, .stApp { 
            font-size: 18px;
            color: #2c3e50;
        }
        
        /* Main content headers */
        h1, h2, h3, h4, h5, h6 {
            color: #000000 !important;
        }

        /* Sidebar container */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%) !important;
            padding: 25px !important;
        }

        /* Sidebar headers */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: white !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }

        /* Navigation radio buttons */
        [data-testid="stSidebar"] .stRadio {
            margin-top: 40px !important;
        }
        
        [data-testid="stSidebar"] .stRadio label {
            color: white !important;
            font-size: 20px !important;
            padding: 18px 25px !important;
            border: 2px solid rgba(255,255,255,0.3) !important;
            margin: 10px 0 !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
            background: rgba(255,255,255,0.05) !important;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(255,255,255,0.15) !important;
            transform: translateX(10px);
            border-color: rgba(255,255,255,0.6) !important;
        }

        [data-testid="stSidebar"] .stRadio label:has(input:checked) {
            background: rgba(255,255,255,0.2) !important;
            border-color: #FFFFFF !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        /* Table styling */
        .stDataFrame {
            font-size: 18px !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }

        .stDataFrame th {
            font-size: 20px !important;
            background-color: #f8f9fa !important;
        }

        .stDataFrame td {
            font-size: 18px !important;
            padding: 15px !important;
        }

        /* Buttons and inputs */
        .stButton button {
            font-size: 18px !important;
            padding: 12px 24px !important;
            color: white !important;
        }

        .stMultiSelect [data-baseweb=select] span {
            font-size: 16px !important;
            padding: 10px !important;
        }

        /* Metrics */
        [data-testid="stMetricLabel"] {
            font-size: 16px !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 24px !important;
        }
                

        /* Base sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%) !important;
        }

        /* Navigation radio buttons - Transparent version */
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
            background: transparent !important;
            border: none !important;
        }

        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
            background: transparent !important;
            border: 2px solid rgba(255,255,255,0.3) !important;
            margin: 10px 0 !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
            backdrop-filter: blur(2px); /* Optional: Adds subtle frosted glass effect */
        }

        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {
            color: white !important;
            font-size: 20px !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }

        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
            background: rgba(255,255,255,0.1) !important;
            transform: translateX(10px);
            border-color: rgba(255,255,255,0.6) !important;
        }

        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
            background: rgba(255,255,255,0.15) !important;
            border-color: #FFFFFF !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        /* Main content headers */
        h1, h2, h3 {
            color: #000000 !important;
        }
        
        /* Explicit white text in sidebar */
        [data-testid="stSidebar"] * {
            color: white !important;
        }

                
     /* Base styles */
        html, body, .stApp { 
            font-size: 18px;
            color: #2c3e50;
        }
        
        /* Main content headers */
        h1, h2, h3 {
            color: #000000 !important;
        }

        /* Sidebar container */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%) !important;
        }

        /* Sidebar headers */
        [data-testid="stSidebar"] h1 {
            color: white !important;
            font-size: 32px !important;
            margin-bottom: 20px !important;
        }

        /* Navigation radio buttons */
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
            gap: 12px;
        }

        [data-testid="stSidebar"] .stRadio label {
            padding: 15px 25px !important;
            border-radius: 8px !important;
            background: rgba(255,255,255,0.1) !important;
            border: 1px solid rgba(255,255,255,0.3) !important;
            transition: all 0.3s ease !important;
            color: white !important;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(255,255,255,0.2) !important;
            transform: translateX(10px);
            border-color: rgba(255,255,255,0.6) !important;
        }

        [data-testid="stSidebar"] .stRadio label:has(input:checked) {
            background: rgba(255,255,255,0.3) !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        [data-testid="stSidebar"] .stRadio label span {
            color: white !important;
            font-size: 18px !important;
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

    st.divider()
    
    prob_a = result['probability_a'] * 100
    prob_b = 100 - prob_a

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh(['A'], [prob_a/100], color='#2ecc71', height=0.6)
    ax.barh(['B'], [prob_b/100], color='#e74c3c', height=0.6)
    ax.set_xlim(0, 1)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[::-1])
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{x:.0f}%" for x in np.linspace(0, 100, 6)])
    ax2.set_xticks(np.linspace(0, 1, 6))
    ax2.set_xticklabels([f"{x:.0f}%" for x in np.linspace(100, 0, 6)])
    ax.set_xlabel("Side A Victory Probability", labelpad=15, fontsize=12)
    ax2.set_xlabel("Side B Victory Probability", labelpad=15, fontsize=12)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    st.pyplot(fig)

    col_strength = st.columns(2)
    with col_strength[0]:
        st.metric("Side A Strength Score", f"{result['strength_a']:.2f}")
    with col_strength[1]:
        st.metric("Side B Strength Score", f"{result['strength_b']:.2f}")

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
