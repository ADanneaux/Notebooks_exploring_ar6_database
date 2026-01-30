"""
AR6 Scenario Database Plotting Tools
=====================================
Helper functions for visualizing IPCC AR6 scenario data.

These functions are designed to be easy to use for students exploring
the AR6 scenario database.
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe

# Suppress warnings that students don't need to see
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Division by zero, etc.
warnings.filterwarnings('ignore', message='.*Glyph.*missing from current font.*')
warnings.filterwarnings('ignore', message='.*Filtered IamDataFrame is empty.*')
warnings.filterwarnings('ignore', message='.*No artists with labels.*')

# Define colors for each scenario category
# These colors follow the IPCC AR6 color scheme
CATEGORY_COLORS = {
    "C1": "#97CEE4",   # Light blue - 1.5°C with no/limited overshoot
    "C2": "#778663",   # Green - 1.5°C with high overshoot
    "C3": "#6F7899",   # Purple-grey - Likely below 2°C
    "C4": "#A7C682",   # Light green - Below 2°C
    "C5": "#8CA7D0",   # Blue - Below 2.5°C
    "C6": "#FAC182",   # Orange - Below 3°C
    "C7": "#F18872",   # Salmon - Below 4°C
    "C8": "#BD7161",   # Brown - Above 4°C
}

# List of all categories in order
ALL_CATEGORIES = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

# ============================================================================
# ILLUSTRATIVE MITIGATION PATHWAYS (IMPs)
# These are key scenarios highlighted in the IPCC AR6 report
# ============================================================================

# Mapping of IMP short names to their scenario names in the database
IMP_SCENARIOS = {
    'CurPol': 'NGFS2_Current Policies',      # Current Policies
    'ModAct': 'EN_INDCi2030_3000f',          # Moderate Action (NDCs) - note leading space
    'Neg': 'EN_NPi2020_400f_lowBECCS',        # Heavy reliance on Negative emissions
    'Ren': 'DeepElec_SSP2_ HighRE_Budg900',   # High Renewables
    'LD': 'LowEnergyDemand_1.3_IPCC',         # Low Demand
    'GS': 'CO_Bridge',                        # Gradual Strengthening - note leading space
    'SP': 'SusDev_SDP-PkBudg1000',            # Shifting Pathways (Sustainable Dev)
}

# Details for each IMP: [model_name, color]
IMP_DETAILS = {
    'CurPol': ['GCAM 5.3', '#E31F2B'],                    # Red
    'ModAct': ['IMAGE 3.0', '#F29424'],                   # Orange
    'Neg': ['COFFEE 1.1', '#84A12B'],                     # Green
    'Ren': ['REMIND-MAgPIE 2.1-4.3', '#2B7C8B'],         # Teal
    'LD': ['MESSAGEix-GLOBIOM 1.0', '#4FA7BF'],          # Light blue
    'GS': ['WITCH 5.0', '#6E7895'],                       # Grey-blue
    'SP': ['REMIND-MAgPIE 2.1-4.2', '#004D52'],          # Dark teal
}

# Path effect for highlighting IMP lines (black outline)
IMP_PATH_EFFECT = [mpe.withStroke(linewidth=2.5, foreground='black')]


def _get_category_column(meta):
    """Find the category column in the metadata."""
    if 'Category' in meta.columns:
        return 'Category'
    elif 'category' in meta.columns:
        return 'category'
    else:
        cat_cols = [col for col in meta.columns if 'categ' in col.lower()]
        if cat_cols:
            return cat_cols[0]
    return None


def plot_timeseries_by_category(df_pyam, variable, categories=None, alpha=0.3, 
                                 figsize=(8, 4), ax=None, show_imps=False):
    """
    Plot time series for a variable, color-coded by scenario category.
    
    Parameters:
    -----------
    df_pyam : pyam.IamDataFrame
        The pyam dataframe containing the data
    variable : str
        The variable to plot (e.g., 'Emissions|CO2|Energy and Industrial Processes')
    categories : list, optional
        List of categories to include (e.g., ['C1', 'C2', 'C3']). 
        If None, all categories are included.
    alpha : float
        Transparency of the lines (0 = invisible, 1 = solid)
    figsize : tuple
        Size of the figure (width, height). Only used if ax is None.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    show_imps : bool, optional
        If True, highlight the Illustrative Mitigation Pathways (IMPs) on top
        with colored lines and black outline. Default is False.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Determine which categories to plot
    cats_to_plot = categories if categories is not None else ALL_CATEGORIES
    
    # Create figure if no axis provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False
    
    # Get the meta data with categories
    meta = df_pyam.meta
    cat_col = _get_category_column(meta)
    
    if cat_col is None:
        print("⚠️ Category column not found in metadata")
        return fig, ax
    
    # Filter data for the selected variable
    df_filtered = df_pyam.filter(variable=variable)
    df_ts = df_filtered.timeseries()
    
    # Sort columns (years) in chronological order
    df_ts = df_ts.reindex(columns=sorted(df_ts.columns))
    
    # Plot each scenario
    legend_handles = {}
    for idx in df_ts.index:
        model, scenario = idx[0], idx[1]
        
        # Get the category for this scenario
        try:
            category = meta.loc[(model, scenario), cat_col]
            if pd.isna(category) or category not in CATEGORY_COLORS:
                continue
            # Skip if category not in the list to plot
            if category not in cats_to_plot:
                continue
            color = CATEGORY_COLORS[category]
        except:
            continue
        
        # Plot the line (filter out NaN values)
        years = np.array(df_ts.columns)
        values = df_ts.loc[idx].values
        mask = ~np.isnan(values)
        if mask.sum() == 0:
            continue
        line, = ax.plot(years[mask], values[mask], color=color, alpha=alpha, linewidth=0.8)
        
        # Store one handle per category for the legend
        if category not in legend_handles:
            legend_handles[category] = line
    
    # Plot Illustrative Mitigation Pathways (IMPs) on top if requested
    imp_legend_handles = {}
    if show_imps:
        for imp_name, scenario_name in IMP_SCENARIOS.items():
            model_name, imp_color = IMP_DETAILS[imp_name]
            
            # Find the IMP in the data by searching through the index
            # The index may have multiple levels: (model, scenario, region, variable, unit)
            found = False
            for idx in df_ts.index:
                idx_model = idx[0]
                idx_scenario = idx[1]
                
                # Check if this is the IMP we're looking for
                if idx_model == model_name and idx_scenario == scenario_name:
                    years = np.array(df_ts.columns)
                    values = df_ts.loc[idx].values
                    
                    # Filter out NaN values
                    mask = ~np.isnan(values)
                    if mask.sum() == 0:
                        continue
                    
                    # Plot with path effect (black outline)
                    line, = ax.plot(years[mask], values[mask], color=imp_color, linewidth=2, 
                                   zorder=10, path_effects=IMP_PATH_EFFECT)
                    imp_legend_handles[imp_name] = line
                    found = True
                    break
            
            # Debug: if not found, print what we were looking for
            if not found and False:  # Set to True for debugging
                print(f"IMP not found: {imp_name} ({model_name}, {scenario_name})")
    
    # Create legend (in order of categories)
    sorted_handles = [(cat, legend_handles[cat]) for cat in cats_to_plot if cat in legend_handles]
    
    # Add IMP handles to legend if they exist
    if show_imps and imp_legend_handles:
        # Add a separator in the legend
        for imp_name in IMP_SCENARIOS.keys():
            if imp_name in imp_legend_handles:
                sorted_handles.append((f'IMP-{imp_name}', imp_legend_handles[imp_name]))
    
    if sorted_handles:
        ax.legend([h for _, h in sorted_handles], [c for c, _ in sorted_handles], 
                  title='Category', loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9)
    
    # Labels and title
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(f'{variable}', fontsize=10)
    title_suffix = " (with IMPs)" if show_imps else ""
    ax.set_title(f'Time Series: {variable}\n(colored by scenario category){title_suffix}', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    if created_fig:
        plt.tight_layout()
        plt.show()
    
    return fig, ax


def plot_scatter_boxplot_by_category(df_pyam, variable, year, categories=None,
                                      figsize=(8, 4), ax=None, show_imps=False):
    """
    Create a scatter plot with boxplots showing the distribution of values 
    at a specific year, grouped by scenario category.
    
    Parameters:
    -----------
    df_pyam : pyam.IamDataFrame
        The pyam dataframe containing the data
    variable : str
        The variable to plot
    year : int
        The year to analyze (e.g., 2030, 2050, 2100)
    categories : list, optional
        List of categories to include (e.g., ['C1', 'C2', 'C3']). 
        If None, all categories are included.
    figsize : tuple
        Size of the figure (width, height). Only used if ax is None.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    show_imps : bool, optional
        If True, highlight the Illustrative Mitigation Pathways (IMPs) 
        with square markers and black borders. Default is False.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Determine which categories to plot
    cats_to_plot = categories if categories is not None else ALL_CATEGORIES
    
    # Create figure if no axis provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False
    
    # Get the meta data with categories
    meta = df_pyam.meta
    cat_col = _get_category_column(meta)
    
    if cat_col is None:
        print("⚠️ Category column not found in metadata")
        return fig, ax
    
    # Filter data for the selected variable
    df_filtered = df_pyam.filter(variable=variable)
    df_ts = df_filtered.timeseries()
    
    # Sort columns (years) in chronological order
    df_ts = df_ts.reindex(columns=sorted(df_ts.columns))
    
    # Check if the year exists in the data
    if year not in df_ts.columns:
        print(f"⚠️ Year {year} not found in data. Available years: {sorted(df_ts.columns)}")
        return fig, ax
    
    # Get values at the selected year
    values_at_year = df_ts[year]
    
    # Prepare data for each category
    data_by_category = {cat: [] for cat in cats_to_plot}
    
    for idx in values_at_year.index:
        model, scenario = idx[0], idx[1]
        value = values_at_year.loc[idx]
        
        if pd.isna(value):
            continue
            
        # Get the category for this scenario
        try:
            category = meta.loc[(model, scenario), cat_col]
            if pd.isna(category) or category not in cats_to_plot:
                continue
            data_by_category[category].append(value)
        except:
            continue
    
    # Plot scatter points and boxplots for each category
    positions = []
    boxplot_data = []
    scatter_positions = []
    scatter_values = []
    scatter_colors = []
    
    for i, cat in enumerate(cats_to_plot):
        if data_by_category[cat]:
            positions.append(i)
            boxplot_data.append(data_by_category[cat])
            
            # Add scatter points with jitter
            for val in data_by_category[cat]:
                scatter_positions.append(i + np.random.uniform(-0.2, 0.2))
                scatter_values.append(val)
                scatter_colors.append(CATEGORY_COLORS[cat])
    
    # Plot scatter points
    ax.scatter(scatter_positions, scatter_values, c=scatter_colors, alpha=0.5, s=20, zorder=2)
    
    # Plot boxplots
    if boxplot_data:
        bp = ax.boxplot(boxplot_data, positions=positions, widths=0.5, patch_artist=True,
                        showfliers=False, zorder=3)
        
        # Color the boxplots
        for patch, pos in zip(bp['boxes'], positions):
            cat = cats_to_plot[pos]
            patch.set_facecolor(CATEGORY_COLORS[cat])
            patch.set_alpha(0.7)
        
        # Style the boxplot elements
        for element in ['whiskers', 'caps']:
            for line in bp[element]:
                line.set_color('black')
                line.set_linewidth(1)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
    
    # Add count labels below each category
    for i, cat in enumerate(cats_to_plot):
        n = len(data_by_category[cat])
        if n > 0:
            ax.text(i, ax.get_ylim()[0], f'n={n}', ha='center', va='top', fontsize=9)
    
    # Plot Illustrative Mitigation Pathways (IMPs) if requested
    imp_legend_handles = {}
    if show_imps:
        for imp_name, scenario_name in IMP_SCENARIOS.items():
            model_name, imp_color = IMP_DETAILS[imp_name]
            
            # Find the IMP in the data and get its category
            for idx in df_ts.index:
                idx_model = idx[0]
                idx_scenario = idx[1]
                
                if idx_model == model_name and idx_scenario == scenario_name:
                    # Get the category for this IMP
                    try:
                        imp_category = meta.loc[(model_name, scenario_name), cat_col]
                        
                        # Only plot if the IMP's category is in the categories being plotted
                        if pd.isna(imp_category) or imp_category not in cats_to_plot:
                            break
                        
                        # Get the value at this year
                        value = df_ts.loc[idx, year]
                        if pd.isna(value):
                            break
                        
                        # Find the x position for this category
                        cat_position = cats_to_plot.index(imp_category)
                        
                        # Plot with square marker, IMP color, black border, high zorder
                        ax.scatter(cat_position, value, 
                                  marker='s',  # square marker
                                  s=80,  # size
                                  c=imp_color, 
                                  edgecolors='black', 
                                  linewidths=1.5,
                                  zorder=10,  # in front of everything
                                  label=f'IMP-{imp_name}')
                        imp_legend_handles[imp_name] = (cat_position, value, imp_color)
                    except:
                        pass
                    break
    
    # Customize the plot
    ax.set_xticks(range(len(cats_to_plot)))
    ax.set_xticklabels(cats_to_plot)
    ax.set_xlabel('Scenario Category', fontsize=12)
    ax.set_ylabel(variable, fontsize=10)
    title_suffix = " (with IMPs)" if show_imps else ""
    ax.set_title(f'{variable}\nin year {year}{title_suffix}', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend for IMPs if they were plotted
    if show_imps and imp_legend_handles:
        # Create custom legend entries for IMPs
        from matplotlib.lines import Line2D
        imp_handles = []
        imp_labels = []
        for imp_name in IMP_SCENARIOS.keys():
            if imp_name in imp_legend_handles:
                _, _, imp_color = imp_legend_handles[imp_name]
                handle = Line2D([0], [0], marker='s', color='w', markerfacecolor=imp_color,
                               markeredgecolor='black', markersize=10, linewidth=0)
                imp_handles.append(handle)
                imp_labels.append(f'IMP-{imp_name}')
        if imp_handles:
            ax.legend(imp_handles, imp_labels, loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9)
    
    if created_fig:
        plt.tight_layout()
        plt.show()
    
    return fig, ax


def show_color_palette():
    """Display the category color palette."""
    fig, ax = plt.subplots(figsize=(6.7, 1))
    for i, (cat, color) in enumerate(CATEGORY_COLORS.items()):
        ax.barh(0, 1, left=i, color=color, edgecolor='white', linewidth=2)
        ax.text(i + 0.5, 0, cat, ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    ax.set_title('Scenario Category Color Palette', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig, ax

def show_imp_palette():
    """Display the Illustrative Mitigation Pathways (IMP) color palette with descriptions."""
    imp_descriptions = {
        'CurPol': 'Current Policies',
        'ModAct': 'Moderate Action (NDCs)',
        'Neg': 'Negative Emissions',
        'Ren': 'High Renewables',
        'LD': 'Low Demand',
        'GS': 'Gradual Strengthening',
        'SP': 'Shifting Pathways',
    }
    
    fig, ax = plt.subplots(figsize=(8, 1.3))
    n_imps = len(IMP_SCENARIOS)
    
    for i, (imp_name, scenario_name) in enumerate(IMP_SCENARIOS.items()):
        model, color = IMP_DETAILS[imp_name]
        ax.barh(0, 1, left=i, color=color, edgecolor='white', linewidth=2)
        ax.text(i + 0.5, 0.1, imp_name, ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(i + 0.5, -0.1, imp_descriptions[imp_name], ha='center', va='top', fontsize=8)
    
    ax.set_xlim(0, n_imps)
    ax.set_ylim(-0.6, 0.6)
    ax.axis('off')
    ax.set_title('Illustrative Mitigation Pathways (IMPs)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig, ax


# ============================================================================
# KAYA EQUATION PLOTTING FUNCTIONS
# ============================================================================

# Define Kaya variables configuration
KAYA_VARIABLES = {
    'CO2': ('Emissions|CO2|Energy and Industrial Processes', 'Mt CO2/yr', 'CO2 Emissions'),
    'GDP': ('GDP|PPP', 'billion US$2010/yr', 'GDP (PPP)'),
    'Population': ('Population', 'billion people', 'Population'),
    'Energy': ('Primary Energy', 'EJ/yr', 'Primary Energy'),
}

KAYA_RATIOS = {
    'GDP per Capita': ('GDP|PPP', 'Population', 'US$2010/person'),
    'Energy Intensity': ('Primary Energy', 'GDP|PPP', 'EJ/billion US$2010'),
    'Carbon Intensity': ('Emissions|CO2|Energy and Industrial Processes', 'Primary Energy', 'Mt CO2/EJ'),
}


def plot_kaya_variables(df_pyam, mode='imps', categories=None, alpha=0.3, figsize=(8, 5.3)):
    """
    Plot the four Kaya variables (CO2, GDP, Population, Primary Energy) in a 2x2 grid.
    
    Parameters:
    -----------
    df_pyam : pyam.IamDataFrame
        The pyam dataframe containing the data
    mode : str
        'imps' - Plot only IMPs with colored lines
        'categories' - Plot all scenarios colored by category  
        'both' - Plot category-colored scenarios with IMPs highlighted on top
    categories : list, optional
        List of categories to include (e.g., ['C1', 'C2', 'C3']). 
        If None, all categories are included.
    alpha : float
        Transparency of category lines (only used when mode is 'categories' or 'both')
    figsize : tuple
        Size of the figure (width, height)
    
    Returns:
    --------
    fig, axs : matplotlib figure and axes objects
    """
    cats_to_plot = categories if categories is not None else ALL_CATEGORIES
    
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    meta = df_pyam.meta
    cat_col = _get_category_column(meta)
    
    plots_config = [
        KAYA_VARIABLES['CO2'],
        KAYA_VARIABLES['GDP'],
        KAYA_VARIABLES['Population'],
        KAYA_VARIABLES['Energy'],
    ]
    
    for ax, (var, ylabel, title) in zip(axs.flatten(), plots_config):
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        # Get data for this variable
        df_filtered = df_pyam.filter(variable=var)
        if len(df_filtered) == 0:
            ax.set_title(f'{title}\n(no data available)', fontsize=12, fontweight='bold')
            continue
            
        df_ts = df_filtered.timeseries()
        df_ts = df_ts.reindex(columns=sorted(df_ts.columns))
        
        # Plot categories if requested
        if mode in ['categories', 'both']:
            for idx in df_ts.index:
                model, scenario = idx[0], idx[1]
                
                try:
                    category = meta.loc[(model, scenario), cat_col]
                    if pd.isna(category) or category not in cats_to_plot:
                        continue
                    color = CATEGORY_COLORS[category]
                except:
                    continue
                
                years = np.array(df_ts.columns)
                values = df_ts.loc[idx].values
                mask = ~np.isnan(values)
                if mask.sum() == 0:
                    continue
                ax.plot(years[mask], values[mask], color=color, alpha=alpha, linewidth=0.5)
        
        # Plot IMPs if requested
        if mode in ['imps', 'both']:
            for imp_name, scenario_name in IMP_SCENARIOS.items():
                model_name, imp_color = IMP_DETAILS[imp_name]
                
                for idx in df_ts.index:
                    if idx[0] == model_name and idx[1] == scenario_name:
                        years = np.array(df_ts.columns)
                        values = df_ts.loc[idx].values
                        mask = ~np.isnan(values)
                        
                        if mask.sum() > 0:
                            ax.plot(years[mask], values[mask], label=imp_name, 
                                   color=imp_color, path_effects=IMP_PATH_EFFECT, 
                                   linewidth=1.7, zorder=10)
                        break
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Year')
    
    # Add shared legend outside to the right
    if mode in ['imps', 'both']:
        handles, labels = axs[1, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    mode_label = {'imps': 'IMPs only', 'categories': 'All scenarios by category', 'both': 'All scenarios + IMPs'}
    plt.suptitle(f'Kaya Variables ({mode_label.get(mode, mode)})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig, axs


def plot_kaya_ratios(df_pyam, mode='imps', categories=None, alpha=0.3, figsize=(9.3, 2.7)):
    """
    Plot the three Kaya ratios (GDP/capita, Energy/GDP, CO2/Energy) in a 1x3 grid.
    
    Parameters:
    -----------
    df_pyam : pyam.IamDataFrame
        The pyam dataframe containing the data
    mode : str
        'imps' - Plot only IMPs with colored lines
        'categories' - Plot all scenarios colored by category  
        'both' - Plot category-colored scenarios with IMPs highlighted on top
    categories : list, optional
        List of categories to include (e.g., ['C1', 'C2', 'C3']). 
        If None, all categories are included.
    alpha : float
        Transparency of category lines (only used when mode is 'categories' or 'both')
    figsize : tuple
        Size of the figure (width, height)
    
    Returns:
    --------
    fig, axs : matplotlib figure and axes objects
    """
    cats_to_plot = categories if categories is not None else ALL_CATEGORIES
    
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    meta = df_pyam.meta
    cat_col = _get_category_column(meta)
    
    ratios_config = [
        (KAYA_RATIOS['GDP per Capita'][0], KAYA_RATIOS['GDP per Capita'][1], 
         KAYA_RATIOS['GDP per Capita'][2], 'GDP per Capita'),
        (KAYA_RATIOS['Energy Intensity'][0], KAYA_RATIOS['Energy Intensity'][1], 
         KAYA_RATIOS['Energy Intensity'][2], 'Energy Intensity (E/GDP)'),
        (KAYA_RATIOS['Carbon Intensity'][0], KAYA_RATIOS['Carbon Intensity'][1], 
         KAYA_RATIOS['Carbon Intensity'][2], 'Carbon Intensity (CO2/E)'),
    ]
    
    for ax, (var_num, var_den, ylabel, title) in zip(axs.flatten(), ratios_config):
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        # Get data for numerator and denominator
        df_num = df_pyam.filter(variable=var_num)
        df_den = df_pyam.filter(variable=var_den)
        
        if len(df_num) == 0 or len(df_den) == 0:
            ax.set_title(f'{title}\n(no data available)', fontsize=12, fontweight='bold')
            continue
        
        ts_num = df_num.timeseries()
        ts_den = df_den.timeseries()
        
        # Find common years and align columns
        common_years = sorted(set(ts_num.columns) & set(ts_den.columns))
        ts_num = ts_num[common_years]
        ts_den = ts_den[common_years]
        
        # Create lookup dict for denominator by (model, scenario)
        den_lookup = {}
        for idx_den in ts_den.index:
            key = (idx_den[0], idx_den[1])  # (model, scenario)
            den_lookup[key] = idx_den
        
        # Plot categories if requested
        if mode in ['categories', 'both']:
            for idx_num in ts_num.index:
                model, scenario = idx_num[0], idx_num[1]
                key = (model, scenario)
                
                if key not in den_lookup:
                    continue
                
                try:
                    category = meta.loc[(model, scenario), cat_col]
                    if pd.isna(category) or category not in cats_to_plot:
                        continue
                    color = CATEGORY_COLORS[category]
                except:
                    continue
                
                idx_den = den_lookup[key]
                ratio = ts_num.loc[idx_num].values / ts_den.loc[idx_den].values
                years = np.array(common_years)
                mask = ~np.isnan(ratio) & ~np.isinf(ratio)
                if mask.sum() == 0:
                    continue
                ax.plot(years[mask], ratio[mask], color=color, alpha=alpha, linewidth=0.5)
        
        # Plot IMPs if requested
        if mode in ['imps', 'both']:
            for imp_name, scenario_name in IMP_SCENARIOS.items():
                model_name, imp_color = IMP_DETAILS[imp_name]
                key = (model_name, scenario_name)
                
                if key not in den_lookup:
                    continue
                
                # Find the numerator for this IMP
                for idx_num in ts_num.index:
                    if idx_num[0] == model_name and idx_num[1] == scenario_name:
                        idx_den = den_lookup[key]
                        ratio = ts_num.loc[idx_num].values / ts_den.loc[idx_den].values
                        years = np.array(common_years)
                        mask = ~np.isnan(ratio) & ~np.isinf(ratio)
                        
                        if mask.sum() > 0:
                            ax.plot(years[mask], ratio[mask], label=imp_name, 
                                   color=imp_color, path_effects=IMP_PATH_EFFECT, 
                                   linewidth=1.7, zorder=10)
                        break
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Year')
    
    # Add shared legend outside to the right
    if mode in ['imps', 'both']:
        handles, labels = axs[2].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    mode_label = {'imps': 'IMPs only', 'categories': 'All scenarios by category', 'both': 'All scenarios + IMPs'}
    plt.suptitle(f'Kaya Decomposition Factors ({mode_label.get(mode, mode)})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig, axs


# ============================================================================
# KAYA DECOMPOSITION PLOT (5-panel figure like IPCC)
# ============================================================================

def plot_kaya_decomposition(df_pyam, mode='both', categories=None, alpha=0.3, figsize=(8.7, 8.7)):
    """
    Plot the full Kaya decomposition in a 5-panel figure (3 top, 2 bottom + legend).
    
    Layout:
        Row 1: CO2 Emissions | Population | GDP per Capita
        Row 2: Legend        | Energy/GDP | CO2/Energy
    
    Parameters:
    -----------
    df_pyam : pyam.IamDataFrame
        The pyam dataframe containing the data
    mode : str
        'imps' - Plot only IMPs with colored lines
        'categories' - Plot all scenarios colored by category  
        'both' - Plot category-colored scenarios with IMPs highlighted on top
    categories : list, optional
        List of categories to include (e.g., ['C1', 'C2', 'C3']). 
        If None, defaults to Paris-compliant categories ['C1', 'C2', 'C3'] for 'categories'/'both' mode.
    alpha : float
        Transparency of category lines (only used when mode is 'categories' or 'both')
    figsize : tuple
        Size of the figure (width, height)
    
    Returns:
    --------
    fig, axs : matplotlib figure and axes objects
    
    Examples:
    ---------
    # Plot only IMPs
    plot_kaya_decomposition(df_pyam, mode='imps')
    
    # Plot Paris-compliant scenarios with IMPs
    plot_kaya_decomposition(df_pyam, mode='both', categories=['C1', 'C2', 'C3'])
    
    # Plot all scenarios by category
    plot_kaya_decomposition(df_pyam, mode='categories', categories=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
    """
    # Default to Paris-compliant categories for background
    if categories is None and mode in ['categories', 'both']:
        cats_to_plot = ['C1', 'C2', 'C3']
    elif categories is None:
        cats_to_plot = ALL_CATEGORIES
    else:
        cats_to_plot = categories
    
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    
    meta = df_pyam.meta
    cat_col = _get_category_column(meta)
    
    # Define variables and ratios for each panel
    # Row 1: CO2, Population, GDP/capita
    # Row 2: Legend (blank), Energy/GDP, CO2/Energy
    panels_config = [
        # (type, variable_or_ratio, ylabel, title, position)
        ('var', 'Emissions|CO2|Energy and Industrial Processes', 'Mt CO2/yr', 'CO2 Emissions', (0, 0)),
        ('var', 'Population', 'Billion people', 'Population', (0, 1)),
        ('ratio', ('GDP|PPP', 'Population'), 'Billion US$2010/Billion people', 'GDP per Capita', (0, 2)),
        ('legend', None, None, None, (1, 0)),  # Legend panel
        ('ratio', ('Primary Energy', 'GDP|PPP'), 'EJ/Billion US$2010', 'Energy Intensity (E/GDP)', (1, 1)),
        ('ratio', ('Emissions|CO2|Energy and Industrial Processes', 'Primary Energy'), 'Mt CO2/EJ', 'Carbon Intensity (CO2/E)', (1, 2)),
    ]
    
    # Helper to plot a single variable
    def plot_variable(ax, var):
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        df_filtered = df_pyam.filter(variable=var)
        if len(df_filtered) == 0:
            return
            
        df_ts = df_filtered.timeseries()
        df_ts = df_ts.reindex(columns=sorted(df_ts.columns))
        
        # Plot categories if requested
        if mode in ['categories', 'both']:
            for idx in df_ts.index:
                model, scenario = idx[0], idx[1]
                try:
                    category = meta.loc[(model, scenario), cat_col]
                    if pd.isna(category) or category not in cats_to_plot:
                        continue
                    color = CATEGORY_COLORS[category]
                except:
                    continue
                
                years = np.array(df_ts.columns)
                values = df_ts.loc[idx].values
                mask = ~np.isnan(values)
                if mask.sum() > 0:
                    ax.plot(years[mask], values[mask], color=color, alpha=alpha, linewidth=0.5)
        
        # Plot IMPs if requested
        if mode in ['imps', 'both']:
            for imp_name, scenario_name in IMP_SCENARIOS.items():
                model_name, imp_color = IMP_DETAILS[imp_name]
                for idx in df_ts.index:
                    if idx[0] == model_name and idx[1] == scenario_name:
                        years = np.array(df_ts.columns)
                        values = df_ts.loc[idx].values
                        mask = ~np.isnan(values)
                        if mask.sum() > 0:
                            ax.plot(years[mask], values[mask], label=imp_name,
                                   color=imp_color, path_effects=IMP_PATH_EFFECT,
                                   linewidth=1.7, zorder=10)
                        break
    
    # Helper to plot a ratio
    def plot_ratio(ax, var_num, var_den):
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        df_num = df_pyam.filter(variable=var_num)
        df_den = df_pyam.filter(variable=var_den)
        
        if len(df_num) == 0 or len(df_den) == 0:
            return
        
        ts_num = df_num.timeseries()
        ts_den = df_den.timeseries()
        
        common_years = sorted(set(ts_num.columns) & set(ts_den.columns))
        ts_num = ts_num[common_years]
        ts_den = ts_den[common_years]
        
        # Create lookup dict for denominator
        den_lookup = {}
        for idx_den in ts_den.index:
            key = (idx_den[0], idx_den[1])
            den_lookup[key] = idx_den
        
        # Plot categories if requested
        if mode in ['categories', 'both']:
            for idx_num in ts_num.index:
                model, scenario = idx_num[0], idx_num[1]
                key = (model, scenario)
                
                if key not in den_lookup:
                    continue
                
                try:
                    category = meta.loc[(model, scenario), cat_col]
                    if pd.isna(category) or category not in cats_to_plot:
                        continue
                    color = CATEGORY_COLORS[category]
                except:
                    continue
                
                idx_den = den_lookup[key]
                ratio = ts_num.loc[idx_num].values / ts_den.loc[idx_den].values
                years = np.array(common_years)
                mask = ~np.isnan(ratio) & ~np.isinf(ratio)
                if mask.sum() > 0:
                    ax.plot(years[mask], ratio[mask], color=color, alpha=alpha, linewidth=0.5)
        
        # Plot IMPs if requested
        if mode in ['imps', 'both']:
            for imp_name, scenario_name in IMP_SCENARIOS.items():
                model_name, imp_color = IMP_DETAILS[imp_name]
                key = (model_name, scenario_name)
                
                if key not in den_lookup:
                    continue
                
                for idx_num in ts_num.index:
                    if idx_num[0] == model_name and idx_num[1] == scenario_name:
                        idx_den = den_lookup[key]
                        ratio = ts_num.loc[idx_num].values / ts_den.loc[idx_den].values
                        years = np.array(common_years)
                        mask = ~np.isnan(ratio) & ~np.isinf(ratio)
                        if mask.sum() > 0:
                            ax.plot(years[mask], ratio[mask], label=imp_name,
                                   color=imp_color, path_effects=IMP_PATH_EFFECT,
                                   linewidth=1.7, zorder=10)
                        break
    
    # Plot each panel
    for panel_type, var_info, ylabel, title, pos in panels_config:
        ax = axs[pos[0], pos[1]]
        
        if panel_type == 'legend':
            # Create legend panel
            ax.axis('off')
            handles = []
            
            # Add IMPs to legend
            if mode in ['imps', 'both']:
                for imp_name in IMP_SCENARIOS.keys():
                    _, imp_color = IMP_DETAILS[imp_name]
                    handles.append(plt.Line2D([], [], color=imp_color, label=imp_name, 
                                             linewidth=2, path_effects=IMP_PATH_EFFECT))
            
            # Add category legend if showing categories
            if mode in ['categories', 'both']:
                handles.append(plt.Line2D([], [], color='grey', label='Paris-compliant', 
                                         linewidth=0.7, alpha=0.5))
            
            ax.legend(handles=handles, loc='center', frameon=False, ncol=1, fontsize=10)
            continue
        
        elif panel_type == 'var':
            plot_variable(ax, var_info)
        
        elif panel_type == 'ratio':
            var_num, var_den = var_info
            plot_ratio(ax, var_num, var_den)
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel('Year', fontsize=9)
        ax.grid(alpha=0.3)
    
    # Title
    mode_labels = {
        'imps': 'Illustrative Mitigation Pathways',
        'categories': f'All Scenarios ({", ".join(cats_to_plot)})',
        'both': f'IMPs + Background ({", ".join(cats_to_plot)})'
    }
    plt.suptitle(f'Kaya Decomposition: {mode_labels.get(mode, mode)}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig, axs


# ============================================================================
# INTERACTIVE KAYA EXPLORATION WIDGET (Single tab: Explore + Compute)
# ============================================================================

def create_kaya_explorer(df_pyam):
    """
    Create an interactive widget to explore Kaya equation for a single pathway.
    Students select a pathway, see the values, and compute custom ratios.
    
    Parameters:
    -----------
    df_pyam : pyam.IamDataFrame
        The pyam dataframe containing the data (must include Kaya variables)
    
    Returns:
    --------
    Interactive widget display
    """
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    
    # Get metadata
    meta = df_pyam.meta
    cat_col = _get_category_column(meta)
    
    # Kaya variable definitions with friendly names
    kaya_vars = {
        'CO2 Emissions': 'Emissions|CO2|Energy and Industrial Processes',
        'GDP (PPP)': 'GDP|PPP',
        'Population': 'Population',
        'Primary Energy': 'Primary Energy',
    }
    
    # IMP descriptions
    imp_descriptions = {
        'CurPol': '🔴 Politiques Actuelles - Statu quo',
        'ModAct': '🟠 Action Modérée - Basé sur les CDN actuelles',
        'Neg': '🟢 Émissions Négatives - Usage important du retrait de carbone',
        'Ren': '🔵 Fort Renouvelable - Déploiement rapide des renouvelables',
        'LD': '🩵 Faible Demande - Consommation énergétique réduite',
        'GS': '🟣 Renforcement Graduel - Politique progressive',
        'SP': '🟤 Trajectoires Durables - Développement durable',
    }
    
    # Output areas
    values_output = widgets.Output()
    ratio_output = widgets.Output()
    
    # ===== Controls =====
    imp_dropdown = widgets.Dropdown(
        options=list(IMP_SCENARIOS.keys()),
        value='LD',
        description='Trajectoire :',
        style={'description_width': '90px'}
    )
    
    year_slider = widgets.IntSlider(
        value=2050, min=2020, max=2100, step=10,
        description='Année :',
        style={'description_width': '90px'}
    )
    
    numerator_dropdown = widgets.Dropdown(
        options=list(kaya_vars.keys()),
        value='GDP (PPP)',
        description='Numérateur :',
        style={'description_width': '100px'}
    )
    
    denominator_dropdown = widgets.Dropdown(
        options=list(kaya_vars.keys()),
        value='Population',
        description='Dénominateur :',
        style={'description_width': '100px'}
    )
    
    # ===== Update Functions =====
    def update_values(imp_name, year):
        """Update the Kaya values display"""
        with values_output:
            clear_output(wait=True)
            
            scenario_name = IMP_SCENARIOS[imp_name]
            model_name, imp_color = IMP_DETAILS[imp_name]
            
            print(f"{imp_descriptions[imp_name]}")
            print(f"   Modèle : {model_name}")
            print(f"\nValeurs en {year} :")
            print("=" * 55)
            
            values = {}
            missing_vars = []
            for friendly_name, var_name in kaya_vars.items():
                df_filtered = df_pyam.filter(
                    model=model_name, 
                    scenario=scenario_name, 
                    variable=var_name
                )
                if len(df_filtered) > 0:
                    ts = df_filtered.timeseries()
                    if year in ts.columns:
                        val = ts[year].values[0]
                        values[friendly_name] = val
                        
                        # Format nicely
                        if 'CO2' in friendly_name:
                            print(f"   {friendly_name}: {val:,.0f} Mt CO2/an")
                        elif 'GDP' in friendly_name:
                            print(f"   {friendly_name}: {val:,.0f} milliards US$2010")
                        elif 'Population' in friendly_name:
                            print(f"   {friendly_name}: {val:,.0f} millions d'habitants")
                        elif 'Energy' in friendly_name:
                            print(f"   {friendly_name}: {val:,.1f} EJ/an")
                    else:
                        missing_vars.append(friendly_name)
                        print(f"   {friendly_name}: Pas de données pour {year}")
                else:
                    missing_vars.append(friendly_name)
                    print(f"   {friendly_name}: Variable non disponible pour ce scénario")
            
            # Show computed Kaya ratios
            if len(values) == 4:
                print("\n" + "=" * 55)
                print("RATIOS DE KAYA (calculés à partir des valeurs ci-dessus) :")
                
                pop = values['Population']
                gdp = values['GDP (PPP)']
                energy = values['Primary Energy']
                co2 = values['CO2 Emissions']
                
                gdp_per_cap = gdp / pop * 1000  # Convert to $/person
                energy_intensity = energy / gdp
                carbon_intensity = co2 / energy
                
                print(f"   PIB par habitant :      {gdp_per_cap:,.0f} US$/personne")
                print(f"   Intensité énergétique : {energy_intensity:.4f} EJ/milliard US$")
                print(f"   Intensité carbone :     {carbon_intensity:.2f} Mt CO2/EJ")
            elif missing_vars:
                print("\n" + "=" * 55)
                print("⚠️ Ratios de Kaya non calculables (variables manquantes)")
    
    def update_ratio(imp_name, numerator, denominator):
        """Update the ratio plot"""
        with ratio_output:
            clear_output(wait=True)
            
            if numerator == denominator:
                print("⚠️ Veuillez sélectionner des variables différentes pour le numérateur et le dénominateur !")
                return
            
            scenario_name = IMP_SCENARIOS[imp_name]
            model_name, imp_color = IMP_DETAILS[imp_name]
            
            var_num = kaya_vars[numerator]
            var_den = kaya_vars[denominator]
            
            df_num = df_pyam.filter(model=model_name, scenario=scenario_name, variable=var_num)
            df_den = df_pyam.filter(model=model_name, scenario=scenario_name, variable=var_den)
            
            if len(df_num) > 0 and len(df_den) > 0:
                ts_num = df_num.timeseries()
                ts_den = df_den.timeseries()
                
                common_years = sorted(set(ts_num.columns) & set(ts_den.columns))
                
                # Create plot
                fig, ax = plt.subplots(figsize=(6.7, 2.3))
                
                ratio = ts_num[common_years].values[0] / ts_den[common_years].values[0]
                years = np.array(common_years)
                mask = ~np.isnan(ratio) & ~np.isinf(ratio)
                
                ax.plot(years[mask], ratio[mask], color=imp_color, linewidth=2.5,
                       path_effects=IMP_PATH_EFFECT)
                ax.axhline(y=0, color='black', linewidth=0.5)
                ax.set_xlabel('Année')
                ax.set_ylabel(f'{numerator} / {denominator}')
                ax.set_title(f'{numerator} ÷ {denominator}', fontsize=12, fontweight='bold')
                ax.grid(alpha=0.3)
                plt.tight_layout()
                plt.show()
                
                # Show values at key years
                print(f"{numerator} ÷ {denominator} pour {imp_name} :")
                for y in [2020, 2050, 2100]:
                    if y in common_years:
                        idx = list(common_years).index(y)
                        print(f"   Année {y} : {ratio[idx]:.4f}")
            else:
                # Show which variable(s) are missing
                missing = []
                if len(df_num) == 0:
                    missing.append(numerator)
                if len(df_den) == 0:
                    missing.append(denominator)
                print(f"⚠️ Impossible de calculer le ratio pour {imp_name}")
                print(f"   Variable(s) non disponible(s) : {', '.join(missing)}")
    
    # ===== Connect widgets =====
    widgets.interactive_output(update_values, {'imp_name': imp_dropdown, 'year': year_slider})
    widgets.interactive_output(update_ratio, {'imp_name': imp_dropdown, 'numerator': numerator_dropdown, 'denominator': denominator_dropdown})
    
    # Trigger initial updates
    update_values(imp_dropdown.value, year_slider.value)
    update_ratio(imp_dropdown.value, numerator_dropdown.value, denominator_dropdown.value)
    
    # Watch for changes
    def on_imp_change(change):
        update_values(imp_dropdown.value, year_slider.value)
        update_ratio(imp_dropdown.value, numerator_dropdown.value, denominator_dropdown.value)
    
    def on_year_change(change):
        update_values(imp_dropdown.value, year_slider.value)
    
    def on_ratio_change(change):
        update_ratio(imp_dropdown.value, numerator_dropdown.value, denominator_dropdown.value)
    
    imp_dropdown.observe(on_imp_change, names='value')
    year_slider.observe(on_year_change, names='value')
    numerator_dropdown.observe(on_ratio_change, names='value')
    denominator_dropdown.observe(on_ratio_change, names='value')
    
    # ===== Build Layout =====
    header = widgets.HTML("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; border-radius: 10px; margin-bottom: 10px;">
            <h2 style="color: white; margin: 0;">Explorateur Interactif de l'Équation de Kaya</h2>
            <p style="color: #e0e0e0; margin: 5px 0 0 0;">
                CO₂ = Population × (PIB/Pop) × (Énergie/PIB) × (CO₂/Énergie)
            </p>
        </div>
    """)
    
    # Section 1: Explore values
    section1_header = widgets.HTML("""
        <div style="background: #f0f4f8; padding: 10px; border-radius: 8px; margin: 10px 0;">
            <h3 style="margin: 0; color: #2d3748;">Étape 1 : Explorer une trajectoire</h3>
            <p style="margin: 5px 0 0 0; color: #718096; font-size: 13px;">
                Sélectionnez une trajectoire et une année pour voir ses valeurs de variables de Kaya
            </p>
        </div>
    """)
    
    # Section 2: Compute ratio
    section2_header = widgets.HTML("""
        <div style="background: #f0f4f8; padding: 10px; border-radius: 8px; margin: 10px 0;">
            <h3 style="margin: 0; color: #2d3748;">Étape 2 : Calculer un ratio</h3>
            <p style="margin: 5px 0 0 0; color: #718096; font-size: 13px;">
                Construisez votre propre ratio de Kaya en sélectionnant numérateur et dénominateur :
                <br><b>PIB ÷ Pop</b> | <b>Énergie ÷ PIB</b> = intensité énergétique | <b>CO2 ÷ Énergie</b> = intensité carbone
            </p>
        </div>
    """)
    
    # Tip for Step 3
    step3_tip = widgets.HTML("""
        <div style="background: #e6fffa; padding: 12px; border-radius: 8px; margin-top: 15px; border-left: 4px solid #38b2ac;">
            <p style="margin: 0; color: #234e52;">
                <b>Étape 3 :</b> Pour comparer TOUTES les trajectoires, utilisez la fonction dans la cellule suivante :
                <br><code style="background: #fff; padding: 2px 6px; border-radius: 4px;">plot_kaya_decomposition(df_pyam, mode='both')</code>
            </p>
        </div>
    """)
    
    layout = widgets.VBox([
        header,
        section1_header,
        widgets.HBox([imp_dropdown, year_slider]),
        values_output,
        section2_header,
        widgets.HBox([numerator_dropdown, denominator_dropdown]),
        ratio_output,
        step3_tip,
    ])
    
    return layout


# ============================================================================
# DATABASE COMPOSITION PLOTS (Model Families & SSP)
# ============================================================================

def plot_database_composition(df_pyam, n_top_models=15, figsize=(10.7, 4.7)):
    """
    Plot database composition showing model families and SSP distribution.
    
    Creates two vertical stacked bar charts:
    1. Scenarios by Model Family (top N models)
    2. Scenarios by SSP Family
    
    Each bar is stacked by assessment status (Climate assessed, No assessment, Failed vetting).
    
    Parameters:
    -----------
    df_pyam : pyam.IamDataFrame
        The pyam dataframe containing the data
    n_top_models : int
        Number of top model families to display (default: 15)
    figsize : tuple
        Size of the figure (width, height)
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    import re
    import seaborn as sns
    
    # Get metadata with model information
    meta = df_pyam.meta
    cat_col = _get_category_column(meta)
    
    meta_analysis = meta.copy()
    meta_analysis['Model'] = meta_analysis.index.get_level_values('model')
    
    # Model families to normalize (case-insensitive matching)
    MODEL_FAMILIES = ['MESSAGE', 'REMIND', 'AIM', 'GCAM', 'TIAM', 'WITCH', 'IMAGE', 'POLES', 'GEM-E3']
    
    def extract_model_family(model_name):
        """Extract the model family name by removing version numbers and normalizing known families."""
        # Remove version patterns like "1.0", "2.1-4.3", "5.3", etc.
        family = re.sub(r'\s*\d+(\.\d+)?(-\d+(\.\d+)?)?$', '', model_name)
        family = family.strip()
        
        # Normalize known model families (case-insensitive)
        model_upper = family.upper()
        for known_family in MODEL_FAMILIES:
            if known_family.replace('-', '') in model_upper.replace('-', ''):
                return known_family
        return family
    
    meta_analysis['Model_Family'] = meta_analysis['Model'].apply(extract_model_family)
    
    # Group small model families (< 50 scenarios) into "Other"
    family_counts = meta_analysis['Model_Family'].value_counts()
    small_families = family_counts[family_counts < 50].index
    meta_analysis['Model_Family'] = meta_analysis['Model_Family'].apply(
        lambda x: 'Other' if x in small_families else x
    )
    
    # Create assessment status column
    def get_assessment_status(category):
        """Classify scenarios by their climate assessment status."""
        if pd.isna(category):
            return 'No assessment'
        elif category in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']:
            return 'Climate assessed (C1-C8)'
        elif 'failed' in str(category).lower() or 'exclude' in str(category).lower():
            return 'Failed vetting'
        else:
            return 'No assessment'
    
    meta_analysis['Assessment_Status'] = meta_analysis[cat_col].apply(get_assessment_status)
    
    # Find SSP-related column
    ssp_col = 'Ssp_family' if 'Ssp_family' in meta_analysis.columns else None
    
    # Custom colors as specified
    status_colors = {
        'Climate assessed (C1-C8)': '#66C2A5',  # Teal green
        'No assessment': '#8DA0CB',              # Blue/purple
        'Failed vetting': '#EDB120',             # Yellow/orange
    }
    
    # Reorder columns for consistent stacking
    status_order = ['Climate assessed (C1-C8)', 'No assessment', 'Failed vetting']
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ===== PLOT 1: Model Families (Vertical Stacked Bar) =====
    ax1 = axes[0]
    
    # Count by model family and assessment status
    model_counts = meta_analysis.groupby(['Model_Family', 'Assessment_Status']).size().unstack(fill_value=0)
    
    # Sort by total count and take top N model families
    model_totals = model_counts.sum(axis=1).sort_values(ascending=False)
    top_models = model_totals.head(n_top_models).index.tolist()
    
    # Move "Other" to the end if present
    if 'Other' in top_models:
        top_models.remove('Other')
        top_models.append('Other')
    
    model_counts = model_counts.loc[top_models]
    
    # Reorder columns
    model_counts = model_counts[[s for s in status_order if s in model_counts.columns]]
    
    # Plot vertical stacked bar chart
    model_counts.plot(kind='bar', stacked=True, ax=ax1, 
                      color=[status_colors.get(s, '#666666') for s in model_counts.columns],
                      edgecolor='black', linewidth=0.5)
    
    ax1.set_ylabel('Number of Scenarios', fontsize=11)
    ax1.set_xlabel('')
    ax1.set_title(f'Scenarios by Model Family\n(Top {n_top_models} models)', fontsize=13, fontweight='bold')
    ax1.get_legend().remove()  # Remove individual legend, will use shared legend
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    
    # ===== PLOT 2: SSP Families (Vertical Stacked Bar) =====
    ax2 = axes[1]
    
    if ssp_col:
        # Clean SSP values - the column contains numeric values 1-5
        def classify_ssp(val):
            if pd.isna(val):
                return 'Other/Unknown'
            try:
                val_num = int(float(val))
                if 1 <= val_num <= 5:
                    return f'SSP{val_num}'
            except:
                pass
            return 'Other/Unknown'
        
        meta_analysis['SSP_Clean'] = meta_analysis[ssp_col].apply(classify_ssp)
        
        # Count by SSP and assessment status
        ssp_counts = meta_analysis.groupby(['SSP_Clean', 'Assessment_Status']).size().unstack(fill_value=0)
        
        # Order SSPs
        ssp_order = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5', 'Other/Unknown']
        ssp_counts = ssp_counts.reindex([s for s in ssp_order if s in ssp_counts.index])
        
        # Reorder columns
        ssp_counts = ssp_counts[[s for s in status_order if s in ssp_counts.columns]]
        
        # Plot
        ssp_counts.plot(kind='bar', stacked=True, ax=ax2,
                        color=[status_colors.get(s, '#666666') for s in ssp_counts.columns],
                        edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('SSP Family', fontsize=11)
        ax2.set_ylabel('Number of Scenarios', fontsize=11)
        ax2.set_title('Scenarios by SSP Family', fontsize=13, fontweight='bold')
        ax2.get_legend().remove()  # Remove individual legend
        ax2.tick_params(axis='x', rotation=0)
    else:
        ax2.text(0.5, 0.5, 'SSP family data not available', 
                 ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title('Scenarios by SSP Family', fontsize=13, fontweight='bold')
    
    # Add single shared legend outside to the right
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, title='Assessment Status', loc='center left', 
               bbox_to_anchor=(1, 0.5), fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes