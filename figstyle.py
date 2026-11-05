import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
METHOD_COLOR = {'Q-DAGFlow': '#2a78d6', 'HEFT-Q': '#eb6834', 'PEFT-Q': '#1baf7a', 'SHEFT-Q': '#eda100',
                'QG-Dyn': '#e87ba4', 'FG-Dyn': '#008300', 'CF-Dyn': '#6250d6'}
METHOD_MARK = {'Q-DAGFlow': 'o', 'HEFT-Q': 's', 'PEFT-Q': 'D', 'SHEFT-Q': '^', 'QG-Dyn': 'v', 'FG-Dyn': 'P', 'CF-Dyn': 'X'}
BACK_COLOR = {'Eagle-S': '#eb6834', 'Eagle-K': '#eda100', 'Heron-T': '#1baf7a', 'Heron-F': '#2a78d6'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
def setup():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.edgecolor': '#9a9993',
        'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 1.6, 'lines.markersize': 5, 'savefig.dpi': 300, 'figure.dpi': 110,
        'legend.frameon': False, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
def save(fig, name):
    fig.savefig('/home/claude/qdag/figs/%s.png' % name, bbox_inches='tight', facecolor='white')
    plt.close(fig)
