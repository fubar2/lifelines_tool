# script for a lifelines ToolFactory KM/CPH tool for Galaxy
# km models for https://github.com/galaxyproject/tools-iuc/issues/5393
# test as
# python plotlykm.py --input_tab rossi.tab --htmlout "testfoo" --time "week" --status "arrest" --title "test" --image_dir images --cphcol="prio,age,race,paro,mar,fin"
# Ross Lazarus July 2023
import argparse

import os
import sys

import lifelines

from matplotlib import pyplot as plt

import pandas as pd


def trimlegend(v):
    """
    for int64 quintiles - must be ints - otherwise get silly legends with long float values
    """
    for i, av in enumerate(v):
        x = int(av)
        v[i] = str(x)
    return v

kmf = lifelines.KaplanMeierFitter()
cph = lifelines.CoxPHFitter()

parser = argparse.ArgumentParser()
a = parser.add_argument
a('--input_tab', default='rossi.tab', required=True)
a('--header', default='')
a('--htmlout', default="test_run.html")
a('--group', default='')
a('--time', default='', required=True)
a('--status',default='', required=True)
a('--cphcols',default='')
a('--title', default='Default plot title')
a('--image_type', default='png')
a('--image_dir', default='images')
a('--readme', default='run_log.txt')
args = parser.parse_args()
sys.stdout = open(args.readme, 'w')
df = pd.read_csv(args.input_tab, sep='\t')
NCOLS = df.columns.size
NROWS = len(df.index)
QVALS = [.2, .4, .6, .8] # for partial cox ph plots
defaultcols = ['col%d' % (x+1) for x in range(NCOLS)]
testcols = df.columns
if len(args.header.strip()) > 0:
    newcols = args.header.split(',')
    if len(newcols) == NCOLS:
        if (args.time in newcols) and (args.status in newcols):
            df.columns = newcols
        else:
            sys.stderr.write('## CRITICAL USAGE ERROR (not a bug!): time %s and/or status %s not found in supplied header parameter %s' % (args.time, args.status, args.header))
            sys.exit(4)
    else:
        sys.stderr.write('## CRITICAL USAGE ERROR (not a bug!): Supplied header %s has %d comma delimited header names - does not match the input tabular file %d columns' % (args.header, len(newcols), NCOLS))
        sys.exit(5)
else: # no header supplied - check for a real one that matches the x and y axis column names
    colsok = (args.time in testcols) and (args.status in testcols) # if they match, probably ok...should use more code and logic..
    if colsok:
        df.columns = testcols # use actual header
    else:
        colsok = (args.time in defaultcols) and (args.status in defaultcols)
        if colsok:
            print('Replacing first row of data derived header %s with %s' % (testcols, defaultcols))
            df.columns = defaultcols
        else:
            sys.stderr.write('## CRITICAL USAGE ERROR (not a bug!): time %s and status %s do not match anything in the file header, supplied header or automatic default column names %s' % (args.time, args.status, defaultcols))
print('## Lifelines tool\nInput data header =', df.columns, 'time column =', args.time, 'status column =', args.status)
os.makedirs(args.image_dir, exist_ok=True)
fig, ax = plt.subplots()
if args.group > '':
    names = []
    times = []
    events = []
    for name, grouped_df in df.groupby(args.group):
        T = grouped_df[args.time]
        E = grouped_df[args.status]
        gfit = kmf.fit(T, E, label=name)
        kmf.plot_survival_function(ax=ax)
        names.append(str(name))
        times.append(T)
        events.append(E)
    ax.set_title(args.title)
    fig.savefig(os.path.join(args.image_dir,'KM_%s.png' % args.title))
    ngroup = len(names)
    if  ngroup == 2: # run logrank test if 2 groups
        results = lifelines.statistics.logrank_test(times[0], times[1], events[0], events[1], alpha=.99)
        print('Logrank test for %s - %s vs %s\n' % (args.group, names[0], names[1]))
        results.print_summary()
else:
    kmf.fit(df[args.time], df[args.status])
    kmf.plot_survival_function(ax=ax)
    ax.set_title(args.title)
    fig.savefig(os.path.join(args.image_dir,'KM_%s.png' % args.title))
    print('#### No grouping variable, so no log rank or other Kaplan-Meier statistical output is available')
survdf = lifelines.utils.survival_table_from_events(df[args.time], df[args.status])
lifedf = lifelines.utils.survival_table_from_events(df[args.time], df[args.status], collapse=True)
print("#### Survival table using time %s and event %s" % (args.time, args.status))
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(survdf)
print("#### Life table using time %s and event %s" % (args.time, args.status))
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(lifedf)
titl = args.title.replace(' ','_')
outpath = os.path.join(args.image_dir,'%s_survival_table.tabular' % titl)
survdf.to_csv(outpath, sep='\t')
outpath = os.path.join(args.image_dir,'%s_life_table.tabular' % titl)
lifedf.to_csv(outpath, sep='\t')
if len(args.cphcols) > 0:
    fig, ax = plt.subplots()
    ax.set_title('Cox-PH model: %s' % args.title)
    cphcols = args.cphcols.strip().split(',')
    cphcols = [x.strip() for x in cphcols]
    notfound = sum([(x not in df.columns) for x in cphcols])
    if notfound > 0:
        sys.stderr.write('## CRITICAL USAGE ERROR (not a bug!): One or more requested Cox PH columns %s not found in supplied column header %s' % (args.cphcols, df.columns))
        sys.exit(6)
    colsdf = df[cphcols]
    print('### Lifelines test of Proportional Hazards results with %s as covariates on %s' % (', '.join(cphcols), args.title))
    cutcphcols = [args.time, args.status] + cphcols
    cphdf = df[cutcphcols]
    ucolcounts = colsdf.nunique(axis=0)
    cph.fit(cphdf, duration_col=args.time, event_col=args.status)
    cph.print_summary()
    for i, cov in enumerate(colsdf.columns):
         if ucolcounts[i] > 10: # a hack - assume categories are sparse - if not imaginary quintiles will have to do
             v = pd.Series.tolist(cphdf[cov].quantile(QVALS))
             vdt = df.dtypes[cov]
             if vdt == 'int64':
                 v = trimlegend(v)
             axp = cph.plot_partial_effects_on_outcome(cov, cmap='coolwarm', values=v)
             axp.set_title('Cox-PH %s quintile partials: %s' % (cov,args.title))
             figr = axp.get_figure()
             oname = os.path.join(args.image_dir,'%s_CoxPH_%s.%s' % (args.title, cov, args.image_type))
             figr.savefig(oname)
         else:
             v = pd.unique(cphdf[cov])
             v = [str(x) for x in v]
             try:
                 axp = cph.plot_partial_effects_on_outcome(cov, cmap='coolwarm', values=v)
                 axp.set_title('Cox-PH %s partials: %s' % (cov,args.title))
                 figr = axp.get_figure()
                 oname = os.path.join(args.image_dir,'%s_CoxPH_%s.%s' % (args.title, cov, args.image_type))
                 figr.savefig(oname)
             except:
                 pass
    cphaxes = cph.check_assumptions(cphdf, p_value_threshold=0.01, show_plots=True)
    for i, ax in enumerate(cphaxes):
        figr = ax[0].get_figure()
        titl = figr._suptitle.get_text().replace(' ','_').replace("'","")
        oname = os.path.join(args.image_dir,'CPH%s.%s' % (titl, args.image_type))
        figr.savefig(oname)
