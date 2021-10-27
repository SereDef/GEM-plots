import argparse
import logging
import time
# data storage and handling
# from dask.distributed import Client
import dask.dataframe as dd # scale pandas func up to +1TB
import numpy as np
# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import seaborn as sns

mpl.rcParams['agg.path.chunksize'] = 10000000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(path: str):
    logger.info("Loading data from %s.", path)

    # client = Client(n_workers=8) # Setup

    res = dd.read_csv(
        path,  # "/Users/Serena/Desktop/GEM/METAANALYSIS_Result_GxEmodel_prenatal_2108121_sorted_moreinfo.txt",
        sep=" ",
        usecols=["SNP", "BRAIN", "P-value"],  # only needed columns
        dtype={"SNP": "object", "BRAIN": "category", "P-value": "float32"},
    )

    res['BRAIN'] = res.BRAIN.cat.as_known()
    res['brain'] = res.BRAIN.cat.codes

    # Split chromosome nr from position
    res['chrom'] = res["SNP"].str.partition(':')[0]
    res['pos'] = res["SNP"].str.partition(':')[2]  # len range from 3 to 9 digits

    # add 0s to the positions with less digits to preserve order
    add0 = res["pos"].apply(lambda x: len(x), meta=('pos', 'int64')).astype('uint8').apply(lambda x: '.' + ('0' * (9 - x)), meta=('pos', 'object'))

    # Create SNP column with floats (chrom.position) rather than strings (chrom:position)
    res['SNP'] = (res['chrom'] + add0 + res['pos'])
    res['snp'] = res['SNP'].astype('float64')

    res['chrom'] = res['chrom'].astype('uint8')
    res['pos'] = res['pos'].astype('uint32')

    # Log-transform p-values
    res['P-value'] = dd.to_numeric(abs(res['P-value'].apply(lambda x: np.log10(x), meta=('P-value', 'float64')))).astype('float16')

    # Dichotomize chromosome numbers (even = 1 and uneven = 0) for color
    res['chrb'] = res["chrom"].apply(lambda x: 1 if x % 2 == 0 else 0, meta=('chrom', 'int64')).astype('uint8')

    res = res.set_index('snp') # Sort the dataframe according to SNP values


    # select only significant points
    # cutoff = abs(np.log10(0.05))
    # res_sign = res.loc[res['P-value'] > cutoff, :]

    # res = client.persist(res)  # save sorted dataset in memory

    # ========== Simple Manhattan plot ==========
    # logger.info("Plotting Simple Manhattan plot")
    # define colors to differentiate chromosomes
    # cmp = mcolors.ListedColormap(["lightseagreen", "mediumpurple"])

    # simple_mnhttn_plt = plt.figure(figsize=(20, 5))
    # plt.scatter(res.index, res["P-value"], c=res.chrb, cmap=cmp)  # to check all variables and get a base for timing
    # plt.xticks(np.arange(1.5, 23.5, 1.0), [str(int(x)) for x in np.arange(1, 23, 1)])
    # plt.xlabel("Chromosome", fontsize=17, fontweight="bold")
    # plt.ylabel("-log10(p-value)", fontsize=17, fontweight="bold")
    # simple_mnhttn_plt.savefig("simple_mnhttn_plt.png")

    # # 3D PLOTS
    # logger.info("Plotting 3D Manhattan poly")
    # Create a list of tuple (SNP, P-value) for every brain region
    # verts = []
    # Get the distinct brain regions
    # brain_reg = res.BRAIN.drop_duplicates().tolist()
    # But for time sake:
    # brain_reg = ['bilat_accumbe_std', 'bilat_amygdal_std', 'bilat_caudate_std', 'bilat_hippoca_std',
    #              'bilat_pallidu_std', 'bilat_putamen_std', 'bilat_thalamu_std', 'eTIV_f9_std']

    # for z in brain_reg:
    #    data = res_sign[res_sign.BRAIN == z]
    #    xs = np.array(data.index)  # SNP
    #    ys = np.array(data['P-value'])
    #    xs = np.r_[0, ys, 23]
    #    print(ys)
    #    ys = np.r_[0, ys, 0]  # to ensure the poligon is drawn face up
    #    verts.append(list(zip(xs, ys)))

    # poly3d = plt.figure(figsize=(25, 20))
    # ax = poly3d.add_subplot(projection='3d')

    # def cc(arg):
    #    return mcolors.to_rgba(arg, alpha=0.4)

    # poly = PolyCollection(verts,
    #                      facecolors=[cc('y'), cc('r') , cc('b'), cc('g'), cc('y'), cc('r'), cc('b'), cc('g')])
    # poly.set_alpha(0.5)
    # ax.add_collection3d(poly, zs=[4, 5, 6, 7, 8, 9, 10, 11], zdir='y')

    # Adjust axes lengths
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 0.5, 1, 1]))

    # X axis settings
    # ax.set_xlim3d(0, 23)
    # ax.axes.set_yticks(np.arange(1.5, 23.5, 1.0))
    # ax.axes.set_xticklabels([str(int(x)) for x in np.arange(1, 23, 1)])
    # ax.set_xlabel('\nChromosome', fontsize=17, fontweight='bold', linespacing=3)
    # Y axis settings
    # ax.set_ylim3d(0, 8)
    # ax.axes.set_yticks(np.arange(0, 8, 1))
    # ax.axes.set_yticklabels(['Accumbens', 'Amygdala', 'Caudate', 'Hippocampus', 'Pallidu', 'Putamen', 'Thalamus', 'eTIV_f9'],
    #                         fontsize=15, fontweight='bold', ha='left', va='center')
    # Z axis settings
    # ax.set_zlim3d(2, 8)
    # ax.set_zlabel('-log10(P-value)', fontsize=17, fontweight='bold', linespacing=3)

    # poly3d.savefig("Polyregion_manhattan.png")

    #
    # td_plt = plt.figure(figsize=(25, 25))
    # ax = td_plt.add_subplot(projection='3d')
    #
    # # Set axis limits
    # ax.set_zlim([1, 7])
    # ax.set_xlim([1, 22])
    # ax.set_ylim([0, 7])
    #
    # # 3D Scatterplot
    # ax.scatter(res_sign.index, res_sign['brain'], res_sign['P-value'], alpha=0.5, c= res_sign['chrom'], cmap = "Set3")
    #
    # ax.view_init(15, 240) # turn the plot
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 0.5, 1, 1]))
    #
    # ax.axes.set_yticklabels(
    #     ['Accumbens', 'Amygdala', 'Caudate', 'Hippocampus', 'Pallidu', 'Putamen', 'Thalamus', 'eTIV_f9'])
    # ax.axes.set_xticks(np.arange(1.5, 23.5, 1.0))
    # ax.axes.set_xticklabels([str(int(x)) for x in np.arange(1, 23, 1)])

    # Set labels for the axes
    # ax.set_xlabel('\nChromosome', fontsize=17, fontweight='bold', linespacing=3)
    # ax.set_ylabel('\nBrain ROI', fontsize=17, fontweight='bold', linespacing=3.2)
    # ax.set_zlabel('-log10(p)', fontsize=17, fontweight='bold', rotation=180)
    #
    # td_plt.savefig("3d_plot.png")
    #
    # HEATMAP
    res1 = res.categorize(columns=['SNP'])
    divnorm = colors.TwoSlopeNorm(vmin=0, vcenter=5.5, vmax=7)

    piv = res1.pivot_table(index='BRAIN', columns='SNP', values='P-value')
    heat_mp = plt.figure(figsize = (20, 7))
    ax = sns.heatmap(piv, cbar_kws={'label': 'log10(p-value)'}, cmap = 'Reds', norm = divnorm)
    ax.set_ylabel("Brain ROI")
    # ax.xticks(np.arange(0, 7, 1.0), ['Accumbens','Amygdala','Caudate','Hippocampus','Pallidu','Putamen','Thalamus','eTIV_f9'])
    ax.set_xlabel("SNP")

    heat_mp.savefig("heatmap_plt.png")


if __name__ == "__main__":
    start_time = time.time()

    # Create the parser
    arg_parser = argparse.ArgumentParser()

    # Add the arguments
    arg_parser.add_argument(
        "-p", "--path", type=str, help="the path to the file", required=True
    )

    # Execute the parse_args() method
    args = arg_parser.parse_args()

    # Plot graphs
    main(args.path)
    logger.info("Execution time: %f min.", (time.time() - start_time) / 60)
