import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
# disable the warning
import warnings
warnings.filterwarnings("ignore")
def get_points(path,dims_of_interest = np.arange(8)):
    data = np.load(path)
    predictions = data["pred_with_artifact"]
    mu = data["mu"]
    reconstruction = data['reconstruct']
    predictions = predictions[predictions!=-1]
    mu = mu[data["pred_with_artifact"]!=-1]
    out = {}
    for p in np.unique(predictions):
        mu_p = mu[predictions==p]
        example_mu = np.mean(mu_p,axis=0)
        out[p] = example_mu
    ranges = {}
    for dim in dims_of_interest:
        # left_bound = 75% and right bound = 25% of the data
        left_bound = np.quantile(mu[:,dim],0.99)
        right_bound = np.quantile(mu[:,dim],0.01)
        ranges[dim] = [left_bound,right_bound]

    mu = mu# data["mu"]
    df = pd.DataFrame(mu, columns = ["feature_"+str(i) for i in range(mu.shape[1])])
    df["vae-predict"] = predictions
    return out,ranges, df

def draw_latent(df, ax, fontsize=12):
    #feature = df[[col for col in df.columns if "feature" in col]].values
    feature, pred, dim = [], [], []
    for f in df.columns:
        if "feature" in f:
            feature.append(df[f].values)
            pred.append(df["vae-predict"].values)
            dim.append([f.split("_")[-1]] * len(df)) 
    feature = np.concatenate(feature)
    pred = np.concatenate(pred)
    dim = np.concatenate(dim)
    df = pd.DataFrame({"feature": feature, "pred": pred, "dim": dim})
    df.loc[df["pred"]==0, "pred"] = "non-mpHFO"
    df.loc[df["pred"]==1, "pred"] = "mpHFO"
    sns.boxplot(data=df, x="feature", y="dim", hue="pred", linewidth=0.5, palette="tab10", ax=ax, fliersize=0.5, width=0.2, hue_order=["non-mpHFO", "mpHFO"])
    # add legend
    handles, labels = ax.get_legend_handles_labels()
    #removev legend title
    ax.legend(handles, labels, loc='lower right', ncol=1, fontsize=fontsize, title="")
    ax.set_xlabel("Value", fontsize=fontsize, labelpad=0.5)
    ax.set_ylabel("Latent Dimension", fontsize=fontsize, labelpad=0.5)
    ax.tick_params(axis='both', which='major', pad=0.5, labelsize=fontsize)
    ax.set_xlim([-6.5,6.5])
    ax.set_ylim([-1,7.5])
def draw_tf(ax, im):
    im = im.reshape(64,64)
    # im[:60,:] = 0
    ax.imshow(im)
    #ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')  
    
def get_stats(tfs, dim):
    def get_ratio(tfs):
        tfs = np.array(tfs)
        fn = "./t_test.npy" 
        data = np.load(fn)
        mean = np.mean(data,axis=0)
        mask = mean > 0.5
        mask_mean = np.sum(tfs[:,:,mask],axis=-1)
        all_power_mean = np.sum(np.sum(tfs,axis=-1), -1)
        ratio = mask_mean/all_power_mean
        return ratio
    def get_max(tfs):
        left_index = 22
        right_index = 42
        all_in = np.squeeze(np.array(tfs))
        all_in = all_in[:,1:48,left_index:right_index]
        max_value = np.mean(all_in,axis=-1) # (100,11,50,-1)
        max_index = np.argmax(max_value,axis=-1)*((10-290)/64) + 290
        return max_index
    def get_ratio_slowwave(tfs):
        all_in = np.array(tfs) # (100,11,50,20)
        hfo_band = all_in[:,:,:,:]
        spike_band = all_in[:,:,60:63,:]
        hfo_mean = np.sum(hfo_band,axis=-2) # (100,11,1,64)
        hfo_mean = np.sum(hfo_mean,axis=-1) 
        spike_mean = np.sum(np.sum(spike_band,axis=-2), -1) # (100,11,1,1)
        ratio = spike_mean/hfo_mean
        return ratio
    if dim == 3:
        return get_ratio(tfs)
    elif dim == 5:
        return get_ratio_slowwave(tfs)
    elif dim == 1:
        return get_max(tfs)
    else:
        raise ValueError("Not implemented")
    
def draw_perturb(axlist1, axlist2, pathological, physio, dim):
    pathological = pathological
    physio = physio
    pathological = np.array(pathological)
    physio = np.array(physio)
    index = [0,4,6,9]
    index = [0,1,2,3,4,6,7,8,9][::-1]
    pathological = [pathological[i] for i in index]
    physio = [physio[i] for i in index]
    for i,ax in enumerate(axlist1):
        draw_tf(ax, pathological[i])
   
    for i,ax in enumerate(axlist2):
        draw_tf(ax, physio[i])
     
    return get_stats(pathological, dim).reshape(-1), get_stats(physio, dim).reshape(-1)

def annotate_precentile(subfigure, color):
    # annotate 1 and 99 precentile on the subfigure
    # annotate mean using color 
    subfigure.text(0.05, 0.5, "1% tile", fontsize=6, horizontalalignment='center', verticalalignment='center', color="black", weight="bold")
    subfigure.text(0.95, 0.5, "99% tile", fontsize=6, horizontalalignment='center', verticalalignment='center', color="black", weight="bold")
    subfigure.text(0.50, 0.5, "Mean", fontsize=6, horizontalalignment='center', verticalalignment='center', color=color, weight="bold")

def draw_all_perturb(subfigure, pathological, physiological, i, df):
    # color the subfigure
    #subfigure.set_facecolor("lightgrey")
    subsubfigs = subfigure.subfigures(5, 1, hspace=0, height_ratios=[1.2,0.35,0.7,0.35,0.7])
    ax = subsubfigs[0].subplots(1,1)

    annotate_precentile(subsubfigs[1], "tab:orange")
    annotate_precentile(subsubfigs[3], "tab:blue")
 
    sff_mp=subsubfigs[2]
    axlist1 = sff_mp.subplots(1, 9)
    sff_mp.set_facecolor("moccasin")
    sff_mp.set_alpha(0.5)
    
    sff_nmp = subsubfigs[4]
    axlist2 = sff_nmp.subplots(1, 9)
    # subfigure 2 facecolor is lightblue
    sff_nmp.set_facecolor("lightblue")
    sff_nmp.set_alpha(0.5)
    feature_pa, feature_phy = draw_perturb(axlist1, axlist2, pathological[i], physiological[i], i)
    df.loc[df["vae-predict"]==0, "vae-predict"] = "non-mpHFO"
    df.loc[df["vae-predict"]==1, "vae-predict"] = "mpHFO"
    sns.boxplot(data=df, x=f"feature_{i}", hue="vae-predict", linewidth=0.5, palette="tab10", ax=ax, fliersize=0.5, width=0.4, hue_order=["non-mpHFO","mpHFO"])
    ax.set_xticks([])
    # draw 1% and 99% percentile
    ax.axvline(np.quantile(df[f"feature_{i}"],0.01), color="red", linestyle="--", linewidth=1)
    ax.axvline(np.quantile(df[f"feature_{i}"],0.99), color="red", linestyle="--", linewidth=1)
    # annotate 1% and 99% percentile
    ax.text(np.quantile(df[f"feature_{i}"],0.01)*0.95, -0.4, "1% tile", fontsize=6, horizontalalignment='center', verticalalignment='center', color="black", weight="bold")
    ax.text(np.quantile(df[f"feature_{i}"],0.99)*1.05, -0.4, "99% tile", fontsize=6, horizontalalignment='center', verticalalignment='center', color="black", weight="bold")
    # draw mean of the mpHFO
    mean = np.mean(df.loc[df["vae-predict"]=="mpHFO", f"feature_{i}"])
    ax.axvline(mean, color="tab:orange", linestyle="--", linewidth=1)
    # annotate the mean of the mpHFO
    #ax.text(mean, -0.4, "Mean mpHFO", fontsize=6, horizontalalignment='center', verticalalignment='center', color="tab:orange", weight="bold")
   
    # draw mean of the non-mpHFO
    mean = np.mean(df.loc[df["vae-predict"]=="non-mpHFO", f"feature_{i}"])
    ax.axvline(mean, color="tab:blue", linestyle="--", linewidth=1)
   
    # reverse the order of y-axis
    # legend text fontsize
    ax.legend(fontsize=6, loc="upper right", markerscale=0.5, title="")
    ax.set_xlabel("")
    #ax.set_ylabel(" \n", fontsize=6, labelpad=1)
    #ax.set_xticks([])
    ax.set_yticks([])
    # REWERSE THE ORDER OF Y-AXIS
    ax.invert_yaxis()
    # add a text "Dimension i" on right top of the ax as legend
    #ax.text(6.5, 0.5, f"Dimension {i}", fontsize=6, horizontalalignment='center', verticalalignment='center')
    # ticks
    ax.tick_params(axis='y', which='major', pad=0.5, labelsize=6)
    #ax.set_xlim([-6.5,6.5])
    # the x-axis ticks and ticks label to inner of the ax
    ax.tick_params(axis='x', direction='in', pad=-8, labelsize=6, length=2)
    #ax.invert_xaxis()
    # annotate Dimension i on the top left of the ax
    text_dict = {
        1: "Peak Frequency Dim.",
        3: "Pathological Dim.",
        5: "Slow-wave Dim.",
    }
    # flip the legend order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper right', ncol=1, fontsize=6, title="", markerscale=0.5, handlelength=0.8, handletextpad=0.5)

    ax.text(0.01, 0.95, text_dict[i], fontsize=6,
        horizontalalignment='left', verticalalignment='top', 
        transform=ax.transAxes, weight="bold")
    

def linefit_plot(data, ax, left="left", top="top"):
    # Calculate the medians for each category
    medians = np.median(data, axis=0)

    # Get the positions of the medians
    positions = range(len(medians))
    # Fit a simple linear regression to these median values
    slope, intercept = np.polyfit(positions, medians, 1)

    # Now, we'll overlay the line plot onto the boxplot using the regression coefficients
    # Generate a sequence of x values spanning the range of positions
    x_values = np.array(positions)

    # Calculate the corresponding y values from the slope and intercept
    y_values = slope * x_values + intercept

    # Plot the line on the same axis as the boxplot
    ax.plot(x_values, y_values, label='Trend', color='red', linewidth=1, linestyle='--', alpha=0.8)
    # add p-value
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(x_values, medians)
    latex_r_2 = r"$r^2$"
    if p_value < 0.001:
        p_value = "< 0.001"
        annotation_text = (
        #f"y = {slope:.2f}x + {intercept:.2f}\n"
        #f"R-value: {r_value:.2f}\n"
        f"p {p_value}\n"  # using scientific notation for p-value
        #f"Std Err: {std_err:.2f}\n"
        f"{latex_r_2} = {r_value**2:.2f}"
        )
    # Add a text annotation to the plot
    else:
        annotation_text = (
        #f"y = {slope:.2f}x + {intercept:.2f}\n"
        #f"R-value: {r_value:.2f}\n"
        f"p = {p_value:.2e}\n"  # using scientific notation for p-value
        #f"Std Err: {std_err:.2f}\n"
        # latex format
        f"{latex_r_2} = {r_value**2:.2f}"
        )
    if left == "left":
        ax.annotate(annotation_text, xy=(0.05, 0.93), xycoords='axes fraction', fontsize=4,
                horizontalalignment=left, verticalalignment=top,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='gray', alpha=0.9))
    else:
        ax.annotate(annotation_text, xy=(0.75, 0.93), xycoords='axes fraction', fontsize=4,
                horizontalalignment="left", verticalalignment=top,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='gray', alpha=0.9))
def draw_box_0(subfig, fontsize=6):
    # two vertical subfigures
    axs = subfig.subplots(2, 1)
    ax = axs[0]
    ax1 = axs[1]
    data = f"./res/{date}/{suffix}/perturbed3/perturbed3.npz"
    loaded = np.load(data)
    index = [0,1,2,3,4,6,7,8,9][::-1]
    ratio = loaded["ratio"]
    ratio = ratio[:,index]
    sns.boxplot(data=ratio, ax=ax, linewidth=0.5, palette="Set3", fliersize=0.5, width=0.5)
    linefit_plot(ratio, ax)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("% of power \n within Template", fontsize=fontsize, labelpad=0.5)
    # ticks
    ax.tick_params(axis='both', which='major', pad=0.5, labelsize=fontsize)
    ax.yaxis.set_tick_params(pad=0.5)
    ax.xaxis.set_tick_params(pad=0)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylim([0.2,0.94])

    # draw the boxplot of the predict
    predict = loaded["predict"][:,index]
    sns.violinplot(data=predict, ax=ax1, linewidth=0.5, palette="Set3", cut=0)
    mean = np.mean(predict, axis=0)
    plt.scatter(range(len(mean)), mean, color='white', edgecolor='red', s=10, label='Mean', zorder=3)
    ax1.set_xlabel("")
    ax1.set_ylabel("Probability", fontsize=fontsize, labelpad=0.5)
    # ticks
    ax1.tick_params(axis='both', which='major', pad=0.5, labelsize=fontsize)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.set_ylim([-0.05,1.05])
    ax1.set_xticks([])
    # ax1.yaxis.set_tick_params(pad=2)
    # ax1.xaxis.set_tick_params(pad=0)
def draw_box_1(subfig, fontsize=6):
    axs = subfig.subplots(2, 1)
    ax = axs[0]
    ax1 = axs[1]
    data = f"./res/{date}/{suffix}/perturbed1/perturbed1.npz"
    loaded = np.load(data)
    index = [0,1,2,3,4,6,7,8,9][::-1]
    max_index = loaded["max_index"]
    max_index = max_index[:,index]
    max_hz = max_index *((10-290)/64) + 290
    sns.boxplot(data=max_hz, ax=ax, linewidth=0.5,  palette="Set3", fliersize=0.5, width=0.5)
    linefit_plot(max_hz, ax, left="center", top="top")
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.set_ylabel("Peak Frequency (Hz) \n", fontsize=fontsize, labelpad=0.5)
    # ticks
    ax.tick_params(axis='both', which='major', pad=0.5, labelsize=fontsize)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # draw the violin plot of the predict
    predict = loaded["predict"][:,index]
    sns.violinplot(data=predict, ax=ax1, linewidth=0.5, palette="Set3", cut=0)
    # draw the mean 
    mean = np.mean(predict, axis=0)
    plt.scatter(range(len(mean)), mean, color='white', edgecolor='red', s=10, label='Mean', zorder=3)

    ax1.set_xlabel("")
    ax1.set_ylabel("Probability", fontsize=fontsize, labelpad=0.5)
    # ticks
    ax1.tick_params(axis='both', which='major', pad=0.5, labelsize=fontsize)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.set_ylim([-0.05,1.05])
    ax1.set_xticks([])

def draw_box_2(subfig, fontsize=6):
    axs = subfig.subplots(2, 1)
    ax = axs[0]
    ax1 = axs[1]
    data = f"./res/{date}/{suffix}/perturbed2/perturbed2.npz"
    loaded = np.load(data)
    index = [0,1,2,3,4,6,7,8,9][::-1]
    ratio = loaded["ratio"]
    ratio = ratio[:,index]
    
    sns.boxplot(data=ratio, ax=ax, linewidth=0.5, palette="Set3"
                , fliersize=0.5, width=0.5)
    linefit_plot(ratio, ax, left="left", top="top")
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.set_ylabel("% of power \n in 10-20 Hz ", fontsize=fontsize, labelpad=0.5)
    # ticks
    ax.tick_params(axis='both', which='major', pad=0.5, labelsize=fontsize)
    # add a space between the ticks label and the ticks
    ax.yaxis.set_tick_params(pad=0.5)
    # move the ticks label to the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylim([0.05,0.6])
    ax.xaxis.set_tick_params(pad=-2)

    # draw the violin plot of the predict
    predict = loaded["predict"][:,index]
    sns.violinplot(data=predict, ax=ax1, linewidth=0.5, palette="Set3", cut=0, inner="box")
    ax1.set_ylim([-0.05,1.05])
    # draw the mean
    mean = np.mean(predict, axis=0)
    # mean is a white color with grey edge
    plt.scatter(range(len(mean)), mean, color='white', edgecolor='red', s=10, label='Mean', zorder=3)
    ax1.set_xlabel("")
    ax1.set_ylabel("Probability", fontsize=fontsize, labelpad=0.5)
    # ticks
    ax1.tick_params(axis='both', which='major', pad=0.5, labelsize=fontsize)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.set_tick_params(pad=0)
    ax1.set_xticks([])
    
def draw_precentile(subfig, fontsize=6):
    ax = subfig.subplots(1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0,1])
    ax.set_ylim([0,2])
    # left
    ax.arrow(0.6, 1.3, 0.3, 0, head_width=0.4, head_length=0.05, fc='k', ec='k')
    ax.text(0.8, 0.25, "Higher %tile up to 99%tile", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
    # right
    ax.arrow(0.4, 1.3, -0.3, 0, head_width=0.4, head_length=0.05, fc='k', ec='k')
    ax.text(0.2, 0.25, "Lower %tile up to 1%tile", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
    # add text "Middle" in the middle
    ax.text(0.50, 0.8, "Mean", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
    ax.axis("off")
    
# create figure with 2 panels
def draw_fig(fold, save_fn=None):
    top, bot = [], []
    out, ranges, df = get_points(f"./res/{date}/fold_{fold}/{suffix}/train_overall_.npz")
    ratio = 0.8
    fig = plt.figure(figsize=(7*ratio,6.5*ratio),constrained_layout=True)
    fontsize = 6
    subfigs = fig.subfigures(4, 1, wspace=0, height_ratios=[1, 1, 1, 0.15], hspace=0.02)
    loaded = np.load(f"./res/{date}/{suffix}/perturbed/fold{fold}.npz")
    pathological = loaded["pathological"]
    physiological = loaded["physiological"]
    left_subsubfigs = []
    right_subsubfigs = []
    for i in range(4):
        s = subfigs[i].subfigures(1, 2, wspace=0.05, width_ratios=[1.2, 1])
        left_subsubfigs.append(s[1])
        right_subsubfigs.append(s[0])
    if fold == 0:
        selected = [0,2,3]
    elif fold == 1:
        selected = [3,1,5]
    elif fold == 3:
        selected = [1,2,7]
    for (i, s ) in enumerate(selected):
        draw_all_perturb(right_subsubfigs[i], pathological, physiological, s, df)
    ax_left1 = left_subsubfigs[0]#.subplots(1, 1)
    ax_left2 = left_subsubfigs[1]#.subplots(1, 1)
    ax_left3 = left_subsubfigs[2]#.subplots(1, 1)
    draw_box_0(ax_left1, fontsize=fontsize)
    draw_box_1(ax_left2, fontsize=fontsize)
    draw_box_2(ax_left3, fontsize=fontsize)
   
    text = [["a", "b"], ["e", "f"], ["i", "j"]]
    for s, t in zip(right_subsubfigs, text):
        s.text(-0.04, 0.95, t[0], fontsize=6+2, weight="bold")
        s.text(-0.04, 0.55, t[1], fontsize=6+2, weight="bold")

    text = [["c", "d"], ["g", "h"], ["k", "l"]]
    for s, t in zip(left_subsubfigs, text):
        s.text(-0.04, 0.95, t[0], fontsize=6+2, weight="bold")
        s.text(-0.04, 0.45, t[1], fontsize=6+2, weight="bold")
    # add a precentile subfigure
    
    draw_precentile(right_subsubfigs[-1], fontsize=fontsize)

    ax = left_subsubfigs[-1].subplots(1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0,1.2])
    ax.set_ylim([0,2])
    # left
    ax.arrow(0.6, 1.3, 0.3, 0, head_width=0.4, head_length=0.05, fc='k', ec='k')
    ax.text(0.8, 0.25, "Higher %tile", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
    # right
    ax.arrow(0.4, 1.3, -0.3, 0, head_width=0.4, head_length=0.05, fc='k', ec='k')
    ax.text(0.2, 0.25, "Lower %tile", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
    # add text "Middle" in the middle
    #ax.text(0.5, 0.8, "Mean", fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
    ax.axis("off")
    plt.savefig(f"./fig/{date}_figure5.jpg", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    date = "2023-12-28_1"
    suffix = "10000_2000_81"
    for i in [1]:
        draw_fig(i)
