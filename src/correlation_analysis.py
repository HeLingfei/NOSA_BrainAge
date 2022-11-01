import math

import numpy as np
import pandas as pd
import re
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as cm
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


def read_sub_data():
    hcp_IS_path = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\结果\聚类结果\K-means' \
                  r'\HCP_Allsub_BNA_score.xlsx '
    return pd.read_excel(hcp_IS_path, header=0, index_col=1)  # type: pd.DataFrame


def get_regions_meta():
    regions_meta_path = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\结果\聚类结果\K-means' \
                        r'\brain_region_id.csv'

    regions_meta = pd.read_csv(regions_meta_path, header=0).iloc[:, 0: 3]  # type: pd.DataFrame
    region = None

    def map_row(row):
        nonlocal region
        if not pd.isnull(row[0]):
            region = re.sub(r'\(.*\)', '', row[0])
        row[0] = region
        return row

    regions_meta = regions_meta.iloc[:, :3].apply(map_row, axis=1, raw=True)
    return regions_meta, pd.MultiIndex.from_frame(regions_meta)


def save_regional_data():
    sub_data = read_sub_data()
    writer = pd.ExcelWriter('../analytical_data/region_data.xlsx')
    region_meta, index = get_regions_meta()
    region_meta.to_excel(writer, sheet_name='meta')
    arr = []

    # def map_column(column):
    #     find = region_meta[region_meta['id'] == column.name]
    #     if not find.empty:
    #         region_name = find['region'].values[0]
    #
    #         column.name = region_name
    #         new_df = pd.concat([sub_data['age'], column], axis=1)
    #         for i, df in enumerate(arr):
    #             if region_name in df.columns.tolist():
    #                 arr[i] = pd.concat([df, new_df])
    #                 break
    #         else:
    #             column.name = region_name
    #             arr.append(new_df)
    #
    # sub_data.apply(map_column)
    # print(index)
    # exit()
    # # arr = regiongroupby(by='')
    #

    region_counts = region_meta.groupby(['region'], sort=False).count().iloc[:, 0].tolist()
    arr = []
    for i, count in enumerate(region_counts):
        start = sum(region_counts[0:i]) + 1
        end = start + count
        arr.append(sub_data.iloc[:, start:end].mean(axis=1))
    arr = pd.concat(arr, axis=1)
    arr.columns = region_meta.groupby(['region'], sort=False).count().index.tolist()

    arr2 = []

    # 计算斯皮尔曼相关系数和对应p值
    def map_pair(df):
        new_df = pd.concat([sub_data['age'], df], axis=1)
        r, p_value = stats.spearmanr(new_df)
        new_df2 = pd.DataFrame([[r, p_value]], index=[new_df.columns.tolist()[1]], columns=['r', 'p_value'])
        arr2.append(new_df2)

    arr.apply(map_pair)
    arr2 = pd.concat(arr2, axis=0)
    arr2.to_excel(writer, sheet_name='region2age')

    arr.to_excel(writer, sheet_name='region_mean')

    arr.corr(method='spearman').to_excel(writer, sheet_name='region2region')

    corr = sub_data.corr(method='spearman')
    corr2age = corr.iloc[0, 1:].copy()  # type: pd.Series
    corr2age = pd.concat([pd.Series(corr2age.index, name='subregion_id'), pd.Series(corr2age.values, name='r')], axis=1)
    age2each_r = pd.merge(left=corr2age, right=region_meta, how='left', left_on='subregion_id', right_on='id').iloc[:,
                 :-1]

    age2each_r.to_excel(writer, sheet_name='subregion2age')

    each2each = sub_data.corr(method='spearman').iloc[1:, 1:].copy()
    each2each.to_excel(writer, sheet_name='subregion2subregion')

    writer.save()


def read_region_data(path='./analytical_data/region_data.xlsx'):
    reader = pd.ExcelFile(path)
    return {name: reader.parse(sheet_name=name, index_col=0) for name in reader.sheet_names}


def plot_region2age(region2age, subregion2age):
    # 外圈
    labels1 = subregion2age['subregion'].values.tolist()
    values1 = np.ones((1, len(subregion2age))).flatten()
    colors1 = get_heat_colors(color_num=len(values1), series=subregion2age['r'])
    # 内圈
    region_data = region2age['r']
    labels2 = subregion2age['region'].unique()
    region_counts = subregion2age.groupby(['region'], sort=False).count()
    values2 = region_counts['subregion'].tolist()
    colors2 = get_heat_colors(color_num=len(values1), series=region_data)
    # 绘图
    fig = plt.figure(figsize=(14, 14), dpi=400)  # type: plt.Figure
    ax = fig.subplots()
    # # 绘制外层饼图
    patches1, texts1 = ax.pie(x=values1,
                              labels=labels1,
                              explode=[0.02] * len(labels1),
                              colors=colors1.tolist(),
                              startangle=180,
                              labeldistance=1.06,
                              textprops={'va': 'center', 'ha': 'center', 'fontsize': 8},
                              rotatelabels=True,
                              wedgeprops={'width': 0.2}
                              )
    # 内层饼图
    patches2, texts2 = ax.pie(x=np.array(values2).flatten(),
                              labels=labels2,  # 标签
                              explode=[0.02] * len(labels2),
                              colors=colors2,
                              labeldistance=0.60,
                              startangle=180,
                              radius=0.79,  # 内圈距外圈距离
                              textprops={'va': 'center', 'ha': 'center', 'fontsize': 12, 'fontweight': 'bold'},
                              rotatelabels=True,
                              )

    # 按大区绘制边缘颜色，但是不好看所以废弃
    # edge_colors = sns.color_palette('Set2', n_colors=len(values2))
    # edge_colors = list(map(cm.colors.rgb2hex, edge_colors))
    # for index, count in enumerate(values2):
    #     color = edge_colors[index]
    #     base = sum(values2[0:index])
    #     patch = patches2[index]
    #     sub_start_patch = patches1[base]
    #     sub_end_patch = patches1[base + count-1]
    #
    #     prop = {
    #         'linewidth': 1,
    #         'edgecolor': color
    #     }
    #     patch.set(**prop)
    #     sub_start_patch.set(**prop)
    #     sub_end_patch.set(**prop)

    return


def get_heat_colors(color_num, min=-0.5, max=0.5, plot=False, series=None):
    # plt.rcParams.update({'figure.dpi': 150})
    cmap = plt.get_cmap('coolwarm', color_num)
    # The number of rgb quantization levels.如果想要分级的colorbar，这里的数值就和级数相同即可
    # 如果想要连续的这里就调成255肉眼就看不出来了
    colors = []
    for i in range(cmap.N):
        rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb
        colors.append(cm.colors.rgb2hex(rgb))
    if plot:
        plt.rcParams.update({'figure.dpi': 500})
        fig, ax = plt.subplots(figsize=(6, 1))
        norm = cm.colors.Normalize(vmin=min, vmax=max)
        scalar = cm.cm.ScalarMappable(norm=norm, cmap=cmap)  # type: cm.cm.ScalarMappable
        color_bar = fig.colorbar(scalar, orientation='horizontal', cax=ax)
        color_bar.ax.set_aspect(0.05)

    if series is not None:
        min_v = series.min()
        max_v = series.max()

        def calc_color(item):
            proportion = (item - min_v) / (max_v - min_v)
            pos = math.floor(proportion * (color_num - 1))
            return colors[pos]

        colors = series.map(calc_color)
    return colors


def plot_region2region(data, min_v=-0.5, max_v=0.5):
    plt.rcParams.update({'figure.dpi': 500})
    sns.set(font_scale=0.5)
    ax1 = sns.heatmap(data, vmin=min_v, vmax=max_v, center=0, cmap=plt.get_cmap('coolwarm', 246))


def get_regions_data_to_target(data, target_name, remove_non_sig=False, sig_level=0.01):
    target_data = data[target_name]
    arr = []
    for region_name in data.columns.tolist():
        if target_name != region_name:
            df = data[region_name]
            df = pd.concat([target_data, df], axis=1)
            r, p_value = stats.spearmanr(df)
            if not remove_non_sig or p_value < sig_level:
                new_df = pd.DataFrame([[r, p_value]], index=[df.columns.tolist()[1]], columns=['r', 'p_value'])
                arr.append(new_df)
    corr_data = pd.concat(arr, axis=0).sort_values(by='r', ascending=False).reset_index().rename(
        columns={'index': 'region'})
    corr_data['target_region'] = target_name
    return corr_data


def plot_region2regions(corr_data):

    ax = sns.barplot(data=corr_data, x='region', y='r')  # type: plt.Axes
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-90)
    # print(corr_data)
    for i, p in enumerate(ax.patches):
        p_value = corr_data['p_value'].iloc[i]
        asterisk_str = ''
        if p_value <= 0.05:
            asterisk_str += '*'
        if p_value <= 0.03:
            asterisk_str += '\n*'
        if p_value <= 0.01:
            asterisk_str += '\n*'
        ax.annotate(asterisk_str, xy=(p.get_x(), p.get_y() + p.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points', ha="left", va="center")
    ax.set_xlabel('')


def plot_regions2regions(data):
    arr = []
    for target_name in data.columns.tolist():
        arr.append(get_regions_data_to_target(data, target_name))
    all_data = pd.concat(arr, axis=0)
    print(all_data)
    all_data['r'] = all_data['r'].abs()
    print(all_data.groupby('target_region').mean())
    # grid = sns.FacetGrid(analytical_data=all_data, x='region', y='r', hue='')



def plot():
    data = read_region_data()
    # print(analytical_data)

    sns.set_theme(context='paper')
    # plot_region2age(analytical_data['region2age'], analytical_data['subregion2age'])
    # get_heat_colors(color_num=246, plot=True)
    # plot_region2region(analytical_data['subregion2subregion'])
    # plot_region2region(analytical_data['region2region'], min_v=-1, max_v=1)
    corr_data = get_regions_data_to_target(data['region_mean'], 'Basal Ganglia')
    plot_region2regions(corr_data)
    # plot_regions2regions(analytical_data['region_mean'])
    plt.tight_layout()
    plt.show()


# save_regional_data()
plot()
