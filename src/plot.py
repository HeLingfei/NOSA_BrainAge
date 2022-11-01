import glob
import logging
import os
import time
import warnings
import matplotlib.ticker as ticker
import matplotlib
import glob
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.text import Text
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader
import uuid
import My_function as mf
import pandas as pd
from Make_datasets import Mydataset
from Network import My_Network
import statsmodels.formula.api as smf
import matplotlib.patheffects as pe
from statsmodels.stats.outliers_influence import summary_table

warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.FloatTensor)


def test_model_inference():
    # base_data_dir = '/HOME/scz0774/run/lfhe/data/SimpleBrainAge'
    base_data_dir = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\数据'

    datapath = f'{base_data_dir}/Train/*.nii'
    labelpath = f'{base_data_dir}/Train/Train.csv'

    # 将所有数据文件路径装入list中
    path = sorted(glob.glob(datapath))
    label = mf.Load_Label(labelpath)

    def get_data_loader_by_indexes(paths, labels, indexes, data_augment, b_size):
        paths = np.array(paths)
        labels = np.array(labels)
        dataset = Mydataset(paths[indexes], labels[indexes], with_augmentation=data_augment)
        return DataLoader(dataset=dataset, batch_size=b_size, shuffle=True)

    def get_model(model_class):
        m = model_class()
        # 多gpu
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            m.cuda()
            if torch.cuda.device_count() > 1:
                m = nn.DataParallel(m)
        return m

    model = get_model(My_Network)
    # print(label)
    loader = get_data_loader_by_indexes(path, labels=label, indexes=list(range(12)), data_augment=False, b_size=12)
    for data_t, label_t in loader:
        # print(data_t.shape)
        train_x = data_t
        train_x = train_x.cuda()
        pred_y = model(train_x)


def test_pyecharts():
    from pyecharts import options as opts
    from pyecharts.charts import Bar
    from pyecharts.faker import Faker

    c = (
        Bar()
        .add_xaxis(Faker.choose())
        .add_yaxis("商家A", Faker.values(), gap="0%")
        .add_yaxis("商家B", Faker.values(), gap="0%")
        .set_global_opts(title_opts=opts.TitleOpts(title="Bar-不同系列柱间距离"))
        .render("bar_different_series_gap.html")
    )


sns.set_theme(context='paper', font_scale=1.5)
pd.set_option('display.max_columns', None)
path = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\结果\聚类结果\Yeo_network.csv'
tb = pd.read_csv(path)


def create_network_info(tb):
    target_tb = tb.iloc[:, 10:12].copy()  # type: pd.DataFrame
    target_tb.columns = ['id', 'name']
    start_index = target_tb[target_tb['id'] == 'Yeo  7 Network'].index.tolist()[0]

    start_flag = False
    indexes = []
    for index, row in target_tb.iterrows():
        if start_flag and not row.isnull()['id'] and type(row['id'] == str):
            start_flag = True
            if row['id'].isdecimal():
                indexes.append(index)
        if row['id'] == 'Yeo  7 Network':
            start_flag = True
        elif row['id'] == 'Yeo  17 Network':
            start_flag = False
    result = target_tb.iloc[indexes].copy()
    return result.reset_index(drop=True)


# net_info = create_network_info(tb)
# subs = tb.dropna(subset=tb.columns.values[:5]).iloc[:, :4]
# print(net_info)
# sub_groups = subs.groupby('Yeo_7network')
# print(sub_groups.count())
dir_path = r'D:\documents\AcademicDocuments\MasterCandidate\research\文献\可解释脑龄预测工作汇总\结果\聚类结果\Yeo-7network'


def get_age_groups_from_net(path, grouped=True):
    df = pd.read_excel(path, sheet_name='Sheet2', header=None, names=['Chronological Age', 'IS'])
    bins = [0]
    bins.extend(range(20, 80, 10))
    bins.append(81)
    df['Age Group'] = pd.cut(df['Chronological Age'], bins=bins, right=False, include_lowest=True,
                             labels=list(range(1, 8, 1)))
    if grouped:
        df = df.groupby(['Age Group'])
    # print(list(df))
    return df


def get_mean_IS_from_groups(groups):
    return groups.mean()[['IS']]


def get_all_sub_mean_by_age(dir_path, net_info):
    net_paths = glob.glob(os.path.join(dir_path, '*.xlsx'))
    net_paths.sort()
    results = []
    for index, path in enumerate(net_paths):
        df = get_age_groups_from_net(path)
        df = get_mean_IS_from_groups(df)
        results.append((net_info.iloc[index]['id'], df))
    return results


def plot_line(data, colors, age_groups, abbrs):
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    ax = sns.lineplot(x='Age Group', y='IS', hue='Network ID', data=data, palette=colors)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.set_xticklabels([r'Group-' + str(i + 1) + '\n' + rf'{span} Years' for i, span in enumerate(age_groups)])
    for tick in ax.xaxis.get_major_ticks():
        label = tick.label1  # type: Text
        label.set_fontstyle('italic')
        # label.set_fontweight('bold')
        label.set_multialignment('center')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(title='', handles=handles, labels=abbrs, loc="upper left")
    return ax


def plot_bar(data, colors, age_groups, abbrs):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    ax = sns.barplot(x=data['IS'].tolist(), y=data['Age Group'].tolist(),
                     hue='Network ID', data=data, orient='h', palette=colors)  # type: plt.Axes
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.set_yticklabels([r'Group-' + str(i + 1) + '\n' + rf'{span} Years' for i, span in enumerate(age_groups)])
    for tick in ax.yaxis.get_major_ticks():
        label = tick.label1  # type: Text
        label.set_fontstyle('italic')
        label.set_fontweight('bold')
        label.set_multialignment('center')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(title='', handles=handles, labels=abbrs)
    return ax


def plot_regress(data, colors, age_groups, abbrs):
    print(data.head(10))


def plot_7networks_IS_by_age_groups(kind="regress"):
    age_groups = ['[8,20)', '[20,30)', '[30,40)',
                  '[40,50)', '[50,60)', '[60,70)', '[70,80]']
    abbrs = ['VN', 'LN', 'DAN', 'VAN', 'SMN', 'FPN', 'DMN']
    colors = pd.read_csv('../data/7NetworksColors.csv')['Hex'].tolist()
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    all_subs = get_all_sub_mean_by_age(dir_path, net_info)
    new_subs = pd.DataFrame([])
    for net_id, net_sub in all_subs:
        net_sub['Network ID'] = net_id
        new_subs = pd.concat([new_subs, net_sub])
    new_subs = new_subs.reset_index()

    if kind == 'bar':
        ax = plot_bar(new_subs, colors, age_groups, abbrs)
    elif kind == 'line':
        ax = plot_line(new_subs, colors, age_groups, abbrs)
    elif kind == 'regress':
        ax = plot_regress(new_subs, colors, age_groups, abbrs)
    sns.despine()
    plt.tight_layout(h_pad=2)
    plt.show()


# plot_7networks_IS_by_age_groups()
def plot_grid():
    pass


def get_data(net_info):
    net_paths = glob.glob(os.path.join(dir_path, '*.xlsx'))
    net_paths.sort()
    results = []
    for index, path in enumerate(net_paths):
        df = get_age_groups_from_net(path, grouped=False)
        df['Network ID'] = net_info.iloc[index]['id']
        results.append(df)
    results = pd.concat(results)
    results = results.reset_index(drop=True)
    return results


def save_network_data(path):
    writer = pd.ExcelWriter(path)
    net_info = create_network_info(tb)
    net_info.to_excel(excel_writer=writer, sheet_name='net_meta', index=False)
    data = get_data(net_info)  # type: pd.DataFrame
    data.rename(columns={'Chronological Age': 'chronological_age', 'Age Group': 'age_group',
                         'Network ID': 'network_id'}, inplace=True)
    data.to_excel(excel_writer=writer, sheet_name='net_data', index=False)
    writer.save()


# save_network_data('./7network_data.xlsx')


def load_network_data(path='./7network_data.xlsx'):
    reader = pd.ExcelFile(path)
    net_meta = reader.parse(sheet_name='net_meta')
    net_data = reader.parse(sheet_name='net_data')
    return net_meta, net_data


def get_regression_data(one_net_data, cat_meta):
    x = one_net_data['chronological_age']
    y = one_net_data['IS']
    orders = [1, 2, 2, 2, 2, 2, 2]
    current_net_row = cat_meta[cat_meta['id'] == one_net_data['network_id'].unique()[0]]
    id_index = current_net_row.index.tolist()[0]
    current_order = orders[id_index]
    formula = 'IS ~ I(chronological_age ** 2) + chronological_age' if current_order == 2 \
        else 'IS ~ chronological_age'
    model = smf.ols(formula, data=one_net_data).fit()
    regression_x = pd.Series(np.linspace(x.min(), x.max(), len(x)), name=x.name)
    pred_y = model.predict(regression_x)
    pred_y.name = 'IS'

    ages = pd.Series(range(8, 81))
    ages.name = 'chronological_age'
    predictions = model.get_prediction(ages).summary_frame(alpha=0.01)
    return {
        'x': x,
        'y': y,
        'reg_x': regression_x,
        'pred_y': pred_y,
        'fitted_model': model,
        'id_index': id_index,
        'pred_y_upper': predictions['mean_ci_upper'].reset_index(drop=True),
        'pred_y_lower': predictions['mean_ci_lower'].reset_index(drop=True)
    }


def plot_with_regression_line(cat_data, **kwargs):
    data = kwargs['data']
    reg_data = get_regression_data(data, cat_data)
    reg_line_data = pd.DataFrame(reg_data['reg_x'])
    reg_line_data.insert(1, 'IS', reg_data['pred_y'])
    color = sns.color_palette('Set2', n_colors=7)[reg_data['id_index']]
    sns.scatterplot(reg_data['x'], reg_data['y'], color=color)
    sns.lineplot(data=reg_line_data, x='chronological_age', y='IS', color=color, linewidth=2,
                 path_effects=[pe.Stroke(linewidth=2.5, foreground='grey'), pe.Normal()])


def plot_network_grid(net_meta, net_data):
    orders = [1, 2, 2, 2, 2, 2, 2]
    print(net_data)
    grid = sns.FacetGrid(net_data, col="network_id", col_wrap=3)
    grid.map_dataframe(plot_with_regression_line, cat_data=net_meta, orders=orders)
    grid.set(xticks=[8, 20, 30, 40, 50, 60, 70, 80])
    grid.set_xlabels('Chronological Age')
    for index, ax in enumerate(grid.axes):  # type: int, plt.Axes
        ax.set_title(net_meta.iloc[index]['abbr'], fontsize='large')

        one_net_data = net_data[net_data['network_id'] == net_meta.iloc[index]['id']]
        model = get_regression_data(one_net_data, net_meta)['fitted_model']
        ax.legend(loc="upper right", handlelength=0, handletextpad=0, shadow=False,
                  labels=[r'$\it{R2}$ = %.2f' % model.rsquared])
        if index == 0:
            fig = ax.get_figure()
            fig.add_subplot()
    grid.tight_layout()
    plt.show()


def create_dataframe_by_columns(columns):
    reg_line_data = pd.DataFrame(columns[0])
    for i in range(1, len(columns)):
        col = columns[i]
        reg_line_data.insert(i, col.name, col)
    return reg_line_data


def plot_networks(net_meta, net_data):
    net_ids = net_meta['id'].values.tolist()
    lines_data = []
    ax = plt.gca()
    palette = sns.color_palette('Set2', n_colors=7)
    for id in net_ids:
        one_net_data = net_data[net_data['network_id'] == id]
        reg_data = get_regression_data(one_net_data, net_meta)
        reg_line_data = create_dataframe_by_columns([reg_data['reg_x'], reg_data['pred_y']])
        reg_line_data['network_abbr'] = net_meta[net_meta['id'] == id]['abbr'].values[0]
        lines_data.append(reg_line_data)
        ax.fill_between(x=np.arange(8, 81, 1), y1=reg_data['pred_y_lower'], y2=reg_data['pred_y_upper'], facecolor=palette[net_ids.index(id)],
                        alpha=0.3)
    reg_lines_data = pd.concat(lines_data).reset_index(drop=True)
    ax = sns.lineplot(data=reg_lines_data, x='chronological_age', y='IS', hue='network_abbr',
                      palette=palette,
                      linewidth=2,
                      path_effects=[pe.Stroke(linewidth=2.5, foreground='grey'), pe.Normal()])  # type: plt.Axes

    ax.set_xticks([8, 20, 30, 40, 50, 60, 70, 80])
    ax.tick_params(labelsize='x-large')
    ax.figure.set_size_inches(16, 8)
    handlers, labels = ax.get_legend_handles_labels()
    ax.legend(loc='upper left', title=None, handles=handlers, labels=[label + ' (***)' for label in labels], fontsize="x-large")
    ax.set_xlabel('Chronological Age', fontsize="x-large")
    ax.set_ylabel(ax.get_ylabel(), fontsize="x-large")
    plt.tight_layout()
    plt.show()


net_meta, net_data = load_network_data()
# plot_network_grid(net_meta, net_data)
plot_networks(net_meta, net_data)
