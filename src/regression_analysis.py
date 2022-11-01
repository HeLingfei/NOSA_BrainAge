import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend import Legend
from correlation_analysis import read_sub_data, read_region_data
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.patheffects as pe
from scipy import stats


# region_data = read_region_data()
# region_IS = region_data['region_mean']
# region2age = region_data['region2age']  # type: pd.DataFrame

# sub_IS = read_sub_data()
# sub2age = region_data['subregion2age']


def sort_by_abs(data, keys, ascending=False):
    indexes = data.abs().sort_values(by=keys, ascending=ascending).index
    return data.loc[indexes]


# print(sort_by_abs(region2age, keys='r'))
def get_max_abs_row(data, num, keys):
    return sort_by_abs(data, keys=keys).head(num)


def get_grid_data(region2age, y_cate='kind'):
    grid_data = []
    for i in range(1, len(region2age.columns)):
        sliced_data = region2age.iloc[:, [0, i]]
        y_name = sliced_data.columns.tolist()[-1]
        sliced_data = sliced_data.rename(columns={y_name: 'y'})
        sliced_data[y_cate] = y_name
        grid_data.append(sliced_data)
    return pd.concat(grid_data, axis=0)


# 默认第一列为x,with_regression为True时，根据orders创建多项式回归
def plot_grid(data, y_cate='kind', orders=None, correlations=None, set_args=None, x_label='', y_label='',
              pallete='Set2'):
    pallete = sns.color_palette('Set2', len(data.columns) - 1)
    grid_data = get_grid_data(data, y_cate=y_cate)
    grid = sns.FacetGrid(grid_data, col=y_cate, col_wrap=6, palette=pallete)
    if set_args is not None:
        grid.set(**set_args)
    grid.set_xlabels(x_label)
    grid.set_ylabels(y_label)

    y_cate_names = grid_data[y_cate].unique().tolist()

    reg_models = []

    # 针对每个子图的操作
    def map_func(data, color):
        current_index = y_cate_names.index(data[y_cate].unique().tolist()[0])
        color = pallete[current_index]
        sns.scatterplot(data=data, x=data.columns.tolist()[0], y='y', color=color)
        if orders is not None:
            order = orders[current_index]
            column_names = data.columns.tolist()
            reg_data = get_regression_data(data=data, order=order)
            reg_line_data = pd.DataFrame(reg_data['reg_x'])
            reg_line_data.insert(1, column_names[1], reg_data['pred_y'])
            reg_column_names = reg_line_data.columns.tolist()
            sns.lineplot(data=reg_line_data, x=reg_column_names[0], y=reg_column_names[1], color=color, linewidth=2,
                         path_effects=[pe.Stroke(linewidth=2.5, foreground='grey'), pe.Normal()])
            reg_models.append(reg_data['fitted_model'])

    grid.map_dataframe(map_func)

    for i, ax in enumerate(grid.axes):  # type: int, plt.Axes
        ax.set_title(y_cate_names[i], fontsize='large')

        if correlations is not None or orders is not None:
            labels = []
            p_values = []

            # 相关性标注
            if correlations is not None:
                r, p_value = correlations.iloc[i, :]
                labels.append(r'  $\it{r}$ = %.2f' % r)
                p_values.append(p_value)

            # 拟合优度标注
            if orders is not None:
                model = reg_models[i]
                labels.append(r'$\it{R2}$ = %.2f' % model.rsquared)
                p_values.append(model.f_pvalue)

            print(p_values)
            # 显著性标注
            if len(p_values) > 0:
                pre_str = '      '
                if all(p_value <= 0.01 for p_value in p_values):
                    labels.append(pre_str + '***')
                elif all(p_value <= 0.05 for p_value in p_values):
                    labels.append(pre_str + '**')
                elif all(p_value <= 0.1 for p_value in p_values):
                    labels.append(pre_str + '*')

            legend = ax.legend(loc="upper right", handlelength=0, handletextpad=0, shadow=False, borderpad=0.2,
                               labels=[],
                               markerscale=0, title='\n'.join(labels))  # type: Legend
            legend.get_title().set_position((0, -20))


def get_poly_formula(order, x_name='x', y_name='y'):
    formula = f'{y_name} ~ '
    sub_str = ''
    for i in range(order):
        item = f'I({x_name} ** {order - i})'
        if order - i > 1:
            item += ' + '
        sub_str += item
    formula += sub_str
    return formula


def get_regression_data(data, order):
    column_names = data.columns.tolist()
    x = data[column_names[0]]
    y = data[column_names[1]]
    formula = get_poly_formula(order, x_name=column_names[0], y_name=column_names[1])
    model = smf.ols(formula, data=data).fit()
    regression_x = pd.Series(np.linspace(x.min(), x.max(), len(x)), name=x.name)
    pred_y = model.predict(regression_x)
    pred_y.name = y.name

    predictions = model.get_prediction(regression_x).summary_frame(alpha=0.01)
    return {
        'x': x,
        'y': y,
        'reg_x': regression_x,
        'pred_y': pred_y,
        'fitted_model': model,
        'pred_y_upper': predictions['mean_ci_upper'].reset_index(drop=True),
        'pred_y_lower': predictions['mean_ci_lower'].reset_index(drop=True)
    }


def plot_init():
    plt.rcParams.update({'figure.dpi': 350})
    sns.set_theme(context='paper', palette='Set2')


def load_init_data(arg={'num': None}, kind='region2age'):
    region_data = read_region_data()
    if kind == 'region2age':
        region_IS = region_data['region_mean']
        region2age = region_data['region2age']
        sub_IS = read_sub_data()
        sorted_region2age = sort_by_abs(region2age, keys='r')
        index = sorted_region2age.index
        num = len(region_IS.columns) if arg['num'] is None else arg['num']
        region_IS = pd.concat([sub_IS['age'], region_IS[index].iloc[:, 0:num]], axis=1)
        region_orders = [
                            3, 3, 3, 3, 3, 5,
                            2, 3, 3, 2, 3, 3,
                            5, 3, 5, 3, 3, 5,
                            5, 5, 5, 5, 3, 3
                        ][0:num]
        return region_IS, region_orders, sorted_region2age

    elif kind == 'region2region':
        region2region = region_data['region2region']
        num = arg['num'] if arg['num'] is not None else len(region2region)
        region2age = get_max_abs_row(region_data['region2age'], num=num, keys='r')
        max_region_names = region2age.index.tolist()
        max_region_corr = region2region[max_region_names]

        def row_func(col):
            col[col.name] = np.nan
            max_abs_v = col.abs().max()
            max_name = col[col == max_abs_v].index.tolist()[0]
            return pd.Series((max_name, col[max_name]), index=['max_region', 'max_correlation'])

        max_region_info = max_region_corr.apply(row_func)
        print(max_region_corr)
        print(max_region_info)
        # print(max_region_corr.idxmax(axis=1).values)
        return

    elif kind == 'region2regions':
        print(region_data['region2region'])


def plot_finish():
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_region2age_reg(num=None):
    plot_init()
    region_IS, region_orders, sorted_region2age = load_init_data(arg={'num': num})
    plot_grid(region_IS, y_cate='region', orders=region_orders, correlations=sorted_region2age,
              set_args={'xticks': [8, 20, 30, 40, 50, 60, 70, 80]}, x_label='Chronological Age', y_label='IS')

    plot_finish()


def plot_multiline_region2age():
    plot_init()
    sns.set(font_scale=1.2)
    region_IS, region_orders, sorted_region2age = load_init_data(arg={'num': 6})
    grid_data = get_grid_data(region_IS, y_cate='region')
    region_names = grid_data['region'].unique().tolist()
    palette = sns.color_palette('Set2', len(region_names))

    line_data = []
    ax = plt.gca()
    for index, region_name in enumerate(region_names):
        init_data = grid_data[grid_data['region'] == region_name]
        reg_data = get_regression_data(init_data, region_orders[index])
        one_line_data = pd.concat([reg_data['reg_x'], reg_data['pred_y'], init_data['region']], axis=1)
        ax.fill_between(x=reg_data['reg_x'], y1=reg_data['pred_y_lower'], y2=reg_data['pred_y_upper'],
                        facecolor=palette[index],
                        alpha=0.3)
        line_data.append(one_line_data)
    line_data = pd.concat(line_data).reset_index(drop=True)
    print(line_data)
    line_data_names = line_data.columns.tolist()
    print(line_data_names)
    ax = sns.lineplot(ax=ax, data=line_data, x=line_data_names[0], y=line_data_names[1], hue=line_data_names[2],
                      palette=palette, legend='brief',
                      linewidth=2,
                      path_effects=[pe.Stroke(linewidth=2.5, foreground='grey'), pe.Normal()])  # type: plt.Axes
    legend = ax.get_legend()  # type: Legend
    legend.set_title('')
    ax.figure.set_size_inches(10, 6)
    ax.set_xlabel('Chronological Age')
    ax.set_ylabel('IS')
    ax.set_xticks([8, 20, 30, 40, 50, 60, 70, 80])
    plot_finish()



plot_region2age_reg()
# plot_region2age_reg(num=6)
# plot_multiline_region2age()
# region2region = load_init_data({'num': 6}, kind='region2region')  # type: pd.DataFrame
# print(region2region)
# matrix = np.array(region2region)
# print(np.triu(matrix, k=1).argmax())
