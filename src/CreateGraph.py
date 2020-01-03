def create_graph(df_target, df_values):

    df_target = pd.get_dummies(df_target)
    df = pd.concat([df_target, df_values], axis = 1)

    df_corr = df.corr()
    df_corr.drop('functional', inplace = True)
    df_corr.drop('non functional', inplace = True)
    df_corr.drop('functional needs repair', inplace = True)

    df_func_10_pos = df_corr.sort_values(by=['functional'], ascending = False)[['functional']].head(10)
    df_func_10_neg = df_corr.sort_values(by=['functional'], ascending = False)[['functional']].tail(11)
    df_func_10_neg.drop('extraction_type_class_other', inplace = True)

    df_func_10 = pd.concat([df_func_10_pos, df_func_10_neg])


    ### creates graph
    labels = list(df_func_10.index)
    x_pos = np.arange(len(labels))
    values = list(df_func_10.functional)
    color_b, color_r = ['#64b3ff']*(len(labels)//2), ['#ffb364']*(len(labels)//2)
    color_b.extend(color_r)

    fig_size = (12,9)

    plt.figure(figsize = fig_size)
    plt.bar(x_pos, values, align='center', color = color_b, alpha = 1)
    plt.grid(zorder=0, alpha= .5, linestyle = '--')
    plt.axvline(x=9.5, color = 'black', linestyle = '--')
    plt.axhline(y=0, color = 'black')
    plt.xticks(x_pos, labels, rotation=80, )
    plt.xlabel('Features', fontsize = (fig_size[0])*3//2)
    plt.ylabel('Percent Correlation', fontsize = (fig_size[0])*3//2)
    plt.ylim(-.4, .4)
    plt.title('Correlation with Well Functionality', fontsize=(fig_size[0])*2, y = 1.03)

    plt.tight_layout()
    plt.savefig('reports/corr_well_func.png', transparent = True)
    return

