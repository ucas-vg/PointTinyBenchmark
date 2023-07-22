TITLE_FONTSIZE = 10
FIGSIZE = (12, 8)


def set_plt(plt):
    fig = plt.gcf()
    plt.axis('off')  # 去掉坐标轴
    # savefig 去除白边
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    return fig
