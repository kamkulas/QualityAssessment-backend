from django.http import HttpResponse
from django.template.loader import render_to_string
from weasyprint import HTML
from loop.models import Loop
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import io


def loop_card(request, id):
    try:
        loop = Loop.objects.get(pk=id)
    except Loop.DoesNotExist:
        return HttpResponse(status=404)
    html = HTML(string=render_to_string('loop_card.html', {'loop': loop}), base_url=request.build_absolute_uri())
    pdf = html.write_pdf()
    response = HttpResponse(pdf, content_type='application/pdf')
    response['Content-Disposition'] = 'inline; filename=loop_card.pdf'
    return response


def trend_plot(request, id):
    try:
        loop = Loop.objects.get(pk=id)
    except Loop.DoesNotExist:
        return HttpResponse(status=404)
    fig, ax_mv = plt.subplots(figsize=(6.5, 3.5))
    t = np.arange(0, len(loop.mv))

    plt.title('Trend')
    ax_mv.plot(t, loop.mv, label='Manipulated variable', color='#00cc00')
    ax_mv.grid(linestyle='--', color='0.75')
    ax_ma = ax_mv.twinx()
    ax_ma.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax_ma.plot(t, loop.ma, label='Mode', color='#0000ff')
    ax_mv.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
    ax_ma.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25))

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response


def control_error_plot(request, id):
    try:
        loop = Loop.objects.get(pk=id)
    except Loop.DoesNotExist:
        return HttpResponse(status=404)
    error = []
    length = len(loop.cv)
    for i in range(length):
        error.append(loop.cv[i] - loop.spa[i])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    t = np.arange(0, length)

    plt.title('Control error')
    ax.grid(linestyle='--', color='0.75')
    ax.plot(t, loop.cv, label='Controlled variable', color='#3333ff')
    ax.plot(t, loop.spa, label='Setpoint', color='#8c1aff')
    ax.fill_between(
        t,
        [float(item) for item in loop.spa],
        [float(item) for item in loop.cv],
        color='#8c1aff',
        alpha=0.5
    )
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response


def error_plot(request, id):
    try:
        loop = Loop.objects.get(pk=id)
    except Loop.DoesNotExist:
        return HttpResponse(status=404)
    error = []
    length = len(loop.cv)
    for i in range(length):
        error.append(loop.cv[i] - loop.spa[i])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    t = np.arange(0, length)

    ax.grid(linestyle='--', color='0.75')
    ax.plot(t, error, label='Control error', color='#26a69a')
    plt.title('Control error')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response


def rs_plot(request, id):
    try:
        loop = Loop.objects.get(pk=id)
    except Loop.DoesNotExist:
        return HttpResponse(status=404)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    plt.title('RS plot')
    ax.grid(linestyle='--', color='0.75', zorder=1)
    ax.plot(loop.xx, loop.yy, '-o', markersize=3,
            markerfacecolor='#1fa815', markeredgecolor='#1fa815', label='yy',
            color='#62ff5e', zorder=2)
    ax.plot(loop.xx, loop.haa, '-o', markersize=3,
            markerfacecolor='#2643e6', markeredgecolor='#2643e6', label='haa',
            color='#628eff', zorder=2)
    ax.plot(loop.xp, loop.yp, label='yp', color='#9b71ff', zorder=2)
    ax.scatter([loop.crossX1], [loop.crossY1], marker='s', color='#FFD600',
               zorder=3)
    ax.scatter([loop.crossX2], [loop.crossY2], marker='s', color='#FFD600',
               zorder=3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response


def characteristics_plot(request, id):
    try:
        loop = Loop.objects.get(pk=id)
    except Loop.DoesNotExist:
        return HttpResponse(status=404)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    plt.title('Characteristics')
    ax.grid(linestyle='--', color='0.75')
    ax.scatter(loop.cv, loop.mv, s=5, facecolors='none', edgecolors='#2196f3', alpha=0.6)
    ax.plot(loop.cv, loop.firstApproximation, linewidth=1, label='Linear approximation', color='#8e24aa')
    ax.plot(loop.cv, loop.secondApproximation, linewidth=1, label='Quadratic approximation', color='#d81b60')
    ax.set_ylabel('Manipulated variable')
    ax.set_xlabel('Controlled variable')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25))

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response


def histogram_plot(request, id):
    try:
        loop = Loop.objects.get(pk=id)
    except Loop.DoesNotExist:
        return HttpResponse(status=404)

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.grid(linestyle='--', color='0.75')
    x = [float(item) for item in loop.histX]
    y = [float(item) for item in loop.histY]
    y.append(float(0))
    width = float(np.max(loop.histX) - np.min(loop.histX))

    ax.bar(x, y, width=0.005*width, color='#FDD835')
    ax.plot(x, loop.gauss, label='Gauss', color='#F06292')
    ax.plot(x, loop.levy, label='Levy', color='#BA68C8')
    ax.plot(x, loop.laplace, label='Laplace', color='#4FC3F7')
    ax.plot(x, loop.huber, label='Huber', color='#4DB6AC')
    plt.title('Histogram')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response


def radar_plot(request, id):
    try:
        loop = Loop.objects.get(pk=id)
    except Loop.DoesNotExist:
        return HttpResponse(status=404)

    categories = ['H0', 'HDE', 'HRE', 'QE', 'IAE', 'ISE', '\u03C3 Huber',
                  '\u03B2 Laplace', '\u03B3', '\u03C3 Gauss', '\u03B1',
                  'H3', 'H2', 'H1']
    values = [loop.h0, np.abs(loop.minHde / loop.hde), loop.minHre / loop.hre,
              loop.minQe / loop.qe, loop.minIae / loop.iae,
              loop.minIse / loop.ise, loop.minRsig / loop.rsig,
              loop.minLb / loop.lb, loop.minSgam / loop.sgam,
              loop.minGsig / loop.gsig, loop.salf - 1, loop.h3, loop.h2,
              loop.h1, loop.h0]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    ax.set_rlabel_position(0)
    ax.spines['polar'].set_color('gray')
    ax.spines['polar'].set_alpha(0.4)
    ax.grid(alpha=0.4)
    plt.title('Radar')
    ticks = [0.2, 0.4, 0.6, 0.8]
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    plt.yticks(ticks, [str(item) for item in ticks], color='grey', size=7)
    plt.ylim(0, 1)
    ax.set_theta_zero_location('N')
    ax.plot(angles, values, color='#ff9800')
    ax.fill(angles, values, color='#ff9800', alpha=0.375)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response
