from django.http import HttpResponse
from django.template.loader import render_to_string
from weasyprint import HTML


def get_loop_card(request, id):
    html = HTML(string=render_to_string('loop_card.html'), base_url=request.build_absolute_uri())
    pdf = html.write_pdf()
    print(id) # tmp solution
    response = HttpResponse(pdf, content_type='application/pdf')
    response['Content-Disposition'] = 'inline; filename=loop_card.pdf'
    return response
