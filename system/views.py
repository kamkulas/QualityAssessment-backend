import os
import json
from django.views import View
from django.http import HttpResponse, HttpResponseBadRequest

from system.models import System
from loop.models import Loop
from system.utils import Reader


class FileUpload(View):
    def post(self, request):
        name = request.POST.get('name')
        description = request.POST.get('description')
        file = request.FILES['file']
        if not (name and description and file):
            response_body = {'ok': False, 'message': 'All fields are required.'}
            return HttpResponseBadRequest(json.dumps(response_body), content_type='application/json')
        system = System.objects.create(name=name, description=description)
        directory_path = f'data_files/{system.id}'
        os.makedirs(directory_path)
        file_path = f'{directory_path}/{file.name}'
        with open(file_path, 'wb+') as destination:
            for chunk in file:
                destination.write(chunk)
        system.filename = file_path
        system.save()

        reader = Reader(system)
        result = reader.read()
        response = {}

        if not result['ok']:
            response['ok'] = False
            response['message'] = result['message']
            return HttpResponse(json.dumps(response), content_type='text/plain')

        response['ok'] = True
        response['id'] = system.id
        response['message'] = 'System saved!'

        for loop in result['loops']:
            Loop.objects.create(
                system=system,
                name=loop['name'],
                mv=loop['mv'],
                ma=loop['ma'],
                spa=loop['spa'],
                cv=loop['cv']
            )

        os.remove(file_path)
        os.rmdir(directory_path)

        return HttpResponse(json.dumps(response), content_type='text/plain')
