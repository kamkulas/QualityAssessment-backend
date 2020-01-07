import graphene
from graphene_django.types import DjangoObjectType

from system.models import System


class SystemType(DjangoObjectType):
    class Meta:
        model = System


class Query(graphene.ObjectType):
    system = graphene.Field(SystemType, id=graphene.Int())
    all_systems = graphene.List(SystemType)

    def resolve_all_systems(self, context):
        return System.objects.all()

    def resolve_system(self, context, id=None):
        if id:
            return System.objects.get(pk=id)
        return None


class SystemMutation(graphene.Mutation):
    class Arguments:
        name = graphene.String()
        description = graphene.String()

    ok = graphene.Boolean()
    system = graphene.Field(SystemType, id=graphene.Int(), name=graphene.String())

    def mutate(self, info, name, description):
        system = System.objects.create(name=name, description=description)
        return SystemMutation(ok=True, system=system)


class DeleteSystemMutation(graphene.Mutation):
    class Arguments:
        id = graphene.Int()

    ok = graphene.Boolean()
    id = graphene.Int()

    def mutate(self, info, id):
        try:
            system = System.objects.get(pk=id)
        except System.DoesNotExist:
            return SystemMutation(ok=False)
        system.delete()
        return SystemMutation(ok=True)


class Mutation(graphene.ObjectType):
    create_system = SystemMutation.Field()
    delete_system = DeleteSystemMutation.Field()
