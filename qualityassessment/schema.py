import graphene
from graphene_django.debug import DjangoDebug

from system.schema import Query as SystemQuery, Mutation as SystemMutation
from loop.schema import Query as LoopQuery


class Query(SystemQuery, LoopQuery, graphene.ObjectType):
    debug = graphene.Field(DjangoDebug, name='_debug')


schema = graphene.Schema(query=Query, mutation=SystemMutation)
