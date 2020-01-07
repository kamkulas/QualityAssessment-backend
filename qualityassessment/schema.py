import graphene
from graphene_django.debug import DjangoDebug

from system.schema import Query as SystemQuery, Mutation as SystemMutation
from loop.schema import Query as LoopQuery, Mutation as DescriptionMutation


class Query(SystemQuery, LoopQuery, graphene.ObjectType):
    debug = graphene.Field(DjangoDebug, name='_debug')


class Mutation(SystemMutation, DescriptionMutation, graphene.ObjectType):
    debug = graphene.Field(DjangoDebug, name='_debug')


schema = graphene.Schema(query=Query, mutation=Mutation)
