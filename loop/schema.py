import graphene
from graphene_django.types import DjangoObjectType

from loop.models import Loop
from utils.calculations import Calculator
from utils.hausdorff import HausDim


class LoopType(DjangoObjectType):
    class Meta:
        model = Loop


class Query(graphene.ObjectType):
    loop = graphene.Field(LoopType, id=graphene.Int())
    all_loops = graphene.List(LoopType, system=graphene.Int())
    is_assessed = graphene.Boolean(id=graphene.Int())

    def resolve_all_loops(self, context, system=None):
        if system:
            return Loop.objects.filter(system__id=system)
        return Loop.objects.all()

    def resolve_loop(self, context, id=None):
        if id:
            loop = Loop.objects.get(pk=id)
            if loop.calculated is True:
                return loop
            else:
                calculator = Calculator(loop)
                indexes = calculator.calculate_all()
                min_values = indexes.get('min_values', None)
                loop.ise = indexes['ise']
                loop.iae = indexes['iae']
                loop.qe = indexes['qe']
                loop.hre = indexes['hre']
                loop.hde = indexes['hde']
                loop.h0 = indexes['h0']
                loop.h1 = indexes['h1']
                loop.h2 = indexes['h2']
                loop.h3 = indexes['h3']
                loop.cr1 = indexes['cr1']
                loop.cr2 = indexes['cr2']
                loop.gsig = indexes['gsig']
                loop.salf = indexes['salf']
                loop.sgam = indexes['sgam']
                loop.lb = indexes['lb']
                loop.rsig = indexes['rsig']
                loop.gauss = indexes['gauss']
                loop.levy = indexes['levy']
                loop.laplace = indexes['laplace']
                loop.huber = indexes['huber']
                loop.histX = indexes['histX']
                loop.histY = indexes['histY']
                loop.skewness = indexes['skewness']
                loop.kurtosis = indexes['kurtosis']
                loop.crossX1 = [item for item in indexes['crossx1']]
                loop.crossY1 = [item for item in indexes['crossy1']]
                loop.crossX2 = [item for item in indexes['crossx2']]
                loop.crossY2 = [item for item in indexes['crossy2']]
                loop.xx = [item[0] for item in indexes['xx']]
                loop.yy = [item[0] for item in indexes['yy']]
                loop.haa = [item for item in indexes['haa']]
                loop.xp = [item[0] for item in indexes['xp']]
                loop.yp = [item[0] for item in indexes['yp']]
                loop.firstApproximation = list(indexes['first_approximation'])
                loop.secondApproximation = list(indexes['second_approximation'])

                min_attrs = ['minIse', 'minIae', 'minQe', 'minGsig', 'minSgam', 'minLb',
                             'minRsig', 'minHre', 'minHde']
                if min_values:
                    for attr in min_attrs:
                        setattr(loop, attr, min_values[attr])
                else:
                    for attr in min_attrs:
                        setattr(loop, attr, None)

                loop.calculated = True
                loop.save()

                if len(loop.cv) >= 4000:
                    haus_dim = HausDim.get_dim(loop)
                    if haus_dim:
                        loop.hausDim, loop.score = haus_dim
                        loop.save()
                return loop
        return None

    def resolve_is_assessed(self, context, id=None):
        if id:
            loop = Loop.objects.get(pk=id)
            return loop.calculated
        else:
            return None


class DescriptionMutation(graphene.Mutation):
    class Arguments:
        id = graphene.Int()
        description = graphene.String()

    ok = graphene.Boolean()

    def mutate(self, info, id, description):
        try:
            loop = Loop.objects.get(pk=id)
        except Loop.DoesNotExist:
            return DescriptionMutation(ok=False)
        loop.description = description
        loop.save()
        return DescriptionMutation(ok=True)


class Mutation(graphene.ObjectType):
    add_description = DescriptionMutation.Field()
