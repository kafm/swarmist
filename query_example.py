import swarmist as sw
from swarmist_old.PSO import PSO, SearchResult
from swarmist_bck.algos.pso import Pso
problem, bounds = sw.benchmark.sphere()
numDimensions = 20
maxGenerations = 1000
populationSize = 40

st = sw.strategy()
st.param("c1", value=2.05, min=0, max=4.1)
st.param("c2", value=2.05, min=0, max=4.1)
st.param("chi", value=0.729, min=0, max=1)
st.init(sw.init.random(), size=populationSize)
st.topology(sw.topology.gbest())
st.pipeline(
    sw.select(sw.all())
    .update(
        gbest=sw.swarm.best(),
        pbest=sw.agent.best(),
        velocity=lambda ctx: ctx.param("chi") * (
            ctx.agent.delta
            + ctx.param("c1") * ctx.random.rand() * (ctx.get("pbest") - ctx.agent.pos)
            + ctx.param("c2") * ctx.random.rand() * (ctx.get("gbest") - ctx.agent.pos)
        ),
        pos=lambda ctx: ctx.agent.pos + ctx.get("velocity"),
    )
    .recombinant(sw.recombination.replace_all()),
    # sw.select(sw.all()).update(
    #     pos=sw.init.random(),
    #     recombination=sw.recombination.get_new()
    # )
)

res_old: SearchResult = PSO(
    fitnessFunction=problem,
    bounds = bounds,
    numDimensions = numDimensions,
    populationSize = populationSize,
    maxGenerations = maxGenerations,
)

res = sw.search(
    sw.minimize(
        problem, bounds, dimensions=numDimensions
    ),#.constrained_by(lambda pos: pos),
    sw.until(max_gen=maxGenerations),
    sw.using(st),
)

print(f"old={res_old.best.fitness}")
print(f"new={min(res, key=lambda x: x.fit)}")
