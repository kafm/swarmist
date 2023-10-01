import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmist as sw

problem, bounds = sw.benchmark.sphere()
numDimensions = 20
maxGenerations = 1000
populationSize = 40

st = sw.strategy()
st.param("c1", value=2.05)
st.param("c2", value=2.05)
st.param("chi", value=0.729)
st.init(sw.init.random(), size=populationSize)
st.topology(sw.topology.gbest())
st.pipeline(
    sw.select(sw.all())
    .update(
        gbest=sw.swarm.best(),
        pbest=sw.agent.best(),
        velocity=lambda ctx: ctx.param("chi")
        * (
            ctx.agent.delta
            + ctx.param("c1") * ctx.random.rand() * (ctx.get("pbest") - ctx.agent.pos)
            + ctx.param("c2") * ctx.random.rand() * (ctx.get("gbest") - ctx.agent.pos)
        ),
        pos=lambda ctx: ctx.agent.pos + ctx.get("velocity"),
    )
    .recombinant(sw.recombination.replace_all()),
)

res = sw.search(
    sw.minimize(problem, bounds, dimensions=numDimensions),
    sw.until(max_gen=maxGenerations),
    sw.using(st),
)
print(res.best[-1])
