import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import swarmist as sw
import datetime

problem, bounds = sw.benchmark.sphere()
ndims = 20
max_gen = 1000
trials = 50

st = sw.strategy()
st.param("c1", value=sw.AutoFloat(min=0, max=4.1))
st.param("c2", value=sw.AutoFloat(min=0, max=4.1))
st.param("chi", value=sw.AutoFloat(min=0, max=1))
st.param("cr", value=sw.AutoFloat(min=0, max=1))
st.init(sw.init.random(), size=sw.AutoInteger(min=10, max=100))
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
    .recombinant(sw.recombination.binomial(cr_probability=lambda ctx: ctx.param("cr"))),
)

start_time = datetime.datetime.now()

res = sw.search(
    sw.minimize(problem, bounds, dimensions=ndims),
    sw.until(max_gen=max_gen),
    sw.tune(st, max_gen=trials),
)

duration = (datetime.datetime.now() - start_time).total_seconds()
print(f"End opt after {duration} seconds")
print(f"fit={res.fit}, params={res.parameters}")
