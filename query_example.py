import swarmist as sw

st = sw.strategy()
st.param("c1", value=0, min=0, max=2.0)
st.param("c2", value=0, min=0, max=2.0)
st.init(sw.init.random)
st.topology(sw.topology.gbest)
st.pipeline(
    sw.select(sw.all()).update(
        gbest=sw.swarm.best(),
        pbest=sw.agent.best(),
        velocity=lambda ctx: (
            ctx.agent.delta +
            ctx.param("c1") * ctx.random.rand() * (ctx.get("gbest").diff(ctx.agent.best)) +
            ctx.param("c2") * ctx.param("r2") * (ctx.get("gbest").diff(ctx.agent.best))
        ),
        pos=lambda ctx: ctx.agent.pos + ctx("velocity")
    ).recombinant(
        sw.recombination.replace_all()
    ).where(lambda ctx: ctx.improved),
    sw.select(sw.all()).update(
        pos=sw.init.random(),
        recombination=sw.recombination.get_new()
    )
)

sw.search(
    sw.space(
        dimensions=sw.dimensions(30),
        bounds=sw.bounded(10, 20),
        constraints=sw.constrained_by(lambda pos: pos ),
        fit_function=sw.minimize(lambda x: sum(x**2))
    ),
    sw.until(max_evals=50000),
    sw.using(st())
)