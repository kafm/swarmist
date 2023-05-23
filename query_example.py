import swarmist_bck as sw

st = sw.strategy()
st.param("c1", default=0, min=0, max=2.0)
st.param("c2", default=0, min=0, max=2.0)
st.init(sw.initialization.random)
st.topology(sw.topology.g_best)
st.pipeline(
    sw.select(sw.all()).update(
        velocity=lambda ctx: ctx.velocity + ctx.param("c1") * ctx.r1 * (ctx.pbest - ctx.position) + ctx.param("c2") * ctx.r2 * (ctx.gbest - ctx.position),
        pos=lambda ctx: ctx.position + ctx.velocity,
        recombination=sw.replace_all(),   
        where=lambda ctx: ctx.fitness < ctx.pbest_fitness
    ),
    sw.select(sw.all()).update(sw.pso(sw.particle().velocity(sw.velocity().clerc(c1="c1", c2="c2")))),
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