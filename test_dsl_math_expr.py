from swarmist.sdl.parser import Parser
import swarmist as sw

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
    .recombinant(sw.recombination.replace_all())
)

    # MINIMIZE X**2
    # CONSTRAINED BY (
    #     X[1] < X[2]   
    # )


expression = """
SEARCH(
    VAR X SIZE(10) BOUNDED BY (-5.12, 5.12) 
    VAR Y SIZE(10) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X**2) + SUM(Y**2)
    SUBJECT TO (
        SUM(X)=SUM(Y)
        SUM(X) < SUM(Y-1)
    ) WITH COEFFICIENT .7
)
USING (
    PARAMETERS (
        C1 = 2.05 BOUNDED BY (0, 8)
        C2 = 2.05 BOUNDED BY (0, 8) 
        CHI = 0.7298 BOUNDED BY (0, 1)
    )
    POPULATION SIZE(40) INIT RANDOM_UNIFORM()
    SELECT ALL (
        UPDATE (
            VELOCITY= PARAM(CHI) * (
                DELTA 
                + PARAM(C1) * RANDOM() * (BEST-POS)
                + PARAM(C2) * RANDOM() * (SWARM_BEST()-POS)
            )
            POS = POS + VELOCITY
        ) 
    )
)
UNTIL (
    GENERATION = 1000
)
"""
#.format(bounds=bounds)
query = Parser().parse(expression) #, start="strategy_expr"
query()
# strategy._pipeline_builders[0].update(
#         gbest=sw.swarm.best(),
#         pbest=sw.agent.best(),
#         velocity=lambda ctx: ctx.param("CHI") * (
#             ctx.agent.delta
#             + ctx.param("C1") * ctx.random.rand() * (ctx.get("pbest") - ctx.agent.pos)
#             + ctx.param("C1") * ctx.random.rand() * (ctx.get("gbest") - ctx.agent.pos)
#         ),
#         pos=lambda ctx: ctx.agent.pos + ctx.get("velocity"),
# ) 
# res = sw.search(
#     sw.minimize(
#         problem, bounds, dimensions=numDimensions
#     ),#.constrained_by(lambda pos: pos),
#     sw.until(**stop_condition),
#     sw.using(strategy),
# )

# print("=========================NORMAL=========================")

# res_normal = res = sw.search(
#     sw.minimize(
#         problem, bounds, dimensions=numDimensions
#     ),#.constrained_by(lambda pos: pos),
#     sw.until(max_gen=maxGenerations),
#     sw.using(st),
# )