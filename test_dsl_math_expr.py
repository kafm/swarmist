import swarmist as sw

# problem, bounds = sw.benchmark.sphere()
# numDimensions = 20
# maxGenerations = 1000
# populationSize = 40
# st = sw.strategy()
# st.param("c1", value=2.05, min=0, max=4.1)
# st.param("c2", value=2.05, min=0, max=4.1)
# st.param("chi", value=0.729, min=0, max=1)
# st.init(sw.init.random(), size=populationSize)
# st.topology(sw.topology.gbest())
# st.pipeline(
#     sw.select(sw.all())
#     .update(
#         gbest=sw.swarm.best(),
#         pbest=sw.agent.best(),
#         velocity=lambda ctx: ctx.param("chi") * (
#             ctx.agent.delta
#             + ctx.param("c1") * ctx.random.rand() * (ctx.get("pbest") - ctx.agent.pos)
#             + ctx.param("c2") * ctx.random.rand() * (ctx.get("gbest") - ctx.agent.pos)
#         ),
#         pos=lambda ctx: ctx.agent.pos + ctx.get("velocity"),
#     )
#     .recombinant(sw.recombination.replace_all())
# )

expression = """
SEARCH(
    VAR X SIZE(20) BOUNDED BY (-5.12, 5.12) 
    MINIMIZE SUM(X**2)
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
            NEIGHBORS = NEIGHBORHOOD()
            PHI = PARAM(C1) + PARAM(C2)
            N = COUNT(NEIGHBORS)
            W = RANDOM(SIZE=N)
            SCT = RANDOM() * (PHI/N) * W * ( ((NEIGHBORS * W)/W) - POS )
            VELOCITY = PARAM(CHI) * (DELTA + SCT)
            POS = POS + VELOCITY
        ) 
    )
)
UNTIL (
    GENERATION = 1000
)
"""
#.format(bounds=bounds)
results = sw.sdl.execute(expression) #, start="strategy_expr"
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